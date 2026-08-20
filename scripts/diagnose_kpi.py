#!/usr/bin/env python3
"""KPI kaybının nedenini kırar: zaman mı, metin mi, sahne okuması mı?

`evaluate_kpi.py` skoru söyler, bu script nedenini söyler. Yeni bir model veya
prompt denedikten sonra nereye dokunacağımızı 5 dakikada görmek için.

    python scripts/diagnose_kpi.py                       # varsayılan: kpi_wide_7b
    python scripts/diagnose_kpi.py --gold data/exports/kpi_wide_7b_gold.json \
        --pred data/predictions_wide --out-csv data/exports/kpi_teshis.csv

Kayıp nedenleri:
    zaman_kacti  : ±2 sn içinde hiç tahmin olayı yok (kare seçimi / hizalama sorunu)
    metin_kacti  : zaman tutuyor ama kelime örtüşmesi eşiğin altında (dil / terminoloji)
    olay_yok     : model hiç olay üretmemiş
    eslesti      : gold olayı yakalanmış
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.demo_pipeline import use_utf8_stdout  # noqa: E402
from utils.kpi import TEXT_JACCARD_MIN, jaccard, spec_of  # noqa: E402
from utils.spec_output import mmss_to_seconds  # noqa: E402
from utils.splits import filter_by_split  # noqa: E402

TOLERANCE_SEC = 2
GOLD_DEFAULT = ROOT / "data" / "exports" / "kpi_wide_7b_gold.json"
PRED_DEFAULT = ROOT / "data" / "predictions_wide"
OUT_CSV = ROOT / "data" / "exports" / "kpi_teshis.csv"

FIELDS = [
    "filename",
    "category",
    "gold_time",
    "gold_event",
    "durum",
    "en_yakin_zaman",
    "zaman_farki_sn",
    "en_iyi_jaccard",
    "eslesen_tahmin",
    "ozet_jaccard",
    "risk_gold",
    "risk_pred",
]


def load_pred_index(pred_path: Path) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if pred_path.is_file():
        data = json.loads(pred_path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [data]
    else:
        for file in sorted(pred_path.glob("*.json")):
            if file.name.endswith("_spec.json"):
                continue
            rows.append(json.loads(file.read_text(encoding="utf-8")))

    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in (row.get("filename"), row.get("video_id")):
            if key:
                indexed[str(key)] = row
                indexed[Path(str(key)).stem] = row
    return indexed


def diagnose_event(gold_event: dict[str, Any], pred_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Tek gold olayı için en iyi tahmin eşleşmesini ve kayıp nedenini bulur."""
    gold_sec = mmss_to_seconds(gold_event.get("time"))
    if not pred_events:
        return {"durum": "olay_yok", "en_yakin_zaman": "", "zaman_farki_sn": "", "en_iyi_jaccard": 0.0, "eslesen_tahmin": ""}

    in_window = [
        pred
        for pred in pred_events
        if abs(mmss_to_seconds(pred.get("time")) - gold_sec) <= TOLERANCE_SEC
    ]
    pool = in_window or pred_events
    best = max(pool, key=lambda pred: jaccard(gold_event.get("event"), pred.get("event")))
    best_score = jaccard(gold_event.get("event"), best.get("event"))
    delta = abs(mmss_to_seconds(best.get("time")) - gold_sec)

    if not in_window:
        durum = "zaman_kacti"
    elif best_score >= TEXT_JACCARD_MIN:
        durum = "eslesti"
    else:
        durum = "metin_kacti"

    return {
        "durum": durum,
        "en_yakin_zaman": best.get("time"),
        "zaman_farki_sn": delta,
        "en_iyi_jaccard": round(best_score, 3),
        "eslesen_tahmin": (best.get("event") or "")[:120],
    }


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    parser.add_argument("--pred", type=Path, default=PRED_DEFAULT)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--split", default="hepsi", choices=["hepsi", "dev", "test"])
    parser.add_argument("--detail", action="store_true", help="Her video için gold/pred metinleri bas")
    args = parser.parse_args()

    gold_rows = filter_by_split(
        json.loads(args.gold.read_text(encoding="utf-8")), args.split
    )
    preds = load_pred_index(args.pred)

    rows: list[dict[str, Any]] = []
    summary_scores: list[float] = []
    risk_miss: list[tuple[str, str, str]] = []
    missing: list[str] = []

    for gold in gold_rows:
        name = gold.get("filename") or ""
        pred = preds.get(name) or preds.get(Path(name).stem)
        gold_spec = spec_of(gold)
        if pred is None:
            missing.append(name)
            continue
        pred_spec = spec_of(pred)
        summary_score = jaccard(gold_spec["summary"], pred_spec["summary"])
        summary_scores.append(summary_score)
        if gold_spec["risk"] != pred_spec["risk"]:
            risk_miss.append((name, gold_spec["risk"], pred_spec["risk"]))

        if args.detail:
            print("=" * 72)
            print(f"{name[:64]} | {gold_spec['category']}")
            print(f"  gold özet: {gold_spec['summary'][:110]}")
            print(f"  pred özet: {pred_spec['summary'][:110]}  (jaccard {summary_score:.2f})")

        for gold_event in gold_spec["events"]:
            finding = diagnose_event(gold_event, pred_spec["events"])
            rows.append(
                {
                    "filename": name,
                    "category": gold_spec["category"],
                    "gold_time": gold_event.get("time"),
                    "gold_event": (gold_event.get("event") or "")[:120],
                    "ozet_jaccard": round(summary_score, 3),
                    "risk_gold": gold_spec["risk"],
                    "risk_pred": pred_spec["risk"],
                    **finding,
                }
            )
            if args.detail:
                print(
                    f"  {gold_event.get('time')} -> {finding['durum']} "
                    f"(en iyi jaccard {finding['en_iyi_jaccard']}, "
                    f"zaman farkı {finding['zaman_farki_sn']}s)"
                )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in FIELDS} for row in rows)

    counts = Counter(row["durum"] for row in rows)
    total = sum(counts.values()) or 1
    summary_pass = sum(1 for score in summary_scores if score >= TEXT_JACCARD_MIN)

    print("\n" + "=" * 60)
    print("KPI TEŞHİSİ")
    print("=" * 60)
    print(f"  Video: {len(summary_scores)}  |  gold olay: {total}  |  tahmini olmayan: {len(missing)}")
    for durum in ("eslesti", "metin_kacti", "zaman_kacti", "olay_yok"):
        n = counts.get(durum, 0)
        print(f"  {durum:12} : {n:3}  ({n / total:.0%})")

    near_miss_text = [row for row in rows if row["durum"] == "metin_kacti"]
    if near_miss_text:
        ortalama = sum(row["en_iyi_jaccard"] for row in near_miss_text) / len(near_miss_text)
        print(
            f"\n  Metin kaçıranların ortalama jaccard'ı: {ortalama:.3f} "
            f"(eşik {TEXT_JACCARD_MIN}) -> terminoloji / ifade işi"
        )
    print(
        f"  Özet eşiği geçen video: {summary_pass}/{len(summary_scores)}  "
        f"(ortalama jaccard {sum(summary_scores) / max(len(summary_scores), 1):.3f})"
    )
    if risk_miss:
        print(f"\n  Risk uyuşmayan {len(risk_miss)} video:")
        for name, gold_risk, pred_risk in risk_miss[:10]:
            print(f"    {name[:50]:50} gold={gold_risk} pred={pred_risk}")
    if missing:
        print(f"\n  Tahmini olmayan {len(missing)} video: {', '.join(m[:30] for m in missing[:5])}")
    print(f"\nSatır satır tablo: {args.out_csv}")


if __name__ == "__main__":
    main()
