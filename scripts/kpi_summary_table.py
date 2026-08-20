#!/usr/bin/env python3
"""KPI raporlarını tek karşılaştırma tablosuna indirger (sunum slaytı için).

    python scripts/kpi_summary_table.py
    python scripts/kpi_summary_table.py --reports data/exports/kpi_wide_*_report.json

Çıktı: data/exports/kpi_final_ozet.csv + ekrana hedef karşılaştırması.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPORTS = ROOT / "data" / "exports"
OUT_CSV = EXPORTS / "kpi_final_ozet.csv"

# Şartname metrikleri için kendi hedeflerimiz (sunumda "hedef vs gerçek" satırı)
TARGETS: dict[str, float] = {
    "risk_accuracy": 0.70,
    "event_recall": 0.50,
    "critical_recall": 0.90,
    "false_alarm_rate": 0.0,
    "summary_ok_rate": 0.40,
    "action_ok_rate": 1.00,
}

METRIC_LABELS: dict[str, str] = {
    "risk_accuracy": "Risk doğruluğu",
    "event_recall": "Olay yakalama (±2 sn)",
    "critical_recall": "Kritik olay yakalama",
    "false_alarm_rate": "Normalde yanlış kaza",
    "summary_ok_rate": "Özet benzerliği",
    "action_ok_rate": "Aksiyon doluluğu",
}

FIELDS = [
    "run",
    "n_video",
    "risk_accuracy",
    "event_recall",
    "critical_recall",
    "false_alarm_rate",
    "summary_ok_rate",
    "action_ok_rate",
    "missing_predictions",
    "report",
]


def run_name(path: Path) -> str:
    return path.stem.removeprefix("kpi_").removesuffix("_report")


def is_self_test(summary: dict) -> bool:
    """Gold'un kendisiyle kıyaslandığı duman testi tabloya girmemeli (hepsi %100)."""
    if "duman testi" in str(summary.get("note", "")):
        return True
    gold, pred = summary.get("gold_path"), summary.get("pred_path")
    return bool(gold and pred and Path(gold).resolve() == Path(pred).resolve())


def load_summary(path: Path, keep_self_test: bool = False) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  atlandı ({path.name}): {exc}")
        return None
    summary = data.get("summary") if isinstance(data, dict) else None
    if not summary:
        return None
    if not summary.get("n_video"):
        print(f"  atlandı ({path.name}): tahmin yok")
        return None
    if is_self_test(summary) and not keep_self_test:
        print(f"  atlandı ({path.name}): gold kendisiyle kıyaslanmış (duman testi)")
        return None
    row = {key: summary.get(key) for key in FIELDS if key in summary}
    row["run"] = run_name(path)
    row["report"] = str(path.relative_to(ROOT))
    row["n_video"] = summary.get("n_video")
    row["missing_predictions"] = summary.get("missing_predictions")
    return row


def meets(metric: str, value: float | None) -> bool:
    if value is None:
        return False
    target = TARGETS[metric]
    return value <= target if metric == "false_alarm_rate" else value >= target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        nargs="*",
        type=Path,
        default=None,
        help="KPI rapor JSON'ları (varsayılan: data/exports/*_report.json)",
    )
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument(
        "--best",
        default="",
        help="Hedef karşılaştırması için koşu adı (varsayılan: en yüksek risk doğruluğu)",
    )
    parser.add_argument(
        "--keep-self-test",
        action="store_true",
        help="Gold-gold duman testi raporlarını da tabloya al",
    )
    args = parser.parse_args()

    reports = args.reports or sorted(EXPORTS.glob("*_report.json"))
    if not reports:
        raise SystemExit(f"Rapor bulunamadı: {EXPORTS}")

    rows = [
        row
        for row in (load_summary(path, args.keep_self_test) for path in reports)
        if row
    ]
    if not rows:
        raise SystemExit("Hiçbir raporda summary alanı yok.")
    rows.sort(key=lambda r: (r.get("risk_accuracy") or 0, r.get("event_recall") or 0), reverse=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDS})

    print(f"{len(rows)} koşu tabloya yazıldı: {args.out_csv}\n")
    header = f"{'koşu':28} {'n':>4}  " + "  ".join(f"{key[:9]:>9}" for key in TARGETS)
    print(header)
    for row in rows:
        cells = []
        for key in TARGETS:
            value = row.get(key)
            cells.append(f"{value:>8.0%} " if isinstance(value, (int, float)) else f"{'-':>9}")
        print(f"{row['run'][:28]:28} {row.get('n_video') or 0:>4}  " + " ".join(cells))

    best = next((r for r in rows if r["run"] == args.best), rows[0])
    print(f"\nHedef karşılaştırması ({best['run']}, n={best.get('n_video')}):")
    for metric, label in METRIC_LABELS.items():
        value = best.get(metric)
        target = TARGETS[metric]
        state = "TAMAM" if meets(metric, value) else "EKSİK"
        shown = f"{value:.0%}" if isinstance(value, (int, float)) else "-"
        print(f"  {label:24} {shown:>6}  (hedef {target:.0%})  {state}")


if __name__ == "__main__":
    main()
