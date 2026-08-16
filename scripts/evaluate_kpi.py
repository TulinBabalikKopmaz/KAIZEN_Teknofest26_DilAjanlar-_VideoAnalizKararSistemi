#!/usr/bin/env python3
"""Gold etiket vs sistem tahmini KPI raporu.

Örnek:
  python scripts/evaluate_kpi.py
  python scripts/evaluate_kpi.py --pred data/predictions
  python scripts/evaluate_kpi.py --pred data/exports/gold_labels_hepsi.json
      # (kendisiyle kıyas: skorlar 1.0 olmalı, duman testi)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.kpi import aggregate, identity_keys, score_video, spec_of

GOLD_DEFAULT = ROOT / "data" / "exports" / "gold_labels_hepsi.json"
OUT_JSON = ROOT / "data" / "exports" / "kpi_report.json"
OUT_CSV = ROOT / "data" / "exports" / "kpi_ozet.csv"


def load_rows(path: Path) -> list[dict]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return [data]
    rows = []
    for file in sorted(path.glob("*.json")):
        if file.name.endswith("_spec.json"):
            continue
        rows.append(json.loads(file.read_text(encoding="utf-8")))
    return rows


def index_preds(rows: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        spec = spec_of(row)
        for key in identity_keys({**row, **spec}):
            indexed[key] = row
    return indexed


def find_pred(gold: dict, indexed: dict[str, dict]) -> dict | None:
    for key in identity_keys(gold):
        if key in indexed:
            return indexed[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    parser.add_argument("--pred", type=Path, default=GOLD_DEFAULT, help="JSON liste, tek JSON veya klasör")
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    gold_rows = load_rows(args.gold)
    pred_rows = load_rows(args.pred)
    indexed = index_preds(pred_rows)

    per_video = [score_video(g, find_pred(g, indexed)) for g in gold_rows]
    summary = aggregate(per_video)
    summary["gold_path"] = str(args.gold)
    summary["pred_path"] = str(args.pred)
    self_test = args.gold.resolve() == args.pred.resolve()
    summary["note"] = (
        "Gold kendisiyle kıyaslandı (duman testi). Gerçek skor için --pred data/predictions kullanın."
        if self_test
        else "Sistem tahmini vs gold."
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"summary": summary, "videos": per_video}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with args.out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "filename",
                "category",
                "risk_gold",
                "risk_pred",
                "risk_ok",
                "event_recall",
                "critical_hit",
                "false_alarm",
                "summary_ok",
                "action_ok",
                "missing_pred",
            ],
        )
        writer.writeheader()
        for row in per_video:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    print("KPI (şartname metrikleri)")
    print(f"  Video sayısı        : {summary['n_video']}")
    print(f"  Risk doğruluğu      : {summary['risk_accuracy']:.0%}")
    print(f"  Olay yakalama       : {summary['event_recall']:.0%}  (±2 sn)")
    print(f"  Kritik olay yakalama: {summary['critical_recall']:.0%}  (kaza/near miss)")
    print(f"  Normalde yanlış kaza: {summary['false_alarm_rate']:.0%}")
    print(f"  Özet benzerliği     : {summary['summary_ok_rate']:.0%}")
    print(f"  Aksiyon doluluğu    : {summary['action_ok_rate']:.0%}")
    print(f"  Tahmini olmayan     : {summary['missing_predictions']}")
    print(f"  {summary['note']}")
    print(f"Yazıldı: {args.out_json}")
    print(f"Yazıldı: {args.out_csv}")


if __name__ == "__main__":
    main()
