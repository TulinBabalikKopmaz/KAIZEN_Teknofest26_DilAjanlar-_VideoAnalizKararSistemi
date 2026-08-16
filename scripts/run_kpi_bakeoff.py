#!/usr/bin/env python3
"""Birden fazla VLM modelini aynı KPI ayarıyla koşturur; leaderboard yazar.

Örnek:
  python scripts/run_kpi_bakeoff.py --n 18 --models qwen2.5vl:7b
  python scripts/run_kpi_bakeoff.py --n all --models qwen2.5vl:7b,llava:13b

Her model için ayrı pred klasörü kullanır (çakışma olmasın).
Mevcut run_kpi_wide.py'yi subprocess ile çağırır; baseline script'i değiştirmez.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDE = ROOT / "scripts" / "run_kpi_wide.py"
LEADERBOARD = ROOT / "data" / "exports" / "bakeoff_leaderboard.csv"


def safe_name(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in model)


def read_metrics(report_path: Path) -> dict:
    if not report_path.exists():
        return {}
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return data.get("metrics") or data.get("aggregate") or data


def append_leaderboard(row: dict) -> None:
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "model",
        "n_arg",
        "split",
        "seed",
        "n_video",
        "risk_accuracy",
        "event_recall",
        "critical_recall",
        "false_alarm_rate",
        "summary_ok_rate",
        "action_ok_rate",
        "report",
    ]
    write_header = not LEADERBOARD.exists()
    with LEADERBOARD.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="qwen2.5vl:7b",
        help="Virgülle ayrılmış Ollama model listesi",
    )
    parser.add_argument("--n", default="18", help="18 veya all (run_kpi_wide ile aynı)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=("all", "train", "holdout"),
        default="all",
    )
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--every-sec", type=float, default=0.5)
    parser.add_argument(
        "--with-second-look",
        action="store_true",
        help="Second look aç (varsayılan kapalı)",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("--models boş")

    print(f"Bakeoff modeller: {models}")
    print(f"n={args.n} split={args.split} seed={args.seed}")

    for model in models:
        tag = safe_name(model)
        pred_dir = ROOT / "data" / f"predictions_bakeoff_{tag}"
        report = ROOT / "data" / "exports" / f"kpi_wide_{tag}_report.json"
        cmd = [
            sys.executable,
            "-u",
            str(WIDE),
            "--n",
            str(args.n),
            "--seed",
            str(args.seed),
            "--split",
            args.split,
            "--holdout-frac",
            str(args.holdout_frac),
            "--model",
            model,
            "--pred-dir",
            str(pred_dir),
            "--max-frames",
            str(args.max_frames),
            "--every-sec",
            str(args.every_sec),
        ]
        if not args.with_second_look:
            cmd.append("--no-second-look")

        print(f"\n===== BAKEOFF {model} =====")
        print("+", " ".join(cmd), flush=True)
        code = subprocess.call(cmd)
        metrics = read_metrics(report)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "n_arg": args.n,
            "split": args.split,
            "seed": args.seed,
            "n_video": metrics.get("n_video", ""),
            "risk_accuracy": metrics.get("risk_accuracy", ""),
            "event_recall": metrics.get("event_recall", ""),
            "critical_recall": metrics.get("critical_recall", ""),
            "false_alarm_rate": metrics.get("false_alarm_rate", ""),
            "summary_ok_rate": metrics.get("summary_ok_rate", ""),
            "action_ok_rate": metrics.get("action_ok_rate", ""),
            "report": str(report),
        }
        append_leaderboard(row)
        print(f"exit={code}  metrics={ {k: row[k] for k in row if k.endswith('rate') or k.endswith('accuracy') or k.endswith('recall') or k=='n_video'} }")
        print(f"Leaderboard: {LEADERBOARD}")

    print(f"\nBitti. Leaderboard → {LEADERBOARD}")


if __name__ == "__main__":
    main()
