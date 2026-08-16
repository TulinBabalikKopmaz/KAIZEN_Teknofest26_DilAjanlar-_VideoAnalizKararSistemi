#!/usr/bin/env python3
"""KPI için Qwen tahminlerini ayrı klasöre yazar (gold dosyaların üzerine yazmaz).

    python scripts/predict_for_kpi.py --limit 6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6, help="Kaç video (0 = gold listedekilerin hepsi uzun sürer)")
    parser.add_argument("--backend", default="ollama")
    args = parser.parse_args()

    out = ROOT / "data" / "predictions"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "auto_label_qwen.py"),
        "--backend",
        args.backend,
        "--out",
        str(out),
        "--frames",
        str(ROOT / "data" / "frames"),
        "--videos",
        str(ROOT / "data" / "videos"),
        "--overwrite",
        "--max-frames",
        "6",
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    print("Çalışıyor:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
