#!/usr/bin/env python3
"""Demo çekirdeği CLI: bir video + bir prompt → zaman damgalı İSG cevabı.

    python scripts/analyze_video.py --video demo.mp4 \
        --prompt "Bu videoda iş kazası var mı, kaçıncı saniyede?"

    python scripts/analyze_video.py --video demo.mp4 --fast --json

Çıktı data/demo_runs/<ad>/ altına yazılır: result.json, spec.json, report.txt, frames/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import config  # noqa: E402
from utils.demo_pipeline import (  # noqa: E402
    DEFAULT_PROMPT,
    run_demo_analysis,
    use_utf8_stdout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Jürinin verdiği serbest metin soru",
    )
    parser.add_argument(
        "--provider",
        choices=config.PROVIDERS,
        default="",
        help="Env'deki PROVIDER'ı geçici olarak ezer",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 = DEMO_MAX_FRAMES")
    parser.add_argument("--every-sec", type=float, default=0.5)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Hızlı mod: YOLO kanıtı, ikinci bakış ve RAG kapalı",
    )
    parser.add_argument("--no-rag", action="store_true", help="Mevzuat referansını atla")
    parser.add_argument("--no-second-look", action="store_true")
    parser.add_argument(
        "--time-budget",
        type=float,
        default=600.0,
        help="Bu saniyeyi aşarsa ikinci bakış / eleştirmen / RAG atlanır (jüri yolu: uzun tut)",
    )
    parser.add_argument("--run-name", default="", help="Çıktı klasörü adı (varsayılan: video adı)")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true", help="Diske yazma")
    parser.add_argument("--json", action="store_true", help="Sonunda ham JSON bas")
    return parser


def main() -> None:
    use_utf8_stdout()
    args = build_parser().parse_args()
    if args.provider:
        os.environ["PROVIDER"] = args.provider

    print(config.describe())
    result = asyncio.run(
        run_demo_analysis(
            args.video,
            args.prompt,
            max_frames=args.max_frames or None,
            every_sec=args.every_sec,
            fast=args.fast or None,
            use_rag=not args.no_rag,
            use_second_look=not args.no_second_look,
            time_budget_s=args.time_budget,
            save=not args.no_save,
            out_root=args.out_root,
            run_name=args.run_name,
        )
    )

    print()
    print(result.report_text())
    if args.json:
        print()
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
