#!/usr/bin/env python3
"""Mevcut prediction JSON'larına refine_label uygular (VLM çağrısı yok).

Ollama crash sonrası --resume ile biriken eski skorlara FA dampen / risk
kurallarını işletmek için. Sonra evaluate_kpi çalıştırır.

Örnek:
  python scripts/re_refine_preds.py --pred-dir data/predictions_wide_all
  python scripts/re_refine_preds.py --pred-dir data/predictions_wide_all --no-eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from auto_label_qwen import competition_view
from extract_frames import safe_id
from utils.risk_rules import refine_label
from utils.scene_evidence import analyze_video

GOLD_PATH = ROOT / "data" / "exports" / "gold_labels_hepsi.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=ROOT / "data" / "predictions_wide_all",
    )
    parser.add_argument(
        "--videos",
        type=Path,
        default=ROOT / "data" / "videos",
        help="SceneEvidence için video kökü",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=GOLD_PATH,
        help="evaluate için gold listesi (veya kpi_wide_*_gold.json)",
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--out-tag",
        default="refined",
        help="Rapor dosya adı etiketi",
    )
    args = parser.parse_args()

    if not args.pred_dir.is_dir():
        raise SystemExit(f"pred-dir yok: {args.pred_dir}")

    videos = {
        p.name: p
        for p in args.videos.rglob("*")
        if p.suffix.lower() == ".mp4"
    }
    # video_id / filename ile de bul
    by_stem = {p.stem: p for p in videos.values()}
    by_safe = {safe_id(p): p for p in videos.values()}

    pred_files = sorted(
        p for p in args.pred_dir.glob("*.json") if not p.name.endswith("_spec.json")
    )
    if not pred_files:
        raise SystemExit(f"JSON yok: {args.pred_dir}")

    print(f"Re-refine başlıyor: {len(pred_files)} dosya  ({args.pred_dir})", flush=True)
    print("(İlk videoda YOLO yüklenir; 1–2 dk sessiz kalabilir.)", flush=True)

    n_ok = 0
    n_changed = 0
    for i, path in enumerate(pred_files, start=1):
        label = json.loads(path.read_text(encoding="utf-8"))
        name = label.get("filename") or ""
        vid = label.get("video_id") or path.stem
        video = videos.get(name) or by_safe.get(vid) or by_stem.get(Path(name).stem)
        evidence = None
        if video is not None:
            try:
                evidence = analyze_video(video)
            except Exception as exc:
                print(f"  [{i}/{len(pred_files)}] kanıt atlandı {name}: {exc}", flush=True)
        before = (label.get("risk"), label.get("category"))
        refined = refine_label(label, evidence)
        after = (refined.get("risk"), refined.get("category"))
        path.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
        spec_path = path.with_name(path.stem + "_spec.json")
        spec_path.write_text(
            json.dumps(competition_view(refined), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        n_ok += 1
        if before != after:
            n_changed += 1
            print(f"  [{i}/{len(pred_files)}] {name or vid}: {before} → {after}", flush=True)
        elif i == 1 or i % 10 == 0 or i == len(pred_files):
            print(f"  [{i}/{len(pred_files)}] OK {name or vid}", flush=True)

    print(f"Re-refine bitti: {n_ok} dosya, {n_changed} değişti  ({args.pred_dir})", flush=True)

    if args.no_eval:
        return

    gold = args.gold
    if not gold.exists():
        # Son full koşunun gold alt kümesi
        alt = ROOT / "data" / "exports" / "kpi_wide_qwen2_5vl_7b_gold.json"
        gold = alt if alt.exists() else GOLD_PATH

    tag = args.out_tag
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_kpi.py"),
                "--gold",
                str(gold),
                "--pred",
                str(args.pred_dir),
                "--out-json",
                str(ROOT / "data" / "exports" / f"kpi_wide_{tag}_report.json"),
                "--out-csv",
                str(ROOT / "data" / "exports" / f"kpi_wide_{tag}_ozet.csv"),
            ]
        )
    )


if __name__ == "__main__":
    main()
