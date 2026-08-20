#!/usr/bin/env python3
"""Eski --resume tahminlerini siler; aynı videolar yeni prompt ile yeniden koşulsun.

Colab:

    python scripts/purge_stale_preds.py --pred-dir data/predictions_wide

Drive symlink olduğu için dosyalar MyDrive/KAIZEN_KPI/data/predictions_wide altında da silinir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_frames import safe_id  # noqa: E402

from utils.demo_pipeline import use_utf8_stdout  # noqa: E402

# 6a çıktısında "resume: mevcut tahmin var, atlandı" yazan 13 video
STALE_FILENAMES = [
    "2uuwpx1ykPQ_trim_0.mp4",
    "3XOG - Workplace accidents happen all the time! All we can do is try learn f... [1615153821189013504].mp4",
    "Beylikdüzü.anlık - Beylikdüzü Yeşil Flamingo Sitesi’nde yaşanan, Aydın Dirikolu’nun (61)... [2088241217045774336].mp4",
    "_XXq1kmmBYY_trim_6.mp4",
    "001e53453441935632ae_run_1_seed_1288693302.ceiling_01.rgb.mp4",
    "00f2bbb80d7badad0134_run_25_seed_750796930.ceiling_00.rgb.mp4",
    "Forklift ear miss pedestrian not watching almost gets hit twice [pLjVBBW_g7c].mp4",
    "How NOT to UNLOAD a truck [t0abC-2oucI].mp4",
    "Near miss walk into the path of a forklift [EHyWjeJi1Oc].mp4",
    "0_tr5.mp4",
    "clip_HME_Work_Shop_20251111_000740_0030.mp4",
    "clip_TAR_Plant_DC-Floor_left_20251115_000000_0000.mp4",
    "clip_Tippler_1&2_CP_IP_Cam_20251127_000108_0002.mp4",
    "clip_Tippler_1&2_WT-1_Load_Side_20260403_003515_0055.mp4",
]


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=ROOT / "data" / "predictions_wide",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.pred_dir.exists():
        raise SystemExit(f"Klasör yok: {args.pred_dir}")

    deleted = 0
    missing = 0
    for name in STALE_FILENAMES:
        vid = safe_id(Path(name))
        for suffix in (".json", "_spec.json"):
            path = args.pred_dir / f"{vid}{suffix}"
            if not path.exists():
                # uzun isimler 80 karakterde kesilmiş olabilir; başla eşle
                hits = [
                    p
                    for p in args.pred_dir.glob(f"{vid[:40]}*{suffix}")
                    if not (suffix == ".json" and p.name.endswith("_spec.json"))
                ]
                if not hits:
                    print(f"  yok: {vid}{suffix}")
                    missing += 1
                    continue
                path = hits[0]
            print(f"  sil {'(dry) ' if args.dry_run else ''}{path.name}")
            if not args.dry_run:
                path.unlink()
            deleted += 1

    print(f"\nSilinen: {deleted} dosya  |  bulunamayan: {missing}")
    print("Sonra: python scripts/run_kpi_wide.py --n all --split dev --resume ...")


if __name__ == "__main__":
    main()
