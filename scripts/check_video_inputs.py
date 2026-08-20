#!/usr/bin/env python3
"""Video giriş dayanıklılığı: jürinin dosyası ne gelirse boru hattı ayakta kalsın.

Model çağırmaz; sadece açma, süre okuma, kare çıkarma ve sensör kanıtı adımlarını
ölçer. Jürinin vereceği dosya mov, avi, dikey veya 4K olabilir; sürprizi burada
yaşamak istiyoruz.

    python scripts/check_video_inputs.py --synthetic          # sentetik varyantlar üret ve dene
    python scripts/check_video_inputs.py --videos data/videos --limit 5
    python scripts/check_video_inputs.py --videos juri_demo.mov --evidence
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_frames import VIDEO_EXTS, extract_video  # noqa: E402

from utils.demo_pipeline import probe_duration, use_utf8_stdout, wake_window  # noqa: E402
from utils.scene_evidence import analyze_video  # noqa: E402

# (ad, genişlik, yükseklik, saniye, fps, uzantı, codec)
SYNTHETIC_CASES: tuple[tuple[str, int, int, int, float, str, str], ...] = (
    ("yatay_mp4_720p", 1280, 720, 8, 25.0, ".mp4", "mp4v"),
    ("dikey_9x16", 720, 1280, 8, 30.0, ".mp4", "mp4v"),
    ("kare_1x1", 720, 720, 6, 25.0, ".mp4", "mp4v"),
    ("dortk_2160p", 3840, 2160, 4, 25.0, ".mp4", "mp4v"),
    ("avi_mjpg", 640, 480, 6, 20.0, ".avi", "MJPG"),
    ("mov_mp4v", 640, 480, 6, 20.0, ".mov", "mp4v"),
    ("uzun_3dk", 640, 360, 180, 20.0, ".mp4", "mp4v"),
    ("cok_kisa_1sn", 640, 360, 1, 25.0, ".mp4", "mp4v"),
    ("dusuk_fps_5", 640, 360, 10, 5.0, ".mp4", "mp4v"),
)


def make_synthetic(case: tuple[str, int, int, int, float, str, str], out_dir: Path) -> Path | None:
    """Hareketli bir çalışan + yaklaşan forklift benzeri sahne üretir."""
    import cv2
    import numpy as np

    name, width, height, seconds, fps, ext, codec = case
    dest = out_dir / f"{name}{ext}"
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        print(f"  {name}: yazıcı açılamadı (codec {codec} yok) — atlandı")
        return None

    total = max(int(seconds * fps), 1)
    peak = int(total * 0.7)
    for i in range(total):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        person_x = int(width * 0.25)
        cv2.rectangle(
            frame,
            (person_x, int(height * 0.45)),
            (person_x + max(width // 24, 4), int(height * 0.8)),
            (200, 200, 200),
            -1,
        )
        progress = min(abs(i - peak) / max(total * 0.2, 1), 1.0)
        vehicle_x = int(width * (0.9 - 0.55 * (1 - progress)))
        cv2.rectangle(
            frame,
            (vehicle_x, int(height * 0.5)),
            (vehicle_x + max(width // 10, 8), int(height * 0.85)),
            (30, 110, 220),
            -1,
        )
        writer.write(frame)
    writer.release()
    return dest if dest.exists() and dest.stat().st_size > 0 else None


def check_one(video: Path, out_root: Path, max_frames: int, with_evidence: bool) -> dict[str, Any]:
    row: dict[str, Any] = {"video": video.name, "boyut_mb": round(video.stat().st_size / 1e6, 1)}
    started = perf_counter()
    duration = probe_duration(video)
    row["sure_sn"] = round(duration, 1)
    if duration <= 0:
        row["durum"] = "AÇILAMADI"
        return row

    try:
        window = None
        if duration > 60:
            evidence = analyze_video(video, 12, use_yolo=False)
            window = wake_window(evidence, duration)
            row["pencere"] = f"{window[0]:.0f}-{window[1]:.0f}s" if window else "-"
        meta = extract_video(video, out_root, 0.5, max_frames, window=window)
        row["kare"] = len(meta["frames"])
        row["kare_zamanlari"] = ",".join(f["time"] for f in meta["frames"])
        row["cozunurluk"] = ""
        first = Path(meta["frames"][0]["path"])
        if first.exists():
            import cv2

            img = cv2.imread(str(first))
            if img is not None:
                row["cozunurluk"] = f"{img.shape[1]}x{img.shape[0]}"
        row["durum"] = "TAMAM"
    except Exception as exc:
        row["durum"] = f"HATA: {exc}"[:80]
        row["kare"] = 0

    if with_evidence and row["durum"] == "TAMAM":
        evidence = analyze_video(video, 16, use_yolo=False)
        row["hareket_tepesi"] = (
            f"{evidence.motion_peak_sec:.1f}s" if evidence.motion_peak_sec is not None else "-"
        )
        row["hareket_yuksek"] = evidence.motion_elevated

    row["gecen_sn"] = round(perf_counter() - started, 1)
    return row


def collect(inputs: list[Path], limit: int) -> list[Path]:
    videos: list[Path] = []
    for item in inputs:
        if item.is_dir():
            videos.extend(p for p in sorted(item.rglob("*")) if p.suffix.lower() in VIDEO_EXTS)
        elif item.suffix.lower() in VIDEO_EXTS:
            videos.append(item)
    return videos[:limit] if limit else videos


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print(f"{'video':26} {'süre':>7} {'kare':>5} {'çözünürlük':>12} {'geçen':>7}  durum")
    print("-" * 78)
    for row in rows:
        print(
            f"{str(row['video'])[:26]:26} "
            f"{row.get('sure_sn', 0):>6}s "
            f"{row.get('kare', 0):>5} "
            f"{str(row.get('cozunurluk', '-')):>12} "
            f"{row.get('gecen_sn', 0):>6}s  "
            f"{row.get('durum')}"
        )
    fails = [row for row in rows if row.get("durum") != "TAMAM"]
    slow = [row for row in rows if isinstance(row.get("gecen_sn"), float) and row["gecen_sn"] > 10]
    print("-" * 78)
    print(f"  {len(rows) - len(fails)}/{len(rows)} dosya sorunsuz işlendi.")
    if slow:
        print(f"  10 sn'yi aşan hazırlık: {', '.join(str(r['video']) for r in slow)}")
    if fails:
        print("  DİKKAT:")
        for row in fails:
            print(f"    {row['video']}: {row['durum']}")


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="*", type=Path, default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Dikey / 4K / avi / mov / 3 dk gibi varyantları üretip dene",
    )
    parser.add_argument("--evidence", action="store_true", help="Hareket kanıtını da ölç")
    parser.add_argument("--keep", action="store_true", help="Sentetik dosyaları silme")
    args = parser.parse_args()

    temp_dir = Path(tempfile.mkdtemp(prefix="isg_video_check_"))
    frames_root = temp_dir / "frames"
    videos: list[Path] = collect(args.videos, args.limit)

    if args.synthetic:
        print(f"Sentetik varyantlar üretiliyor: {temp_dir}")
        for case in SYNTHETIC_CASES:
            made = make_synthetic(case, temp_dir)
            if made:
                videos.append(made)

    if not videos:
        raise SystemExit("Video yok. --videos ile dosya/klasör verin veya --synthetic kullanın.")

    rows = [check_one(video, frames_root, args.max_frames, args.evidence) for video in videos]
    print_table(rows)

    if args.keep:
        print(f"\nSentetik dosyalar: {temp_dir}")
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if any(row.get("durum") != "TAMAM" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
