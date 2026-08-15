#!/usr/bin/env python3
"""Videodan az sayıda zaman damgalı kare çıkarır (Qwen'e göndermek için)."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def iter_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def seconds_to_mmss(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    return f"{total // 60:02d}:{total % 60:02d}"


def safe_id(video_path: Path) -> str:
    stem = video_path.stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    if 0 < len(cleaned) <= 80:
        return cleaned
    digest = hashlib.md5(video_path.name.encode("utf-8")).hexdigest()[:10]
    return f"{video_path.parent.name}_{digest}"


def sample_times(duration: float, every_sec: float, max_frames: int) -> list[float]:
    if duration <= 0:
        return [0.0]
    natural = max(int(duration / every_sec) + 1, 1)
    n = min(max(natural, 2), max_frames)
    if n == 1:
        return [duration / 2]
    return [i * (duration / (n - 1)) for i in range(n)]


def extract_video(
    video_path: Path,
    out_root: Path,
    every_sec: float = 2.0,
    max_frames: int = 12,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0.0
    times = sample_times(duration, every_sec, max_frames)

    out_dir = out_root / video_path.parent.name / safe_id(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        stamp = seconds_to_mmss(t)
        dest = out_dir / f"{stamp.replace(':', '-')}.jpg"
        h, w = frame.shape[:2]
        scale = 768 / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        annotated = frame.copy()
        cv2.putText(
            annotated,
            stamp,
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(dest), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        saved.append({"path": str(dest), "time": stamp, "t_sec": t})
    cap.release()

    if not saved:
        raise RuntimeError(f"Kare çıkmadı: {video_path}")

    return {
        "video": str(video_path),
        "video_id": safe_id(video_path),
        "duration_sec": round(duration, 2),
        "fps": fps,
        "frames": saved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="data/videos", type=Path)
    parser.add_argument("--out", default="data/frames", type=Path)
    parser.add_argument("--every-sec", default=2.0, type=float)
    parser.add_argument("--max-frames", default=12, type=int)
    args = parser.parse_args()

    videos = iter_videos(args.videos)
    if not videos:
        raise SystemExit(f"Video bulunamadı: {args.videos.resolve()}")

    print(f"{len(videos)} video, en fazla {args.max_frames} kare")
    for video in videos:
        info = extract_video(video, args.out, args.every_sec, args.max_frames)
        print(f"  {video.name}: {len(info['frames'])} kare, {info['duration_sec']}s")


if __name__ == "__main__":
    main()
