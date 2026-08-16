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
    """Eşit aralıklı yedek örnekleme (hareket skoru yoksa)."""
    if duration <= 0:
        return [0.0]
    natural = max(int(duration / every_sec) + 1, 1)
    n = min(max(natural, 2), max_frames)
    if n == 1:
        return [duration / 2]
    return [i * (duration / (n - 1)) for i in range(n)]


def target_frame_count(duration: float, max_frames: int) -> int:
    """Kısa klipte daha sık kare; uzun klipte tavan max_frames."""
    if duration <= 0:
        return 1
    dense = int(round(duration / 0.7)) + 1
    return max(2, min(max_frames, max(4, dense)))


def _clamp_time(t: float, duration: float) -> float:
    return min(max(t, 0.0), max(duration - 0.01, 0.0))


def pick_times(
    duration: float,
    max_frames: int,
    motion: list[tuple[float, float]] | None = None,
    min_gap: float | None = None,
) -> list[float]:
    """Her saniye en fazla bir kare. Kısa klipte tüm saniyeler; uzunda hareket tepeleri."""
    del min_gap  # saniye hizası zaten çakışmayı önler
    if duration <= 0:
        return [0.0]
    last = max(int(round(max(duration - 0.05, 0.0))), 0)
    seconds = list(range(0, last + 1))
    n = min(max(2, max_frames), len(seconds)) if last > 0 else 1

    def to_time(sec: int) -> float:
        return _clamp_time(float(sec), duration)

    if len(seconds) <= max_frames:
        return [to_time(s) for s in seconds]

    chosen: list[int] = []

    def add_sec(sec: int) -> None:
        sec = min(max(sec, 0), last)
        if sec not in chosen and len(chosen) < n:
            chosen.append(sec)

    add_sec(0)
    add_sec(last)
    if motion:
        for t, score in sorted(motion, key=lambda item: item[1], reverse=True):
            if len(chosen) >= n:
                break
            if score <= 0:
                continue
            add_sec(int(round(t)))
    if n > 1:
        step = last / (n - 1)
        for i in range(n):
            if len(chosen) >= n:
                break
            add_sec(int(round(i * step)))
    return [to_time(s) for s in sorted(chosen)]


def motion_scores(cap: cv2.VideoCapture, fps: float) -> list[tuple[float, float]]:
    """Küçük gri kare farkıyla hareket skoru. ~6 örnek/sn."""
    step = max(1, int(round((fps or 25.0) / 6)))
    prev = None
    scores: list[tuple[float, float]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90))
        t = idx / fps if fps else 0.0
        if prev is not None:
            diff = cv2.absdiff(small, prev)
            scores.append((t, float(diff.mean())))
        prev = small
        idx += 1
    return scores


def _write_frame(frame, dest: Path, stamp: str) -> None:
    h, w = frame.shape[:2]
    scale = 512 / max(h, w)
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
    cv2.imwrite(str(dest), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])


def extract_video(
    video_path: Path,
    out_root: Path,
    every_sec: float = 0.75,
    max_frames: int = 6,
    *,
    use_motion: bool = True,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0.0

    motion: list[tuple[float, float]] = []
    if use_motion:
        motion = motion_scores(cap, fps)
    cap.release()

    if motion:
        times = pick_times(duration, max_frames, motion)
    else:
        times = sample_times(duration, every_sec, max_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Açılamadı: {video_path}")

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
        _write_frame(frame, dest, stamp)
        saved.append({"path": str(dest), "time": stamp, "t_sec": round(t, 2)})
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
    parser.add_argument("--every-sec", default=0.75, type=float)
    parser.add_argument("--max-frames", default=6, type=int)
    parser.add_argument("--no-motion", action="store_true", help="Hareket skoru kullanma")
    args = parser.parse_args()

    videos = iter_videos(args.videos)
    if not videos:
        raise SystemExit(f"Video bulunamadı: {args.videos.resolve()}")

    print(f"{len(videos)} video, en fazla {args.max_frames} kare")
    for video in videos:
        info = extract_video(
            video,
            args.out,
            args.every_sec,
            args.max_frames,
            use_motion=not args.no_motion,
        )
        print(f"  {video.name}: {len(info['frames'])} kare, {info['duration_sec']}s")


if __name__ == "__main__":
    main()
