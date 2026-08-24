"""EVREN VLM için kısa mp4 klibi: hareket tepesi çevresi, gövde sınırı ~190 MB ham."""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

CLIP_MAX_S = 45.0
PAD_BEFORE_S = 10.0
PAD_AFTER_S = 20.0
# EVREN gövde tavanı ~256 MB (base64). Ham dosya ~190 MB üstünde taşar.
RAW_BYTE_LIMIT = int(256 * 1024 * 1024 / 1.34)
# ffmpeg yoksa OpenCV mp4v şişiriyor; 640px / 12 fps ile küçük tut.
MAX_SEND_W = 640
TARGET_FPS = 12.0


def clip_span(
    duration_s: float,
    peak_sec: float | None = None,
    peaks: list[float] | None = None,
) -> tuple[float, float]:
    """Gönderilecek [start, end) saniye aralığı. Kısa videoda tüm süre."""
    duration_s = max(0.0, float(duration_s))
    if duration_s <= CLIP_MAX_S:
        return 0.0, duration_s
    points = [p for p in (peaks or []) if p is not None]
    if peak_sec is not None:
        points.append(float(peak_sec))
    if not points:
        start = max(0.0, duration_s / 2 - CLIP_MAX_S / 2)
        return start, min(duration_s, start + CLIP_MAX_S)
    center = sum(points) / len(points)
    start = max(0.0, center - PAD_BEFORE_S)
    end = min(duration_s, max(points) + PAD_AFTER_S)
    if end - start > CLIP_MAX_S:
        end = min(duration_s, start + CLIP_MAX_S)
    if end - start < 2.0:
        end = min(duration_s, start + 2.0)
    return start, end


def cut_clip(
    source: Path | str,
    dest: Path | str,
    start_s: float,
    end_s: float,
) -> Path:
    """ffmpeg ile keser; yoksa OpenCV. dest üzerine yazar."""
    source = Path(source)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if end_s <= start_s:
        end_s = start_s + 1.0
    duration = max(0.5, end_s - start_s)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.2f}",
            "-t",
            f"{duration:.2f}",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-vf",
            f"scale='min({MAX_SEND_W},iw)':-2",
            "-an",
            "-movflags",
            "+faststart",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return dest
    return _cut_opencv(source, dest, start_s, end_s)


def _cut_opencv(source: Path, dest: Path, start_s: float, end_s: float) -> Path:
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    if width > MAX_SEND_W:
        height = max(2, int(height * MAX_SEND_W / width) // 2 * 2)
        width = MAX_SEND_W
    out_fps = min(float(src_fps), TARGET_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, out_fps, (width, height))
    start_f = int(start_s * src_fps)
    end_f = int(end_s * src_fps)
    step = max(1, int(round(src_fps / out_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    index = start_f
    while index <= end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (index - start_f) % step == 0:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        index += 1
    cap.release()
    writer.release()
    return dest


def assert_under_body_limit(path: Path | str) -> None:
    size = Path(path).stat().st_size
    if size > RAW_BYTE_LIMIT:
        raise RuntimeError(
            f"{path} {size / 1e6:.1f} MB — EVREN gövde sınırı (~190 MB ham) aşılır. "
            "Klibi kısaltın."
        )


def encode_video_b64(path: Path | str) -> str:
    """mp4 → data-URI için ham base64. Anahtar / path loglanmaz."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Video klibi yok: {source}")
    assert_under_body_limit(source)
    return base64.b64encode(source.read_bytes()).decode("ascii")


def prepare_clip(
    source: Path | str,
    dest: Path | str,
    duration_s: float,
    peak_sec: float | None = None,
    peaks: list[float] | None = None,
) -> tuple[Path, float, float]:
    """Gönderilecek dosya ve [start, end) saniye. Kısa/küçük videoda kaynağı olduğu gibi kullanır."""
    source = Path(source)
    dest = Path(dest)
    start, end = clip_span(duration_s, peak_sec, peaks)
    almost_full = start <= 0.05 and end >= float(duration_s) - 0.05
    if almost_full and source.stat().st_size <= RAW_BYTE_LIMIT:
        return source, 0.0, max(0.0, float(duration_s))
    cut_clip(source, dest, start, end)
    if dest.exists() and dest.stat().st_size >= source.stat().st_size:
        # Transcode şişti (OpenCV mp4v); orijinal daha küçük ve limit altındaysa onu gönder.
        if source.stat().st_size <= RAW_BYTE_LIMIT:
            return source, 0.0, max(0.0, float(duration_s))
    assert_under_body_limit(dest)
    return dest, start, end
