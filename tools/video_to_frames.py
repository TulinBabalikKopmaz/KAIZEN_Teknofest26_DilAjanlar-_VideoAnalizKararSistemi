"""
Golden Dataset — ham videolardan 1 FPS kare çıkarma aracı.

Kullanım (proje kökünden):
    python tools/video_to_frames.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_VIDEOS_DIR: Path = PROJECT_ROOT / "dataset" / "raw_videos"
FRAMES_DIR: Path = PROJECT_ROOT / "dataset" / "frames"
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi")
TARGET_FPS: float = 1.0


def ensure_directories() -> None:
    """dataset/raw_videos ve dataset/frames klasörlerini oluşturur."""
    RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def find_videos(directory: Path = RAW_VIDEOS_DIR) -> list[Path]:
    """Klasördeki .mp4 / .avi dosyalarını sıralı liste olarak döner."""
    videos: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(directory.glob(f"*{ext}"))
        videos.extend(directory.glob(f"*{ext.upper()}"))
    # Aynı dosyanın hem .mp4 hem .MP4 ile yakalanmasını engelle
    unique = {path.resolve(): path for path in videos}
    return sorted(unique.values(), key=lambda p: p.name.lower())


def _estimate_saved_frames(video_path: Path) -> int:
    """İlerleme çubuğu için yaklaşık kaydedilecek kare sayısını hesaplar."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if total_frames <= 0:
            return 0
        if fps <= 0:
            return total_frames
        interval = max(1, int(round(fps / TARGET_FPS)))
        return max(1, (total_frames + interval - 1) // interval)
    finally:
        cap.release()


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    progress: tqdm | None = None,
) -> int:
    """
    Videoyu 1 FPS ile karelere böler ve output_dir altına kaydeder.

    Returns:
        Kaydedilen kare sayısı.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    frame_interval = max(1, int(round(fps / TARGET_FPS)))

    frame_idx = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                out_path = output_dir / f"frame_{saved_count:04d}.jpg"
                if not cv2.imwrite(str(out_path), frame):
                    raise RuntimeError(f"Kare yazılamadı: {out_path}")
                saved_count += 1
                if progress is not None:
                    progress.update(1)

            frame_idx += 1
    finally:
        cap.release()

    if saved_count == 0:
        raise RuntimeError(f"Hiç kare çıkarılamadı: {video_path.name}")

    return saved_count


def process_all_videos(videos: Iterable[Path] | None = None) -> dict[str, int]:
    """
    raw_videos altındaki tüm videoları işler.

    Returns:
        video_adı -> kaydedilen kare sayısı eşlemesi.
    """
    ensure_directories()

    video_list = list(videos) if videos is not None else find_videos()
    if not video_list:
        print(f"Uyarı: {RAW_VIDEOS_DIR} içinde .mp4 / .avi bulunamadı.")
        print("Ham videoları bu klasöre koyup scripti tekrar çalıştırın.")
        return {}

    total_estimate = sum(_estimate_saved_frames(v) for v in video_list)
    results: dict[str, int] = {}

    with tqdm(
        total=max(total_estimate, 1),
        desc="Kare çıkarma",
        unit="frame",
        dynamic_ncols=True,
    ) as pbar:
        for video_path in video_list:
            video_stem = video_path.stem
            out_dir = FRAMES_DIR / video_stem
            pbar.set_postfix_str(video_path.name)

            try:
                saved = extract_frames_from_video(video_path, out_dir, progress=pbar)
                results[video_stem] = saved
            except Exception as exc:
                tqdm.write(f"[HATA] {video_path.name}: {exc}")
                results[video_stem] = 0

    return results


def main() -> None:
    print("Golden Dataset — Video → Frame dönüştürücü")
    print(f"Kaynak : {RAW_VIDEOS_DIR}")
    print(f"Hedef  : {FRAMES_DIR}")
    print("-" * 50)

    results = process_all_videos()
    if not results:
        sys.exit(0)

    ok = sum(1 for n in results.values() if n > 0)
    total_frames = sum(results.values())
    print("-" * 50)
    print(f"Tamamlanan video: {ok}/{len(results)}")
    print(f"Toplam kare     : {total_frames}")
    for name, count in results.items():
        status = f"{count} kare" if count > 0 else "BAŞARISIZ"
        print(f"  • {name}: {status}")


if __name__ == "__main__":
    main()
