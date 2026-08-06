"""
Uçtan uca (E2E) İSG Wake-Up sistemi.

Ana thread: video okuma + YOLOv8n ByteTrack + hız / süreklilik analizi.
Arka plan: tetiklenen kareleri ThreadPoolExecutor ile Mock VLM'e gönderir.

Kesintisiz akış: tetikte break yok; 150 kare cooldown uygulanır.

Kullanım:
    python main.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2

from tools.wake_up_detector import (
    HISTORY_LEN,
    MODEL_PATH,
    SAMPLE_VIDEO,
    TARGET_CLASS_IDS,
    TRACKER_CONFIG,
    TrackHistory,
    TrackedObject,
    TriggerEvent,
    draw_tracked_objects,
    ensure_trigger_directory,
    extract_tracked_objects,
    format_trigger_message,
    load_detector,
    resolve_sample_video,
    save_trigger_frame,
)

# --- E2E ayarları ---
SPEED_THRESHOLDS: dict[str, float] = {
    "person": 60.0,
    "truck": 50.0,
    "car": 50.0,
    "bus": 50.0,
}
PERSISTENCE_FRAMES: int = 3
COOLDOWN_FRAMES: int = 150  # ~5 sn @ 30 FPS
VLM_MAX_WORKERS: int = 2


def evaluate_triggers(
    tracked_objects: list[TrackedObject],
    trigger_counters: dict[int, int],
    speed_thresholds: dict[str, float] = SPEED_THRESHOLDS,
    persistence_frames: int = PERSISTENCE_FRAMES,
) -> list[TriggerEvent]:
    """
    Tüm sınıflar (person dahil) için hız + süreklilik teyidi.

    Hız eşiği art arda persistence_frames kare aşılırsa tetik üretir.
    """
    person_present = any(obj["class_name"] == "person" for obj in tracked_objects)
    events: list[TriggerEvent] = []
    active_ids: set[int] = set()

    for obj in tracked_objects:
        track_id = obj["track_id"]
        active_ids.add(track_id)

        threshold = speed_thresholds.get(obj["class_name"])
        if threshold is None:
            continue

        if obj["speed"] > threshold:
            trigger_counters[track_id] = trigger_counters.get(track_id, 0) + 1
        else:
            trigger_counters[track_id] = 0

        count = trigger_counters[track_id]
        if count >= persistence_frames:
            events.append(
                {
                    "track_id": track_id,
                    "class_name": obj["class_name"],
                    "speed": obj["speed"],
                    "threshold": threshold,
                    "persistence_count": count,
                    "condition_a": True,
                    "condition_b_critical": person_present and obj["class_name"] != "person",
                }
            )

    for track_id in list(trigger_counters.keys()):
        if track_id not in active_ids:
            trigger_counters[track_id] = 0

    return events


def mock_vlm_analysis(image_path: Path) -> None:
    """
    Geçici Mock VLM — gerçek ajan pipeline'ı yerine 3 sn simülasyon.
    Arka plan thread'inde çalışır; ana video döngüsünü bloklamaz.
    """
    print(f"[VLM Thread] Analiz kuyruğa alındı: {image_path.name}")
    time.sleep(3)
    print(
        "--- VLM ANALİZİ: Kaza riski tespit edildi. "
        "İlgili İSG maddesi getiriliyor... ---"
    )
    print(f"[VLM Thread] Tamamlandı: {image_path.name}")


def run_e2e_pipeline(
    video_path: Path | None = None,
    model_path: str = MODEL_PATH,
) -> None:
    """Ana thread'de YOLO/track döngüsü; VLM'i ThreadPoolExecutor ile çalıştırır."""
    path = resolve_sample_video(video_path or SAMPLE_VIDEO)
    model = load_detector(model_path)
    ensure_trigger_directory()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {path}")

    track_history: TrackHistory = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    trigger_counters: dict[int, int] = {}
    cooldown_remaining: int = 0
    pending_vlm: list[Future[Any]] = []

    print("E2E Wake-Up + Mock VLM")
    print(f"Video      : {path}")
    print(f"Tracker    : {TRACKER_CONFIG} (persist=True)")
    print(f"Eşikler    : {SPEED_THRESHOLDS}")
    print(f"Süreklilik : art arda {PERSISTENCE_FRAMES} kare")
    print(f"Cooldown   : {COOLDOWN_FRAMES} kare")
    print("-" * 60)

    frame_index = 0

    try:
        with ThreadPoolExecutor(max_workers=VLM_MAX_WORKERS) as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Video sonu.")
                    break

                t0 = time.perf_counter()
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    classes=TARGET_CLASS_IDS,
                    verbose=False,
                )
                inference_s = time.perf_counter() - t0
                fps = (1.0 / inference_s) if inference_s > 0 else 0.0

                tracked = extract_tracked_objects(
                    results[0], model, track_history, frame_index
                )

                in_cooldown = cooldown_remaining > 0
                if in_cooldown:
                    cooldown_remaining -= 1

                trigger_events: list[TriggerEvent] = []
                if not in_cooldown:
                    trigger_events = evaluate_triggers(tracked, trigger_counters)

                speed_summary = ", ".join(
                    f"#{o['track_id']}:{o['class_name']}={o['speed']:.1f}"
                    f"(c={trigger_counters.get(o['track_id'], 0)})"
                    for o in tracked
                ) or "-"
                cooldown_tag = f" cooldown={cooldown_remaining}" if in_cooldown else ""
                print(
                    f"[frame {frame_index:06d}] "
                    f"FPS={fps:.1f} | tracks={len(tracked)} | "
                    f"speeds=[{speed_summary}]{cooldown_tag}"
                )

                if trigger_events:
                    print("Dinamik Olay Tetiklendi!")
                    for event in trigger_events:
                        print(format_trigger_message(event))

                    annotated = draw_tracked_objects(frame, tracked)
                    saved = save_trigger_frame(annotated, frame_index)
                    print(f"Tetik karesi kaydedildi: {saved}")
                    print(
                        f"VLM arka plana gönderildi | "
                        f"cooldown={COOLDOWN_FRAMES} kare başlıyor"
                    )

                    pending_vlm.append(executor.submit(mock_vlm_analysis, saved))
                    cooldown_remaining = COOLDOWN_FRAMES
                    trigger_counters.clear()

                frame_index += 1

            # Video bitti; devam eden VLM işlerini bekle
            for fut in pending_vlm:
                fut.result()
    finally:
        cap.release()

    print("-" * 60)
    print("E2E pipeline tamamlandı.")


def main() -> None:
    try:
        # Path nesnesine çevirerek yolluyoruz
        run_e2e_pipeline(video_path=Path("dataset/raw_videos/ramak_kala.mp4"))
    except FileNotFoundError as exc:
        print(f"[HATA] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[HATA] E2E pipeline başarısız: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
