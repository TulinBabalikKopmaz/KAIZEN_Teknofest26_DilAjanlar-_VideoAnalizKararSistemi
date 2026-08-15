"""
Wake-Up Detector — ByteTrack + hız tabanlı dinamik olay tetikleyici.

YOLOv8n track ile person / car / truck / bus izler.
Son 5 karelik centroid yer değişimine göre hız hesaplar;
sınıf eşiğini aşan nesne VLM'yi uyandırır.

Kullanım (proje kökünden):
    python tools/wake_up_detector.py
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, TypedDict

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_VIDEOS_DIR: Path = PROJECT_ROOT / "dataset" / "raw_videos"
TRIGGER_DIR: Path = PROJECT_ROOT / "dataset" / "frames" / "dynamic_triggers"
SAMPLE_VIDEO: Path = RAW_VIDEOS_DIR / "sample.mp4"
MODEL_PATH: str = "yolov8n.pt"
TRACKER_CONFIG: str = "bytetrack.yaml"
HISTORY_LEN: int = 5

# Piksel / 5-frame cinsinden kritik hız eşikleri (optimize edilebilir)
# Not: person bilerek yok — insan hızı sistemi tetiklemez.
SPEED_THRESHOLDS: dict[str, float] = {
    "truck": 50.0,
    "car": 50.0,
    "bus": 50.0,
}
# Araç hızı art arda kaç kare eşiği aşmalı (jitter / sahte sıçrama filtresi)
PERSISTENCE_FRAMES: int = 3

# COCO: person=0, car=2, bus=5, truck=7
PERSON_CLASS_ID: int = 0
PERSON_MIN_CONF: float = 0.65  # person için conf filtresi; araçlara uygulanmaz
VEHICLE_CLASS_IDS: frozenset[int] = frozenset({2, 5, 7})
TARGET_CLASS_IDS: list[int] = [PERSON_CLASS_ID, *sorted(VEHICLE_CLASS_IDS)]

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),
    2: (255, 128, 0),
    5: (0, 165, 255),
    7: (0, 0, 255),
}

Centroid = tuple[float, float]
# (x, y, frame_index) — re-ID teleportasyonunu yakalamak için kare numarası tutulur
TrackPoint = tuple[float, float, int]
TrackHistory = dict[int, Deque[TrackPoint]]


class TrackedObject(TypedDict):
    track_id: int
    class_id: int
    class_name: str
    conf: float
    xyxy: tuple[int, int, int, int]
    centroid: Centroid
    speed: float


class TriggerEvent(TypedDict):
    track_id: int
    class_name: str
    speed: float
    threshold: float
    persistence_count: int
    condition_a: bool
    condition_b_critical: bool


def ensure_trigger_directory(directory: Path = TRIGGER_DIR) -> None:
    """Dinamik tetik karelerinin kaydedileceği klasörü oluşturur."""
    directory.mkdir(parents=True, exist_ok=True)


def resolve_sample_video(video_path: Path = SAMPLE_VIDEO) -> Path:
    """sample.mp4 yolunu doğrular; yoksa FileNotFoundError fırlatır."""
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Test videosu bulunamadı: {video_path}\n"
            "Lütfen dataset/raw_videos/sample.mp4 dosyasını yerleştirip tekrar deneyin."
        )
    return video_path


def load_detector(model_path: str = MODEL_PATH) -> YOLO:
    """YOLOv8n modelini yükler (yoksa ultralytics otomatik indirir)."""
    print(f"Model yükleniyor: {model_path}")
    return YOLO(model_path)


def class_name_from_id(model: YOLO, class_id: int) -> str:
    """COCO sınıf id'sini isme çevirir."""
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(class_id)


def box_centroid(xyxy: tuple[int, int, int, int]) -> Centroid:
    """Bounding box merkez noktasını (x, y) döner."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean_distance(p1: Centroid, p2: Centroid) -> float:
    """İki nokta arası Öklid mesafesi (piksel)."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def compute_speed(history: Deque[TrackPoint]) -> float:
    """
    Son N ardışık karedeki yer değiştirme (ilk ↔ son centroid) = hız skoru.

    Birim: piksel / HISTORY_LEN-frame penceresi.
    """
    if len(history) < 2:
        return 0.0
    x0, y0, _ = history[0]
    x1, y1, _ = history[-1]
    return euclidean_distance((x0, y0), (x1, y1))


def update_track_history(
    track_history: TrackHistory,
    track_id: int,
    centroid: Centroid,
    frame_index: int,
    maxlen: int = HISTORY_LEN,
) -> float:
    """
    Track geçmişine (centroid, frame) ekler ve güncel hızı döner.

    ID kaybolup 1'den fazla kare sonra yeniden görünürse (re-ID / teleport):
    geçmişi sıfırlar ve bu kareyi ilk tespit sayar (hız = 0).
    """
    if track_id not in track_history:
        track_history[track_id] = deque(maxlen=maxlen)

    history = track_history[track_id]
    cx, cy = centroid

    if history:
        last_seen_frame = history[-1][2]
        if frame_index - last_seen_frame > 1:
            history.clear()
            history.append((cx, cy, frame_index))
            return 0.0

    history.append((cx, cy, frame_index))
    return compute_speed(history)


def extract_tracked_objects(
    result: object,
    model: YOLO,
    track_history: TrackHistory,
    frame_index: int,
    allowed_classes: list[int] = TARGET_CLASS_IDS,
) -> list[TrackedObject]:
    """ByteTrack sonucundan ID'li nesneleri çıkarır ve hızları günceller."""
    allowed = set(allowed_classes)
    tracked: list[TrackedObject] = []

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return tracked

    track_ids = boxes.id
    if track_ids is None:
        return tracked

    ids = track_ids.int().cpu().tolist()
    for box, track_id in zip(boxes, ids):
        class_id = int(box.cls[0].item())
        if class_id not in allowed:
            continue

        conf = float(box.conf[0].item())
        # Person: düşük güvenli false-positive'leri ele; araçlara conf filtresi yok
        if class_id == PERSON_CLASS_ID and conf < PERSON_MIN_CONF:
            continue

        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        xyxy = (x1, y1, x2, y2)
        centroid = box_centroid(xyxy)
        speed = update_track_history(
            track_history,
            int(track_id),
            centroid,
            frame_index,
        )
        name = class_name_from_id(model, class_id)

        tracked.append(
            {
                "track_id": int(track_id),
                "class_id": class_id,
                "class_name": name,
                "conf": conf,
                "xyxy": xyxy,
                "centroid": centroid,
                "speed": speed,
            }
        )

    return tracked


def evaluate_triggers(
    tracked_objects: list[TrackedObject],
    trigger_counters: dict[int, int],
    speed_thresholds: dict[str, float] = SPEED_THRESHOLDS,
    persistence_frames: int = PERSISTENCE_FRAMES,
) -> list[TriggerEvent]:
    """
    Condition A: araç hızı eşiği art arda persistence_frames kare aşarsa tetik.
    Condition B: A aktifken karede person varsa Kritik Risk bayrağı (ilerisi için).

    person sınıfı hız kontrolünden tamamen hariç tutulur.
    """
    person_present = any(obj["class_id"] == PERSON_CLASS_ID for obj in tracked_objects)
    events: list[TriggerEvent] = []
    active_vehicle_ids: set[int] = set()

    for obj in tracked_objects:
        # İnsan hızı ASLA tetiklemez (Condition A/B atlanır)
        if obj["class_id"] == PERSON_CLASS_ID:
            continue
        if obj["class_id"] not in VEHICLE_CLASS_IDS:
            continue

        track_id = obj["track_id"]
        active_vehicle_ids.add(track_id)

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
                    "condition_b_critical": person_present,
                }
            )

    # Bu karede görünmeyen araçların süreklilik sayacını sıfırla
    for track_id in list(trigger_counters.keys()):
        if track_id not in active_vehicle_ids:
            trigger_counters[track_id] = 0

    return events


def draw_tracked_objects(
    frame: np.ndarray,
    tracked_objects: list[TrackedObject],
) -> np.ndarray:
    """Track ID, sınıf, hız ve bbox çizer."""
    annotated = frame.copy()
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["xyxy"]
        color = CLASS_COLORS.get(obj["class_id"], (255, 255, 255))
        label = (
            f"ID#{obj['track_id']} {obj['class_name']} "
            f"spd={obj['speed']:.1f} conf={obj['conf']:.2f}"
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cx, cy = obj["centroid"]
        cv2.circle(annotated, (int(cx), int(cy)), 4, color, -1)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def save_trigger_frame(
    frame: np.ndarray,
    frame_index: int,
    output_dir: Path = TRIGGER_DIR,
) -> Path:
    """Tetiklenen kareyi dynamic_triggers altına kaydeder."""
    ensure_trigger_directory(output_dir)
    out_path = output_dir / f"dynamic_trigger_{frame_index:06d}.jpg"
    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError(f"Tetik karesi yazılamadı: {out_path}")
    return out_path


def format_trigger_message(event: TriggerEvent) -> str:
    """Terminale yazılacak tetik mesajını üretir."""
    display_name = event["class_name"].capitalize()
    msg = (
        f"{display_name} ID #{event['track_id']} hızı "
        f"{event['speed']:.1f} piksel/{HISTORY_LEN}-frame "
        f"(eşik {event['threshold']:.1f}) "
        f"art arda {event['persistence_count']} kare aşarak tetiklendi!"
    )
    if event["condition_b_critical"]:
        msg += " [Condition B] Kritik Risk: karede insan da mevcut."
    return msg


def run_wake_up_detection(
    video_path: Path | None = None,
    model_path: str = MODEL_PATH,
) -> Path | None:
    """
    Videoyu ByteTrack ile tarar; kritik hızda dinamik wake-up tetikler.

    Returns:
        Kaydedilen tetik karesinin yolu; tetik yoksa None.
    """
    path = resolve_sample_video(video_path or SAMPLE_VIDEO)
    model = load_detector(model_path)
    ensure_trigger_directory()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {path}")

    track_history: TrackHistory = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    trigger_counters: dict[int, int] = {}

    print(f"Video     : {path}")
    print(f"Tracker   : {TRACKER_CONFIG} (persist=True)")
    print(f"Sınıflar  : {TARGET_CLASS_IDS}")
    print(f"Eşikler   : {SPEED_THRESHOLDS} (person hariç)")
    print(f"Süreklilik: art arda {PERSISTENCE_FRAMES} kare")
    print("-" * 50)

    frame_index = 0
    trigger_path: Path | None = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video sonu: dinamik hız tetiklemesi oluşmadı.")
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

            result = results[0]
            tracked = extract_tracked_objects(
                result, model, track_history, frame_index
            )
            trigger_events = evaluate_triggers(tracked, trigger_counters)

            speed_summary = ", ".join(
                f"#{o['track_id']}:{o['class_name']}={o['speed']:.1f}"
                f"(c={trigger_counters.get(o['track_id'], 0)})"
                for o in tracked
                if o["class_id"] in VEHICLE_CLASS_IDS
            ) or "-"
            print(
                f"[frame {frame_index:06d}] "
                f"inference={inference_s * 1000:.1f} ms | "
                f"FPS={fps:.1f} | "
                f"tracks={len(tracked)} | "
                f"vehicles=[{speed_summary}]"
            )

            if trigger_events:
                print("Dinamik Olay Tetiklendi!")
                for event in trigger_events:
                    print(format_trigger_message(event))

                annotated = draw_tracked_objects(frame, tracked)
                trigger_path = save_trigger_frame(annotated, frame_index)
                print(f"Tetik karesi kaydedildi: {trigger_path}")
                break

            frame_index += 1
    finally:
        cap.release()

    return trigger_path


def main() -> None:
    print("Wake-Up Detector — ByteTrack + Hız Analizi")
    print("-" * 50)
    try:
        trigger = run_wake_up_detection()
    except FileNotFoundError as exc:
        print(f"[HATA] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[HATA] Wake-up detection başarısız: {exc}")
        sys.exit(1)

    if trigger is None:
        print("Sonuç: Tetik oluşmadı.")
        sys.exit(0)

    print(f"Sonuç: Dinamik VLM wake-up tetiklendi → {trigger}")


if __name__ == "__main__":
    main()
