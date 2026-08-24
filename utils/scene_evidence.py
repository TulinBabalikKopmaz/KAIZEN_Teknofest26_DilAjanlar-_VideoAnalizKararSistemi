"""Videodan Qwen'e yardımcı 'kanıt' çıkarır.

Düşünce: Qwen tek başına bazen sahneyi yanlış okur.
Biz ona cevap anahtarı vermiyoruz; sadece ölçülebilir ipuçları veriyoruz:
- hareket tepe anı (kare farkı)
- kişi–araç yakınlığı (YOLO varsa)
- yangın/duman benzeri renk artışı (basit HSV)

Bunlar model değil; sensör notu gibi düşün.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# COCO: person=0, car=2, motorcycle=3, bus=5, truck=7
PERSON_ID = 0
VEHICLE_IDS = frozenset({2, 3, 5, 7})

# Kutu merkezleri arası mesafe / kişi kutusu diyagonali
CLOSE_RATIO = 1.35
VERY_CLOSE_RATIO = 0.85


@dataclass
class SceneEvidence:
    duration_sec: float = 0.0
    motion_peak_sec: float | None = None
    motion_peak_score: float = 0.0
    # Tek tepe olay anını ±2 sn içinde yalnızca %25 buluyor, ilk üç tepe %69
    # (ölçüm: scripts/eval_wakeup.py). Odak kareleri ve uzun video penceresi
    # bu yüzden tek tepe yerine tepe listesini kullanıyor.
    motion_peaks: list[float] = field(default_factory=list)
    motion_elevated: bool = False
    person_count_max: int = 0
    vehicle_count_max: int = 0
    min_person_vehicle_ratio: float | None = None
    person_vehicle_close: bool = False
    person_vehicle_very_close: bool = False
    fire_like_ratio_max: float = 0.0
    fire_like_rise: float = 0.0
    fire_suspect: bool = False
    yolo_available: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SceneEvidence | None:
        """Prediction JSON'daki kanıt bloğunu yükler (ek anahtarları yok sayar)."""
        if not raw:
            return None
        aliases = {
            "motion_elevated": "motion_elevated",
            "person_count_max": "person_count_max",
            "vehicle_count_max": "vehicle_count_max",
            "fire_suspect": "fire_suspect",
            "yolo_available": "yolo_available",
            "person_vehicle_close": "person_vehicle_close",
            "person_vehicle_very_close": "person_vehicle_very_close",
        }
        data = dict(raw)
        for src, dest in aliases.items():
            if src in data and dest not in data:
                data[dest] = data[src]
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def prompt_block(self) -> str:
        """Modele gidecek kısa Türkçe kanıt metni."""
        lines = ["Sensör kanıtları (uydurma değil, ölçüm; sen yine görseli oku):"]
        if self.motion_elevated and self.motion_peak_sec is not None:
            lines.append(
                f"- Ani hareket tepe noktası ~{self._mmss(self.motion_peak_sec)} "
                f"(skor {self.motion_peak_score:.1f}). Kritik an buraya yakın olabilir."
            )
            others = [p for p in self.motion_peaks if abs(p - self.motion_peak_sec) > 1.0]
            if others:
                lines.append(
                    "- Diğer hareketli anlar: "
                    + ", ".join(self._mmss(p) for p in others)
                    + ". Olay bunlardan birinde de olabilir."
                )
        if self.yolo_available:
            lines.append(
                f"- YOLO: en fazla {self.person_count_max} kişi, "
                f"{self.vehicle_count_max} araç/motosiklet."
            )
            if self.person_vehicle_very_close:
                lines.append(
                    "- Kişi ile araç/motosiklet ÇOK YAKIN (kutular neredeyse örtüşüyor). "
                    "Çarpışma veya ramak kala ihtimali yüksek."
                )
            elif self.person_vehicle_close:
                lines.append(
                    "- Kişi ile araç/motosiklet yakın. Tehlikeli yaklaşma olabilir."
                )
        if self.fire_suspect:
            lines.append(
                f"- Turuncu/kırmızı parlak bölge arttı "
                f"(max oran {self.fire_like_ratio_max:.3f}, artış {self.fire_like_rise:.3f}). "
                "Yanma / kıvılcım ihtimalini özellikle kontrol et."
            )
        if len(lines) == 1:
            lines.append("- Belirgin ekstra kanıt yok; sadece görsele güven.")
        return "\n".join(lines)

    def suggests_second_look(self) -> bool:
        """Model 'rutin/Düşük' derse ikinci bakışa değer mi?

        YOLO kişi-araç kutusu kalabalık depoda sık örtüşür; hareket tepe yoksa
        ikinci bakış sahte ramak üretir.
        """
        yolo_close = self.person_vehicle_close or self.person_vehicle_very_close
        return bool(self.fire_suspect or self.motion_elevated or (yolo_close and self.motion_elevated))

    @staticmethod
    def _mmss(seconds: float) -> str:
        total = max(int(round(seconds)), 0)
        return f"{total // 60:02d}:{total % 60:02d}"


def _box_diag(xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return math.hypot(x2 - x1, y2 - y1) or 1.0


def _center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def fire_like_ratio(frame_bgr: np.ndarray) -> float:
    """Turuncu-kırmızı parlak piksel oranı (kaba yangın/kıvılcım ipucu)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # kırmızı + turuncu (iki H aralığı)
    mask1 = cv2.inRange(hsv, (0, 80, 120), (18, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 80, 120), (179, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    return float(mask.mean() / 255.0)


def motion_profile(video_path: Path, sample_fps: float = 6.0) -> list[tuple[float, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
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
        t = idx / fps
        if prev is not None:
            scores.append((t, float(cv2.absdiff(small, prev).mean())))
        prev = small
        idx += 1
    cap.release()
    return scores


def _try_load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    weights = Path(__file__).resolve().parents[1] / "yolov8n.pt"
    model_path = str(weights) if weights.exists() else "yolov8n.pt"
    try:
        return YOLO(model_path)
    except Exception:
        return None


def top_motion_peaks(
    motion: list[tuple[float, float]],
    count: int = 3,
    min_gap: float = 2.0,
) -> list[float]:
    """En yüksek skorlu, birbirinden min_gap saniye uzak tepeler (zaman sıralı)."""
    picked: list[float] = []
    for t, score in sorted(motion, key=lambda item: item[1], reverse=True):
        if score <= 0:
            continue
        if all(abs(t - other) >= min_gap for other in picked):
            picked.append(t)
        if len(picked) >= count:
            break
    return sorted(picked)


def analyze_video(
    video_path: Path,
    max_probe_frames: int = 24,
    *,
    use_yolo: bool = True,
) -> SceneEvidence:
    """Videoyu hızlı tarayıp SceneEvidence üretir. Gold/etiket okumaz.

    use_yolo=False: demo süre bütçesi için kişi-araç yakınlığı atlanır,
    hareket ve yangın rengi ipuçları yine hesaplanır.
    """
    evidence = SceneEvidence()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        evidence.notes.append("video açılamadı")
        return evidence

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    evidence.duration_sec = frame_count / fps if fps else 0.0

    motion = motion_profile(video_path)
    if motion:
        peak_t, peak_s = max(motion, key=lambda item: item[1])
        evidence.motion_peak_sec = peak_t
        evidence.motion_peak_score = peak_s
        evidence.motion_peaks = top_motion_peaks(motion)
        vals = [s for _, s in motion]
        median = float(np.median(vals)) if vals else 0.0
        evidence.motion_elevated = peak_s >= max(8.0, median * 2.2)

    # Eşit aralıklı karelerde yangın + (opsiyonel) YOLO
    n = min(max_probe_frames, max(frame_count, 1))
    indices = (
        [int(i * (frame_count - 1) / (n - 1)) for i in range(n)]
        if frame_count > 1 and n > 1
        else [0]
    )
    fire_ratios: list[float] = []
    model = _try_load_yolo() if use_yolo else None
    evidence.yolo_available = model is not None
    if not use_yolo:
        evidence.notes.append("hızlı mod: YOLO atlandı")
    min_ratio: float | None = None

    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        ratio = fire_like_ratio(frame)
        fire_ratios.append(ratio)

        if model is None:
            continue
        try:
            result = model.predict(frame, verbose=False, conf=0.35)[0]
        except Exception as exc:
            evidence.notes.append(f"yolo hata: {exc}")
            continue
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        persons: list[tuple[float, float, float, float]] = []
        vehicles: list[tuple[float, float, float, float]] = []
        for box in boxes:
            cls_id = int(box.cls[0].item())
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            if cls_id == PERSON_ID:
                persons.append(xyxy)
            elif cls_id in VEHICLE_IDS:
                vehicles.append(xyxy)
        evidence.person_count_max = max(evidence.person_count_max, len(persons))
        evidence.vehicle_count_max = max(evidence.vehicle_count_max, len(vehicles))
        for p in persons:
            pc = _center(p)
            pdiag = _box_diag(p)
            for v in vehicles:
                dist = math.hypot(pc[0] - _center(v)[0], pc[1] - _center(v)[1])
                r = dist / pdiag
                min_ratio = r if min_ratio is None else min(min_ratio, r)

    cap.release()

    evidence.min_person_vehicle_ratio = min_ratio
    if min_ratio is not None:
        evidence.person_vehicle_close = min_ratio <= CLOSE_RATIO
        evidence.person_vehicle_very_close = min_ratio <= VERY_CLOSE_RATIO

    if fire_ratios:
        evidence.fire_like_ratio_max = max(fire_ratios)
        evidence.fire_like_rise = max(fire_ratios) - min(fire_ratios)
        # Turuncu makine/yelek için false positive azalt: daha sıkı eşik
        evidence.fire_suspect = (
            evidence.fire_like_ratio_max >= 0.07 and evidence.fire_like_rise >= 0.035
        ) or evidence.fire_like_ratio_max >= 0.12

    return evidence
