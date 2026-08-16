"""
Uçtan uca (E2E) İSG Wake-Up sistemi.

Ana thread: video okuma + YOLOv8n ByteTrack + hız / süreklilik + frame differencing.
Arka plan: tetiklenen kareleri ThreadPoolExecutor ile gerçek VLM + RAG ajan pipeline'ına gönderir.

Kesintisiz akış: tetikte break yok; 150 kare cooldown uygulanır.

Kullanım:
    python main.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2

from graph_pipeline import run_pipeline
from utils.spec_output import incidents_to_spec, pipeline_result_to_spec
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
    "person": 40.0,
    "truck": 40.0,
    "car": 40.0,
    "bus": 40.0,
}
PERSISTENCE_FRAMES: int = 3
COOLDOWN_FRAMES: int = 150  # ~5 sn @ 30 FPS
VLM_MAX_WORKERS: int = 1  # VLM/RAG ağır; eşzamanlı tek iş daha stabil

# Frame differencing — YOLO sınıfı tanımasa bile ani piksel değişimini yakalar
MOTION_DIFF_THRESHOLD: int = 25  # absdiff binary eşik
MOTION_SPIKE_THRESHOLD: float = 0.10  # ekranın %10'undan fazlası değiştiyse tetik

# Video boyunca tetiklenen olayların özeti (saniye, görsel, VLM future)
incident_logs: list[dict[str, Any]] = []


def compute_motion_ratio(
    prev_gray: Any,
    curr_gray: Any,
    diff_threshold: int = MOTION_DIFF_THRESHOLD,
) -> float:
    """
    İki gri kare arasındaki değişen piksel oranını (0–1) döner.

    absdiff → threshold → countNonZero / (W*H)
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, binary = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(binary)
    height, width = curr_gray.shape[:2]
    total = max(width * height, 1)
    return changed / float(total)


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


def print_jury_report(image_path: Path, result: dict[str, Any]) -> None:
    """Jüri için VLM yorumu + RAG kanun maddeleri + aksiyonları şık yazdırır."""
    kanun = (result.get("isg_kanun_maddeleri") or "").strip()
    aksiyonlar = result.get("onerilen_aksiyonlar") or []

    print("\n" + "=" * 64)
    print("  VLM + RAG ANALİZ SONUCU (JÜRİ ÖZETİ)")
    print("=" * 64)
    print(f"  Kare              : {image_path.name}")
    print(f"  Tetik Sebebi      : {result.get('tetik_sebebi', '-')}")
    print(f"  Zaman Damgası     : {result.get('zaman_damgasi', '-')}")
    print(f"  Risk Seviyesi     : {result.get('risk_seviyesi', '-')}")
    print("-" * 64)
    print("  VLM Yorumu / Olay Özeti:")
    print(f"    {result.get('olay_ozeti', '-')}")
    print("-" * 64)
    print("  İSG Kanun Maddeleri (RAG):")
    if kanun:
        for i, block in enumerate(kanun.split("\n\n"), start=1):
            preview = " ".join(block.split())
            if len(preview) > 320:
                preview = preview[:320] + "..."
            print(f"    [{i}] {preview}")
    else:
        print("    (İlgili madde bulunamadı veya RAG atlandı)")
    print("-" * 64)
    print("  Önerilen Aksiyonlar:")
    if aksiyonlar:
        for i, action in enumerate(aksiyonlar, start=1):
            print(f"    {i}. {action}")
    else:
        print("    (Aksiyon üretilmedi)")
    spec = result.get("spec") or pipeline_result_to_spec(result)
    print("-" * 64)
    print("  Şartname JSON (gold ile aynı kalıp):")
    print(f"    {json.dumps(spec, ensure_ascii=False)}")
    print("=" * 64 + "\n")


def format_vlm_result(result: dict[str, Any] | None) -> str:
    """Future sonucunu tek satırlık okunabilir VLM özetine çevirir."""
    if not result:
        return "(sonuç yok)"
    olay = result.get("olay_ozeti", "-")
    risk = result.get("risk_seviyesi", "-")
    aksiyonlar = result.get("onerilen_aksiyonlar") or []
    aksiyon_txt = " | ".join(str(a) for a in aksiyonlar) if aksiyonlar else "-"
    return f"{olay} | Risk: {risk} | Aksiyon: {aksiyon_txt}"


def print_incident_summary(logs: list[dict[str, Any]]) -> None:
    """Video bitince tüm tetiklenen olayların Incident Summary raporunu basar."""
    print("\n" + "═" * 64)
    print("  OLAY ÖZETİ RAPORU (INCIDENT SUMMARY)")
    print("═" * 64)

    if not logs:
        print("  Bu videoda tetiklenen olay bulunamadı.")
        print("═" * 64 + "\n")
        return

    for idx, entry in enumerate(logs, start=1):
        future_obj: Future[Any] = entry["future"]
        try:
            vlm_result = future_obj.result()
        except Exception as exc:
            vlm_result = {"olay_ozeti": f"Hata: {exc}", "risk_seviyesi": "-", "onerilen_aksiyonlar": []}

        print(f"🔴 OLAY [{idx}]")
        print(f"⏰ Zaman: {entry['saniye']:.2f}. saniye")
        print(f"📸 Görsel: {entry['gorsel']}")
        print(f"⚡ Tetik: {entry.get('tetik_sebebi', '-')}")
        print(f"🤖 VLM Analizi: {format_vlm_result(vlm_result)}")
        print("-" * 64)

    print(f"  Toplam olay: {len(logs)}")
    print("═" * 64 + "\n")

    resolved: list[dict[str, Any]] = []
    for entry in logs:
        future_obj = entry["future"]
        try:
            vlm_result = future_obj.result()
        except Exception as exc:
            vlm_result = {
                "olay_ozeti": f"Hata: {exc}",
                "risk_seviyesi": "Bilinmiyor",
                "onerilen_aksiyonlar": [],
            }
        resolved.append({"saniye": entry.get("saniye"), "vlm_result": vlm_result})

    spec = incidents_to_spec(resolved)
    print("ŞARTNAME JSON (tüm video, gold kalıbı)")
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    out_path = Path("data/exports/last_system_output.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kaydedildi: {out_path}\n")


def build_trigger_reason(
    motion_spike: bool,
    trigger_events: list[TriggerEvent],
) -> str:
    """Wake-Up tetik sebebini VLM'e iletilecek metne çevirir."""
    reasons: list[str] = []
    if motion_spike:
        reasons.append("Ani hareket/düşme tespit edildi")
    for event in trigger_events:
        reasons.append(
            f"İnsan/Araç aşırı hızlandı "
            f"({event['class_name']} ID#{event['track_id']}, "
            f"hız={event['speed']:.1f})"
        )
    return " | ".join(reasons) if reasons else "Dinamik olay tetiklendi"


def run_vlm_rag_analysis(
    image_path: Path,
    trigger_reason: str = "",
) -> dict[str, Any]:
    """
    Gerçek LangGraph ajan pipeline'ı (Video Analyzer → Risk → Action+RAG).

    ThreadPoolExecutor worker'ından güvenli çağrı için asyncio.run kullanır.
    """
    print(f"\n[VLM Thread] Ajan pipeline başlıyor: {image_path}")
    if trigger_reason:
        print(f"[VLM Thread] Tetik sebebi: {trigger_reason}")
    try:
        result = asyncio.run(
            run_pipeline(
                keyframes=[str(image_path)],
                trigger_reason=trigger_reason,
            )
        )
        print_jury_report(image_path, result)
        return result
    except Exception as exc:
        print(f"[VLM Thread] Analiz hatası ({image_path.name}): {exc}")
        return {
            "zaman_damgasi": "-",
            "olay_ozeti": f"Analiz başarısız: {exc}",
            "risk_seviyesi": "Bilinmiyor",
            "onerilen_aksiyonlar": [],
            "isg_kanun_maddeleri": "",
            "tetik_sebebi": trigger_reason,
        }


def run_e2e_pipeline(
    video_path: Path | None = None,
    model_path: str = MODEL_PATH,
) -> None:
    """Ana thread'de YOLO/track döngüsü; VLM+RAG'i ThreadPoolExecutor ile çalıştırır."""
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
    prev_gray: Any | None = None
    incident_logs.clear()

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 0:
        video_fps = 30.0

    print("E2E Wake-Up + VLM/RAG Ajan Pipeline")
    print(f"Video      : {path}")
    print(f"Video FPS  : {video_fps:.2f}")
    print(f"Tracker    : {TRACKER_CONFIG} (persist=True)")
    print(f"Eşikler    : {SPEED_THRESHOLDS}")
    print(f"Süreklilik : art arda {PERSISTENCE_FRAMES} kare")
    print(f"Motion     : spike>{MOTION_SPIKE_THRESHOLD:.0%} (diff≥{MOTION_DIFF_THRESHOLD})")
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

                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Işık titremesi / sensör gürültüsünü yumuşat (prev_gray da aynı blur'lu kopyadan gelir)
                curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)
                motion_ratio = 0.0
                if prev_gray is not None:
                    motion_ratio = compute_motion_ratio(prev_gray, curr_gray)

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
                motion_spike = False
                if not in_cooldown:
                    trigger_events = evaluate_triggers(tracked, trigger_counters)
                    if motion_ratio > MOTION_SPIKE_THRESHOLD:
                        motion_spike = True

                speed_summary = ", ".join(
                    f"#{o['track_id']}:{o['class_name']}={o['speed']:.1f}"
                    f"(c={trigger_counters.get(o['track_id'], 0)})"
                    for o in tracked
                ) or "-"
                cooldown_tag = f" cooldown={cooldown_remaining}" if in_cooldown else ""
                print(
                    f"[frame {frame_index:06d}] "
                    f"FPS={fps:.1f} | tracks={len(tracked)} | "
                    f"motion={motion_ratio:.3f} | "
                    f"speeds=[{speed_summary}]{cooldown_tag}"
                )

                if trigger_events or motion_spike:
                    print("Dinamik Olay Tetiklendi!")
                    if motion_spike:
                        print(
                            "Sınıf Bağımsız Ani Hareket/Düşme Tespit Edildi! "
                            f"(motion_ratio={motion_ratio:.3f} > {MOTION_SPIKE_THRESHOLD})"
                        )
                    for event in trigger_events:
                        print(format_trigger_message(event))

                    annotated = draw_tracked_objects(frame, tracked)
                    if motion_spike:
                        cv2.putText(
                            annotated,
                            f"MOTION SPIKE {motion_ratio:.2%}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    saved = save_trigger_frame(annotated, frame_index)
                    print(f"Tetik karesi kaydedildi: {saved}")

                    trigger_reason = build_trigger_reason(motion_spike, trigger_events)
                    print(f"Tetik sebebi: {trigger_reason}")
                    print(
                        f"VLM+RAG arka plana gönderildi | "
                        f"cooldown={COOLDOWN_FRAMES} kare başlıyor"
                    )

                    zaman_sn = frame_index / video_fps
                    future_obj = executor.submit(
                        run_vlm_rag_analysis,
                        saved,
                        trigger_reason,
                    )
                    pending_vlm.append(future_obj)
                    incident_logs.append(
                        {
                            "saniye": zaman_sn,
                            "gorsel": saved.name,
                            "tetik_sebebi": trigger_reason,
                            "future": future_obj,
                        }
                    )
                    cooldown_remaining = COOLDOWN_FRAMES
                    trigger_counters.clear()

                prev_gray = curr_gray.copy()
                frame_index += 1

            for fut in pending_vlm:
                fut.result()
    finally:
        cap.release()

    print_incident_summary(incident_logs)
    print("E2E pipeline tamamlandı.")


def main() -> None:
    try:
        run_e2e_pipeline(video_path=Path("dataset/raw_videos/normal.mp4"))
    except FileNotFoundError as exc:
        print(f"[HATA] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[HATA] E2E pipeline başarısız: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
