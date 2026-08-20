#!/usr/bin/env python3
"""Gerçek zamanlı akış: RTSP / webcam / dosya → wake-up tetikli asenkron analiz.

Statik klasör taramasının canlı karşılığı. Tasarım kararı: akış hiçbir zaman
analiz için beklemez. Okuyucu ayrı bir iş parçacığında canlı kalır, hareket
tetiklenmesinde biriken kareler kuyruğa atılır, VLM çağrıları kuyruktan asenkron
tüketilir. Model yavaşsa kareler düşer, akış donmaz.

    # dosyayla dene (kamera gerekmez)
    python tools/rtsp_stream.py --source data/videos/ornek.mp4 --loop

    # webcam
    python tools/rtsp_stream.py --source 0

    # saha kamerası
    python tools/rtsp_stream.py --source rtsp://kullanici:sifre@10.0.0.12:554/stream1

Sağlayıcı `.env` üzerinden gelir; MODEL_PROVIDER=mock ile modelsiz de akış
mantığı test edilebilir.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, strftime
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import config  # noqa: E402
from utils.demo_pipeline import use_utf8_stdout  # noqa: E402
from utils.label_json import dedupe_events, label_to_spec, parse_json  # noqa: E402
from utils.model_client import ModelCallError, chat_vlm  # noqa: E402
from utils.risk_rules import refine_label  # noqa: E402
from utils.scene_evidence import SceneEvidence  # noqa: E402
from utils.spec_output import seconds_to_mmss  # noqa: E402

PROMPT_PATH = ROOT / "prompts" / "video_label_prompt.txt"
ALERTS_DIR = ROOT / "data" / "stream_alerts"

RISK_MARK = {"Yüksek": "[KRİTİK]", "Orta": "[ORTA]  ", "Düşük": "[DÜŞÜK] "}


@dataclass
class StreamConfig:
    source: str = "0"
    sample_fps: float = 4.0
    """Saniyede kaç kare inceleyeceğiz. Hareket ölçümü için 3-5 yeterli."""
    clip_frames: int = 4
    """Tetiklenmede VLM'e gidecek kare sayısı (tetik öncesi + sonrası)."""
    pre_frames: int = 2
    motion_threshold: float = 12.0
    """Kare farkı skoru eşiği; ortam gürültülüyse yükselt."""
    cooldown_s: float = 8.0
    """Aynı olayı tekrar tekrar raporlamamak için tetik sonrası sessizlik."""
    max_side: int = 512
    max_workers: int = 2
    loop_file: bool = False
    save_alerts: bool = True


@dataclass
class Incident:
    trigger_t: float
    motion_score: float
    frames: list[dict[str, Any]] = field(default_factory=list)
    """[{'path': ..., 'time': 'MM:SS', 't_sec': float}]"""


class MotionWakeUp:
    """Ardışık kare farkından hareket skoru; wake-up katmanının canlı hali."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._prev = None
        self.scores: deque[float] = deque(maxlen=60)

    def score(self, frame) -> float:
        import cv2

        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._prev is None:
            self._prev = gray
            return 0.0
        value = float(cv2.absdiff(gray, self._prev).mean())
        self._prev = gray
        self.scores.append(value)
        return value

    def triggered(self, value: float) -> bool:
        """Sabit eşik + ortama uyarlanan eşik: gürültülü sahnede yanlış tetik azalır."""
        if len(self.scores) < 10:
            return value >= self.threshold
        ordered = sorted(self.scores)
        median = ordered[len(ordered) // 2]
        return value >= max(self.threshold, median * 2.5)


class StreamReader(threading.Thread):
    """Akışı canlı tutar; tetiklenen klipleri kuyruğa bırakır. Analizi beklemez."""

    def __init__(self, cfg: StreamConfig, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.loop = loop
        self.queue = queue
        self.stop_event = threading.Event()
        self.frames_seen = 0
        self.triggers = 0
        self.dropped = 0
        self.tmp_dir = ALERTS_DIR / "_frames"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:  # noqa: C901 - okuma döngüsü tek parça daha okunur
        import cv2

        source: str | int = self.cfg.source
        if str(source).isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Akış açılamadı: {self.cfg.source}")
            self.stop_event.set()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(int(round(fps / max(self.cfg.sample_fps, 0.5))), 1)
        wake = MotionWakeUp(self.cfg.motion_threshold)
        buffer: deque[tuple[float, Any]] = deque(maxlen=max(self.cfg.pre_frames, 1))
        pending: Incident | None = None
        last_trigger = -1e9
        index = 0
        print(
            f"Akış açıldı: {self.cfg.source} | kaynak fps {fps:.0f}, "
            f"örnekleme {self.cfg.sample_fps:.1f}/sn, eşik {self.cfg.motion_threshold}"
        )

        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if self.cfg.loop_file and not str(source).isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    index = 0
                    continue
                break
            index += 1
            if index % step:
                continue

            t_sec = index / fps
            self.frames_seen += 1
            value = wake.score(frame)

            if pending is not None:
                pending.frames.append(self._save(frame, t_sec))
                if len(pending.frames) >= self.cfg.clip_frames:
                    self._dispatch(pending)
                    pending = None
                continue

            buffer.append((t_sec, frame.copy()))
            if wake.triggered(value) and (t_sec - last_trigger) >= self.cfg.cooldown_s:
                last_trigger = t_sec
                self.triggers += 1
                pending = Incident(trigger_t=t_sec, motion_score=value)
                pending.frames = [self._save(f, ts) for ts, f in buffer]
                buffer.clear()
                print(f"  ~ hareket tetiklendi  t={seconds_to_mmss(t_sec)}  skor={value:.1f}")

        cap.release()
        if pending is not None and pending.frames:
            self._dispatch(pending)
        self.stop_event.set()
        print(f"Akış bitti. İncelenen kare: {self.frames_seen}, tetiklenme: {self.triggers}")

    def _save(self, frame, t_sec: float) -> dict[str, Any]:
        import cv2

        height, width = frame.shape[:2]
        if max(height, width) > self.cfg.max_side:
            scale = self.cfg.max_side / float(max(height, width))
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        path = self.tmp_dir / f"f_{int(t_sec * 1000):09d}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return {"path": str(path), "time": seconds_to_mmss(t_sec), "t_sec": round(t_sec, 2)}

    def _dispatch(self, incident: Incident) -> None:
        """Kuyruk doluysa klibi düşür — akışı yavaşlatmaktan iyidir."""
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, incident)
        except asyncio.QueueFull:
            self.dropped += 1
            print("  ! kuyruk dolu, klip düşürüldü (model yetişemiyor)")
        except RuntimeError:
            pass


def _incident_prompt(incident: Incident) -> str:
    times = ", ".join(frame["time"] for frame in incident.frames)
    return (
        "Canlı kamera akışında hareket tetiklendi; aşağıdaki kareler o ana ait.\n"
        f"Tetik zamanı: {seconds_to_mmss(incident.trigger_t)} "
        f"(hareket skoru {incident.motion_score:.1f})\n"
        f"Kare zamanları (events[].time SADECE bunlardan biri): {times}\n"
        "Kareler sıralı. Gerçekten kaza/tehlike yoksa category=normal, risk=Düşük yaz; "
        "tetiklenme tek başına olay kanıtı değildir.\n\n"
        + PROMPT_PATH.read_text(encoding="utf-8")
    )


async def analyze_incident(incident: Incident, index: int) -> dict[str, Any] | None:
    started = perf_counter()
    try:
        result = await chat_vlm(
            _incident_prompt(incident),
            [frame["path"] for frame in incident.frames],
            json_mode=True,
            max_tokens=512,
        )
    except ModelCallError as exc:
        print(f"  x analiz başarısız (t={seconds_to_mmss(incident.trigger_t)}): {exc}")
        return None

    parsed = parse_json(result.text) or {}
    evidence = SceneEvidence(
        motion_peak_sec=incident.trigger_t,
        motion_peak_score=incident.motion_score,
        motion_elevated=True,
    )
    label = refine_label(
        {
            "summary": parsed.get("summary") or "",
            "events": dedupe_events(parsed.get("events") or []),
            "risk": parsed.get("risk") or "Düşük",
            "category": parsed.get("category") or "normal",
            "actions": parsed.get("actions") or [],
        },
        evidence,
    )
    spec = label_to_spec(label)
    elapsed = perf_counter() - started

    mark = RISK_MARK.get(spec.get("risk", "Düşük"), "[?]     ")
    print(
        f"{mark} {strftime('%H:%M:%S')}  akış t={seconds_to_mmss(incident.trigger_t)}  "
        f"({elapsed:.1f} sn)  {spec.get('summary', '')[:90]}"
    )
    for event in spec.get("events") or []:
        print(f"           olay {event.get('time')}  {event.get('event', '')[:80]}")
    if spec.get("risk") in {"Orta", "Yüksek"}:
        for action in (spec.get("actions") or [])[:3]:
            print(f"           aksiyon: {action[:80]}")

    return {
        "index": index,
        "trigger_time": seconds_to_mmss(incident.trigger_t),
        "trigger_t_sec": incident.trigger_t,
        "motion_score": round(incident.motion_score, 2),
        "frames": incident.frames,
        "spec": spec,
        "label": label,
        "latency_s": round(elapsed, 2),
        "provider": result.provider,
        "model": result.model,
    }


async def consume(queue: asyncio.Queue, cfg: StreamConfig, reader: StreamReader) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    inflight: set[asyncio.Task] = set()
    index = 0

    async def finish(tasks: set[asyncio.Task]) -> None:
        for task in tasks:
            record = await task
            if record:
                alerts.append(record)

    while not (reader.stop_event.is_set() and queue.empty()):
        try:
            incident = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            done = {task for task in inflight if task.done()}
            inflight -= done
            await finish(done)
            continue

        index += 1
        inflight.add(asyncio.create_task(analyze_incident(incident, index)))
        while len(inflight) >= cfg.max_workers:
            done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            await finish(done)

    if inflight:
        done, _ = await asyncio.wait(inflight)
        await finish(done)
    return sorted(alerts, key=lambda item: item["index"])


async def run(cfg: StreamConfig, duration_s: float) -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=cfg.max_workers * 2)
    reader = StreamReader(cfg, loop, queue)
    reader.start()

    consumer = asyncio.create_task(consume(queue, cfg, reader))
    if duration_s > 0:
        await asyncio.sleep(duration_s)
        reader.stop_event.set()
    try:
        alerts = await consumer
    except asyncio.CancelledError:
        reader.stop_event.set()
        raise
    reader.stop_event.set()
    return alerts


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="RTSP URL, kamera indeksi (0) veya video dosyası")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--motion-threshold", type=float, default=12.0)
    parser.add_argument("--cooldown", type=float, default=8.0)
    parser.add_argument("--clip-frames", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2, help="Eşzamanlı VLM analizi sayısı")
    parser.add_argument("--duration", type=float, default=0.0, help="0 = akış bitene / Ctrl+C'ye kadar")
    parser.add_argument("--loop", action="store_true", help="Dosya kaynağını başa sar (canlı taklidi)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = StreamConfig(
        source=args.source,
        sample_fps=args.sample_fps,
        motion_threshold=args.motion_threshold,
        cooldown_s=args.cooldown,
        clip_frames=args.clip_frames,
        max_workers=max(args.workers, 1),
        loop_file=args.loop,
        save_alerts=not args.no_save,
    )

    endpoint = config.vlm_endpoint()
    print(f"VLM: {endpoint.provider} / {endpoint.model}")
    try:
        alerts = asyncio.run(run(cfg, args.duration))
    except KeyboardInterrupt:
        print("\nDurduruldu.")
        return

    print("\n" + "=" * 60)
    print(f"Toplam uyarı: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert['trigger_time']}  {alert['spec'].get('risk')}  {alert['spec'].get('summary', '')[:70]}")
    if alerts:
        latencies = [alert["latency_s"] for alert in alerts]
        print(f"  Analiz gecikmesi: ortalama {sum(latencies) / len(latencies):.1f} sn, en kötü {max(latencies):.1f} sn")

    if cfg.save_alerts and alerts:
        ALERTS_DIR.mkdir(parents=True, exist_ok=True)
        out = ALERTS_DIR / f"stream_{strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(
            json.dumps({"source": cfg.source, "alerts": alerts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Kayıt: {out}")


if __name__ == "__main__":
    main()
