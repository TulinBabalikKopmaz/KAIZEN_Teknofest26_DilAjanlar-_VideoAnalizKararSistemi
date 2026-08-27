"""Canlı saha izleme: webcam / RTSP → wake-up → kısa klip → VLM.

Jüri demo pipeline'ına dokunmaz. Akış analiz için beklemez; model yavaşsa
kare düşer. EVREN `vlm` JPEG reddeder, tetik kareleri kısa mp4 olur.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, strftime
from typing import Any

from utils import config
from utils.display import law_support_note, watch_banner
from utils.evren_rag import retrieve_mevzuat_lexical
from utils.label_json import dedupe_events, label_to_spec, parse_json
from utils.model_client import ModelCallError, chat_vlm
from utils.risk_rules import has_unhedged_accident
from utils.spec_output import lock_category_risk, seconds_to_mmss
from utils.video_clip import frames_to_clip

ROOT = Path(__file__).resolve().parents[1]
ALERTS_DIR = ROOT / "data" / "stream_alerts"
STATUS_PATH = ALERTS_DIR / "live_status.json"

RISK_MARK = {"Yüksek": "[KRİTİK]", "Orta": "[ORTA]  ", "Düşük": "[DÜŞÜK] "}

# Canlı özet "kaçındı" derken kartı kaza yapma. Jüri kural dosyasına dokunulmaz.
_LIVE_NEAR = re.compile(
    r"kaçınd|kaçınarak|kaçt[ıi]|neredeyse|son\s+anda|ramak|az\s+kaldı|yakınından",
    re.IGNORECASE,
)
_LIVE_COMPLETED = re.compile(
    r"hareketsiz|yerde\s+yat|yere\s+seril|yaralan|enkaz|altında\s+kald",
    re.IGNORECASE,
)


def encode_jpeg(frame: Any, quality: int = 70, max_width: int = 0) -> bytes:
    """OpenCV karesini tam JPEG baytına çevirir (dosyaya yarım yazılmaz)."""
    import cv2

    work = frame
    if max_width and int(work.shape[1]) > max_width:
        height, width = work.shape[:2]
        new_h = max(2, int(height * max_width / width) // 2 * 2)
        work = cv2.resize(work, (max_width, new_h))
    ok, buf = cv2.imencode(".jpg", work, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return bytes(buf)


def looks_like_jpeg(data: bytes) -> bool:
    return len(data) >= 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def select_even_frames(frames: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Pencereyi koruyarak VLM'e gidecek kare sayısını kısar (ilk ve son kalır)."""
    if limit <= 0 or len(frames) <= limit:
        return list(frames)
    if limit == 1:
        return [frames[-1]]
    last = len(frames) - 1
    picked: list[int] = []
    for i in range(limit):
        idx = int(round(i * last / (limit - 1)))
        if not picked or idx != picked[-1]:
            picked.append(idx)
    if picked[-1] != last:
        picked[-1] = last
    return [frames[i] for i in picked]


@dataclass
class StreamConfig:
    source: str = "0"
    sample_fps: float = 5.0
    clip_frames: int = 16
    pre_frames: int = 6
    motion_threshold: float = 12.0
    cooldown_s: float = 8.0
    max_side: int = 512
    max_workers: int = 1
    loop_file: bool = False
    save_alerts: bool = True
    preview_fps: float = 18.0
    vlm_frames: int = 12


@dataclass
class Incident:
    trigger_t: float
    motion_score: float
    frames: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LiveStatus:
    phase: str = "idle"
    source: str = ""
    frames_seen: int = 0
    triggers: int = 0
    dropped: int = 0
    motion_score: float = 0.0
    trigger_time: str = ""
    analyzing: bool = False
    error: str = ""
    preview: str = ""
    spec: dict[str, Any] = field(default_factory=dict)
    label: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    provider: str = ""
    updated: str = ""
    event_time: str = ""
    law_note: str = ""
    law_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        category = (self.label or {}).get("category")
        risk = (self.spec or {}).get("risk")
        has_brief = bool(
            (self.spec or {}).get("summary")
            or (self.spec or {}).get("actions")
            or (self.spec or {}).get("events")
        )
        banner_phase = self.phase
        if banner_phase == "idle" and has_brief:
            banner_phase = "decided"
        banner = watch_banner(banner_phase, category, risk)
        return {
            "phase": self.phase,
            "source": self.source,
            "frames_seen": self.frames_seen,
            "triggers": self.triggers,
            "dropped": self.dropped,
            "motion_score": round(self.motion_score, 2),
            "trigger_time": self.trigger_time,
            "analyzing": self.analyzing,
            "error": self.error,
            "preview": self.preview,
            "spec": self.spec,
            "label": self.label,
            "latency_s": round(self.latency_s, 2),
            "provider": self.provider,
            "updated": self.updated,
            "banner": banner,
            "event_time": self.event_time,
            "law_note": self.law_note,
            "law_detail": self.law_detail,
            "has_brief": has_brief,
        }


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

    def triggered(self, value: float, *, file_mode: bool = False) -> bool:
        # Dosya: kısa ısınma, ham eşik. Webcam: daha uzun baz + medyan çarpanı.
        need = 3 if file_mode else 10
        if len(self.scores) < need:
            return False
        if file_mode:
            return value >= self.threshold
        ordered = sorted(self.scores)
        median = ordered[len(ordered) // 2]
        return value >= max(self.threshold, median * 2.5)


def open_capture(source: str | int):
    import cv2

    if isinstance(source, int) or str(source).isdigit():
        index = int(source)
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(str(source))


class StreamReader(threading.Thread):
    """Akışı canlı tutar; tetiklenen klipleri kuyruğa bırakır. Analizi beklemez."""

    def __init__(
        self,
        cfg: StreamConfig,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        hub: LiveHub | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.loop = loop
        self.queue = queue
        self.hub = hub
        self.generation = hub._generation if hub else 0
        self.stop_event = threading.Event()
        self.frames_seen = 0
        self.triggers = 0
        self.dropped = 0
        self.tmp_dir = ALERTS_DIR / "_frames"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._last_preview = 0.0

    def run(self) -> None:  # noqa: C901
        import cv2

        from utils.demo_pipeline import use_utf8_stdout

        use_utf8_stdout()
        source: str | int = self.cfg.source
        cap = open_capture(source)
        if not cap.isOpened():
            print(f"Akış açılamadı: {self.cfg.source}")
            if self.hub:
                self.hub.set_error(f"Kaynak açılamadı: {self.cfg.source}", generation=self.generation)
            self.stop_event.set()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 1.0:
            fps = 25.0
        step = max(int(round(fps / max(self.cfg.sample_fps, 0.5))), 1)
        wake = MotionWakeUp(self.cfg.motion_threshold)
        buffer: deque[tuple[float, Any]] = deque(maxlen=max(self.cfg.pre_frames, 1))
        pending: Incident | None = None
        last_trigger = -1e9
        index = 0
        is_file = not (isinstance(source, int) or str(source).isdigit())
        wall0 = perf_counter()
        loop_base = 0.0
        preview_dt = 1.0 / max(self.cfg.preview_fps, 5.0)
        print(
            f"Akış açıldı: {self.cfg.source} | kaynak fps {fps:.0f}, "
            f"analiz {self.cfg.sample_fps:.0f}/sn x {self.cfg.clip_frames} kare, "
            f"ekran {self.cfg.preview_fps:.0f} fps, VLM {self.cfg.vlm_frames} kare"
            + (" | dosya gerçek zaman" if is_file else "")
        )
        if self.hub:
            self.hub.set_phase("watching", generation=self.generation)

        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if self.cfg.loop_file and is_file:
                    loop_base += max(index, 1) / fps
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    index = 0
                    wake._prev = None
                    wake.scores.clear()
                    continue
                break
            index += 1
            media_t = loop_base + index / fps
            now = perf_counter()
            if now - self._last_preview >= preview_dt:
                self._preview(frame)
                self._last_preview = now
            if is_file:
                lag = media_t - (now - wall0)
                if lag > 0.002 and self.stop_event.wait(timeout=min(lag, 1.0)):
                    break
            if index % step:
                continue

            t_sec = media_t
            self.frames_seen += 1
            value = wake.score(frame)
            if self.hub:
                self.hub.touch_motion(value, self.frames_seen, generation=self.generation)

            if pending is not None:
                pending.frames.append(self._save(frame, t_sec))
                if len(pending.frames) >= self.cfg.clip_frames:
                    self._dispatch(pending)
                    pending = None
                continue

            buffer.append((t_sec, frame.copy()))
            past_open = (not is_file) or ((t_sec - loop_base) >= 0.4)
            if (
                past_open
                and wake.triggered(value, file_mode=is_file)
                and (t_sec - last_trigger) >= self.cfg.cooldown_s
            ):
                last_trigger = t_sec
                self.triggers += 1
                pending = Incident(trigger_t=t_sec, motion_score=value)
                pending.frames = [self._save(item, ts) for ts, item in buffer]
                buffer.clear()
                print(f"  ~ hareket tetiklendi  t={seconds_to_mmss(t_sec)}  skor={value:.1f}")
                if self.hub:
                    self.hub.mark_candidate(
                        t_sec, value, self.triggers, generation=self.generation
                    )

        cap.release()
        if pending is not None and pending.frames:
            self._dispatch(pending)
        self.stop_event.set()
        print(f"Akış bitti. İncelenen kare: {self.frames_seen}, tetiklenme: {self.triggers}")

    def _preview(self, frame) -> None:
        if not self.hub:
            return
        self.hub.set_preview_jpeg(
            encode_jpeg(frame, quality=52, max_width=480),
            generation=self.generation,
        )

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
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, incident)
        except asyncio.QueueFull:
            self.dropped += 1
            print("  ! kuyruk dolu, klip düşürüldü (model yetişemiyor)")
            if self.hub:
                self.hub.status.dropped = self.dropped
        except RuntimeError:
            pass


def lock_live_label(raw: dict[str, Any]) -> dict[str, Any]:
    """Canlı etiket: kilitli çift. Jüri FA budaması (yalnız hareket → near_miss) yok.

    Özet kaçındı / neredeyse diyorsa kart ramak kala kalır; yerde hareketsiz
    gibi fiili sonuç varsa kaza yükselmesi durur.
    """
    out = {
        "summary": str(raw.get("summary") or ""),
        "events": dedupe_events(raw.get("events") or []),
        "risk": raw.get("risk") or "Düşük",
        "category": raw.get("category") or "normal",
        "actions": list(raw.get("actions") or []),
    }
    blob = f"{out['summary']} {' '.join(str(item.get('event') or '') for item in out['events'])}"
    near = bool(_LIVE_NEAR.search(blob))
    completed = bool(_LIVE_COMPLETED.search(blob))
    if near and not completed:
        out["category"] = "near_miss"
        lock_category_risk(out, policy="category")
        return out
    if has_unhedged_accident(blob) and out["category"] != "accident":
        out["category"] = "accident"
    lock_category_risk(out)
    return out


def pick_event_time(spec: dict[str, Any], trigger_time: str) -> str:
    """Operatör saati: asıl olay saniyesi. Açılış 00:00 tetiklerini ezmez."""
    trigger = (trigger_time or "").strip()
    times = [
        str(item.get("time") or "").strip()
        for item in (spec.get("events") or [])
        if str(item.get("time") or "").strip()
    ]
    later = [item for item in times if item not in {"00:00", "0:00"}]
    if trigger in later:
        return trigger
    if later:
        return later[-1]
    if trigger and trigger not in {"00:00", "0:00"}:
        return trigger
    if times:
        return times[-1]
    return trigger


def attach_live_support(spec: dict[str, Any], trigger_time: str) -> dict[str, str]:
    """Mevzuat notu + algılanan an. Şartname JSON'una yazılmaz; aksiyon listesi aynı kalır."""
    event_time = pick_event_time(spec, trigger_time)
    blob = " ".join(
        [
            str(spec.get("summary") or ""),
            str(spec.get("risk") or ""),
            " ".join(str(item) for item in (spec.get("actions") or []) if item),
            " ".join(str(item.get("event") or "") for item in (spec.get("events") or [])),
        ]
    )
    rag = retrieve_mevzuat_lexical(blob)
    return {
        "event_time": event_time,
        "law_note": law_support_note(rag),
        "law_detail": rag,
    }


def _incident_prompt(incident: Incident, frames: list[dict[str, Any]] | None = None) -> str:
    rows = frames if frames is not None else incident.frames
    times = ", ".join(frame["time"] for frame in rows)
    return (
        "Kısa saha kamerası klibi. Hareket tepesi civarı; kareler zaman sırasıyla.\n"
        f"Tetik: {seconds_to_mmss(incident.trigger_t)} (skor {incident.motion_score:.1f}).\n"
        f"events[].time yalnızca bunlar: {times}\n\n"
        "SON KARELERE BAK. İlk kare sakin olabilir; asıl olay genelde sonra biter "
        "(düşme, devrilme, çarpışma, ezilme, yerde kişi, tutuşma).\n"
        "Fiili sonuç görüyorsan category=accident ve risk=Yüksek. "
        "Kaza olmadı ama neredeyse olduysa (kaçındı, son anda) near_miss / Orta. "
        "Özet ile category çelişmesin. "
        "Net kaza yoksa normal / Düşük. Uydurma; görünür kazayı küçümseme.\n"
        "Hareket tetiklenmesi tek başına kaza değildir.\n"
        "Özet düz Türkçe, 1-2 cümle. Sadece JSON:\n"
        '{"category":"normal|near_miss|accident","summary":"...",'
        '"events":[{"time":"MM:SS","event":"..."}],'
        '"risk":"Düşük|Orta|Yüksek","actions":["..."]}'
    )


async def analyze_incident(
    incident: Incident,
    index: int,
    vlm_frames: int = 12,
) -> dict[str, Any] | None:
    started = perf_counter()
    picked = select_even_frames(incident.frames, vlm_frames)
    image_paths = [frame["path"] for frame in picked]
    video_path: Path | None = None
    if config.vlm_endpoint().provider == "teknofest":
        dest = ALERTS_DIR / "_frames" / f"clip_{index}_{int(incident.trigger_t * 1000)}.mp4"
        try:
            video_path = frames_to_clip(image_paths, dest, fps=8.0, hold=1)
            image_paths = []
        except (OSError, RuntimeError) as exc:
            print(f"  x klip yazılamadı, JPEG denenmeyecek (EVREN reddeder): {exc}")
            return None
    try:
        result = await chat_vlm(
            _incident_prompt(incident, picked),
            image_paths,
            video_path=video_path,
            json_mode=True,
            max_tokens=420,
        )
    except ModelCallError as exc:
        print(f"  x analiz başarısız (t={seconds_to_mmss(incident.trigger_t)}): {exc}")
        return None

    parsed = parse_json(result.text) or {}
    label = lock_live_label(parsed)
    spec = label_to_spec(label)
    trigger_time = seconds_to_mmss(incident.trigger_t)
    support = attach_live_support(spec, trigger_time)
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
        "trigger_time": trigger_time,
        "trigger_t_sec": incident.trigger_t,
        "event_time": support["event_time"],
        "motion_score": round(incident.motion_score, 2),
        "frames": incident.frames,
        "spec": spec,
        "label": label,
        "latency_s": round(elapsed, 2),
        "provider": result.provider,
        "model": result.model,
        "law_note": support["law_note"],
        "law_detail": support["law_detail"],
    }


async def _analyze_tagged(
    incident: Incident,
    index: int,
    vlm_frames: int,
    generation: int,
) -> dict[str, Any] | None:
    record = await analyze_incident(incident, index, vlm_frames)
    if record is None:
        return {"generation": generation, "_failed": True}
    record["generation"] = generation
    return record


async def consume(
    queue: asyncio.Queue,
    cfg: StreamConfig,
    reader: StreamReader,
    hub: LiveHub | None = None,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    inflight: set[asyncio.Task] = set()
    index = 0

    async def finish(tasks: set[asyncio.Task]) -> None:
        for task in tasks:
            record = await task
            if not record:
                continue
            if record.get("_failed"):
                if hub:
                    hub.set_error(
                        "Görsel model bu pencereyi okuyamadı",
                        generation=record.get("generation") if isinstance(record.get("generation"), int) else None,
                    )
                continue
            alerts.append(record)
            if hub:
                hub.mark_decided(record)

    while not (reader.stop_event.is_set() and queue.empty() and not inflight):
        try:
            incident = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            done = {task for task in inflight if task.done()}
            inflight -= done
            await finish(done)
            continue

        index += 1
        started_gen = hub._generation if hub else 0
        if hub:
            hub.set_phase("analyzing", generation=started_gen)
        inflight.add(
            asyncio.create_task(_analyze_tagged(incident, index, cfg.vlm_frames, started_gen))
        )
        while len(inflight) >= cfg.max_workers:
            done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            await finish(done)

    if inflight:
        done, _ = await asyncio.wait(inflight)
        await finish(done)
    return sorted(alerts, key=lambda item: item["index"])


async def run(cfg: StreamConfig, duration_s: float, hub: LiveHub | None = None) -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=max(cfg.max_workers * 2, 2))
    reader = StreamReader(cfg, loop, queue, hub=hub)
    if hub:
        hub.attach_reader(reader)
        hub.status.source = cfg.source
        hub.status.provider = config.vlm_endpoint().provider
        hub.write()
    reader.start()

    consumer = asyncio.create_task(consume(queue, cfg, reader, hub=hub))
    if duration_s > 0:
        await asyncio.sleep(duration_s)
        reader.stop_event.set()
    try:
        alerts = await consumer
    except asyncio.CancelledError:
        reader.stop_event.set()
        raise
    reader.stop_event.set()
    if hub and hub.status.phase not in {"decided", "alert"}:
        hub.set_phase("idle")
    return alerts


class LiveHub:
    """Streamlit / CLI'nin paylaştığı anlık durum."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = LiveStatus()
        self._jpeg = b""
        self._thread: threading.Thread | None = None
        self._reader: StreamReader | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._generation = 0

    def _accept(self, generation: int | None) -> bool:
        return generation is None or generation == self._generation

    def attach_reader(self, reader: StreamReader) -> None:
        self._reader = reader

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.status.to_dict()

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def set_preview_jpeg(self, data: bytes, generation: int | None = None) -> None:
        if not self._accept(generation):
            return
        if not looks_like_jpeg(data):
            return
        with self._lock:
            self._jpeg = data
            self.status.preview = "memory"

    def write(self) -> None:
        ALERTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot()
        STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def set_phase(self, phase: str, generation: int | None = None) -> None:
        if not self._accept(generation):
            return
        with self._lock:
            if phase == "watching" and self.status.phase in {"decided", "analyzing", "candidate"}:
                return
            self.status.phase = phase
            self.status.updated = strftime("%H:%M:%S")
            if phase == "watching":
                self.status.error = ""
        self.write()

    def touch_motion(self, score: float, frames_seen: int, generation: int | None = None) -> None:
        if not self._accept(generation):
            return
        with self._lock:
            self.status.motion_score = score
            self.status.frames_seen = frames_seen

    def mark_candidate(
        self,
        t_sec: float,
        score: float,
        triggers: int,
        generation: int | None = None,
    ) -> None:
        if not self._accept(generation):
            return
        with self._lock:
            self.status.phase = "candidate"
            self.status.trigger_time = seconds_to_mmss(t_sec)
            self.status.event_time = ""
            self.status.motion_score = score
            self.status.triggers = triggers
            self.status.analyzing = True
            self.status.spec = {}
            self.status.label = {}
            self.status.law_note = ""
            self.status.law_detail = ""
            self.status.latency_s = 0.0
            self.status.error = ""
            self.status.updated = strftime("%H:%M:%S")
        self.write()

    def mark_decided(self, record: dict[str, Any]) -> None:
        generation = record.get("generation")
        if isinstance(generation, int) and not self._accept(generation):
            return
        with self._lock:
            self.status.phase = "decided"
            self.status.analyzing = False
            self.status.spec = dict(record.get("spec") or {})
            self.status.label = dict(record.get("label") or {})
            self.status.latency_s = float(record.get("latency_s") or 0.0)
            self.status.provider = str(record.get("provider") or "")
            self.status.trigger_time = str(record.get("trigger_time") or self.status.trigger_time)
            self.status.event_time = str(record.get("event_time") or self.status.trigger_time)
            self.status.law_note = str(record.get("law_note") or "")
            self.status.law_detail = str(record.get("law_detail") or "")
            self.status.error = ""
            self.status.updated = strftime("%H:%M:%S")
        self.write()

    def set_error(self, message: str, generation: int | None = None) -> None:
        if not self._accept(generation):
            return
        with self._lock:
            self.status.error = message
            self.status.analyzing = False
            if self.status.phase == "analyzing":
                self.status.phase = "watching"
            self.status.updated = strftime("%H:%M:%S")
        self.write()

    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, cfg: StreamConfig) -> None:
        self._generation += 1
        gen = self._generation
        if self.running():
            self.stop()
        if self.running():
            self._thread = None
            self._reader = None
            self._loop = None
        self.status = LiveStatus(
            phase="watching",
            source=cfg.source,
            provider=config.vlm_endpoint().provider,
        )
        self._generation = gen
        with self._lock:
            self._jpeg = b""
        self._thread = threading.Thread(target=self._run_thread, args=(cfg,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._reader:
            self._reader.stop_event.set()
        thread = self._thread
        with self._lock:
            self.status.phase = "idle"
            self.status.analyzing = False
            self.status.updated = strftime("%H:%M:%S")
        self.write()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.5)

    def _run_thread(self, cfg: StreamConfig) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run(cfg, duration_s=0.0, hub=self))
        finally:
            loop.close()
            self._loop = None


_HUB: LiveHub | None = None
_HUB_LOCK = threading.Lock()


def get_hub() -> LiveHub:
    global _HUB
    with _HUB_LOCK:
        if _HUB is None:
            _HUB = LiveHub()
        return _HUB
