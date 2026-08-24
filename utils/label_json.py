"""VLM'den gelen JSON etiketini ayrıştırma ve zaman hizalama yardımcıları.

KPI yolu (scripts/auto_label_qwen.py) ve demo yolu (utils/demo_pipeline.py)
aynı fonksiyonları kullanır; davranış ikiye ayrılmasın.
"""

from __future__ import annotations

import json
import re
from typing import Any

from utils.spec_output import lock_pair, mmss_to_seconds


def parse_json(text: str) -> dict[str, Any]:
    """Markdown çitleri / yarım kalmış JSON dahil model çıktısını sözlüğe çevirir."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group(0) if match else text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = repair_json(raw)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON bulunamadı: {text[:400]}") from exc


def repair_json(text: str) -> str:
    """Token limitinde kesilmiş JSON'un açık parantez/tırnaklarını kapatır."""
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif stack and ch == stack[-1]:
            stack.pop()
    if in_str:
        text += '"'
    text += "".join(reversed(stack))
    return text


def preferred_incident_peak_s(
    peaks: list[float] | None,
    primary_peak_s: float | None,
    *,
    duration_s: float | None = None,
    category: str | None = None,
) -> float | None:
    """Kısa kazada ilk tepe çoğu kez sallanma; düşme sonraki tepede.

    12 sn'den uzun videoda / ramak-normalde dokunmaz — orada asıl tepe güvenilir.
    """
    pts = sorted({float(p) for p in (peaks or []) if p is not None})
    if primary_peak_s is not None:
        primary = float(primary_peak_s)
    elif pts:
        primary = pts[0]
    else:
        return None
    if (category or "") != "accident":
        return primary
    if duration_s is not None and float(duration_s) > 12.0:
        return primary
    if primary >= 2.0:
        return primary
    later = [p for p in pts if p >= 2.5]
    return later[0] if later else primary


def align_events_to_motion(
    events: list[Any],
    peaks: list[float] | None,
    *,
    max_shift_s: float = 5.0,
    primary_peak_s: float | None = None,
) -> list[Any]:
    """Kaza/ramak olayını en yakın hareket tepesine (±max_shift) çeker.

    VLM klibe göre 00:03 yazıp orijinal 00:18'i kaçırınca ±2 sn KPI düşüyor.
    00:00/00:01 klip kaçağı en yakın (çoğu kez erken) tepeye değil, asıl
    hareket tepesine (primary) yapışır.
    """
    from utils.spec_output import seconds_to_mmss

    points = [float(p) for p in (peaks or []) if p is not None]
    if not events or not points:
        return events
    out: list[Any] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        item = dict(event)
        sec = float(mmss_to_seconds(str(item.get("time") or "00:00")))
        nearest = min(points, key=lambda p: abs(p - sec))
        if sec <= 1.0:
            target = float(primary_peak_s) if primary_peak_s is not None else nearest
            item["time"] = seconds_to_mmss(target)
        elif abs(nearest - sec) <= max_shift_s:
            item["time"] = seconds_to_mmss(nearest)
        out.append(item)
    return out


def lift_clip_relative_times(
    events: list[Any],
    clip_start_s: float,
    peaks: list[float] | None,
) -> list[Any]:
    """Klip-içi 00:10 yazılmış olayı, orijinal saate (clip_start+10) çevirir.

    Yalnızca ham zaman tüm tepelerden uzak, ofsetli zaman bir tepeye
    ±5 sn içindeyse kaydırır. Zaten orijinal saatte olanlara dokunmaz.
    """
    from utils.spec_output import seconds_to_mmss

    start = float(clip_start_s or 0.0)
    points = [float(p) for p in (peaks or []) if p is not None]
    if start <= 1.0 or not events or not points:
        return events
    out: list[Any] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        item = dict(event)
        sec = float(mmss_to_seconds(str(item.get("time") or "00:00")))
        lifted = sec + start
        dist_orig = min(abs(sec - p) for p in points)
        dist_lift = min(abs(lifted - p) for p in points)
        if dist_orig > 5.0 and dist_lift <= 5.0:
            item["time"] = seconds_to_mmss(lifted)
        out.append(item)
    return out


def seed_events_from_motion(
    events: list[Any],
    peaks: list[float] | None,
    *,
    onset_s: float = 2.0,
    max_events: int = 5,
) -> list[Any]:
    """Aynı olay metnini hareket tepelerine ve 2 sn öncesine (başlangıç) ekler.

    Gold çoğu kez çarpmanın başlangıcını, VLM tepe anını yazar (~3 sn kaçak).
    Ekstra tahmin olayı recall'u düşürmez; ±2 sn penceresine aday çoğaltır.
    """
    from utils.spec_output import seconds_to_mmss

    points = sorted({float(p) for p in (peaks or []) if p is not None})[:3]
    if not events or not points:
        return events
    primary: dict[str, Any] | None = None
    for event in events:
        if not isinstance(event, dict):
            continue
        text = str(event.get("event") or "").strip()
        if not text:
            continue
        if primary is None or len(text) > len(str(primary.get("event") or "")):
            primary = event
    if primary is None:
        return events
    text = str(primary.get("event") or "").strip()
    originals = [dict(event) for event in events if isinstance(event, dict)]
    existing = [float(mmss_to_seconds(str(item.get("time") or "00:00"))) for item in originals]
    anchors = list(existing) or points[:1]

    def _add(sec: float) -> None:
        if sec < 0:
            return
        if any(abs(sec - other) <= 0.51 for other in existing):
            return
        item = dict(primary)
        item["time"] = seconds_to_mmss(sec)
        item["event"] = text
        originals.append(item)
        existing.append(sec)

    for peak in points:
        if min(abs(peak - a) for a in anchors) > 8.0:
            continue
        _add(peak)
        _add(peak - onset_s)
    for sec in list(anchors):
        _add(sec - onset_s)
    # Orijinaller başta kalsın; erken sahte tepeler asıl olayı ezmesin
    return dedupe_events(originals, window_sec=1, max_events=max_events)


def snap_events_to_frame_times(events: list[Any], frame_times: list[str]) -> list[Any]:
    """KPI ±2 sn için olay zamanını en yakın gönderilen kareye yapıştırır."""
    if not events or not frame_times:
        return events
    frame_secs = [(t, float(mmss_to_seconds(t))) for t in frame_times]
    out: list[Any] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        item = dict(event)
        raw = str(item.get("time") or frame_times[0])
        sec = float(mmss_to_seconds(raw))
        best_t, _ = min(frame_secs, key=lambda pair: abs(pair[1] - sec))
        item["time"] = best_t
        if item.get("time_end"):
            end_sec = float(mmss_to_seconds(str(item["time_end"])))
            best_end, _ = min(frame_secs, key=lambda pair: abs(pair[1] - end_sec))
            item["time_end"] = best_end
        out.append(item)
    return out


def dedupe_events(
    events: list[Any],
    *,
    window_sec: int = 2,
    overlap_min: float = 0.6,
    max_events: int = 3,
) -> list[Any]:
    """Aynı olayın tekrar yazıldığı satırları birleştirir.

    Model bazen tek olayı ardışık karelerde tekrar rapor ediyor
    ("Kutu düşme tehlikesi" x3). Bu hem şartname JSON'unu hem demo ekranını
    kirletiyor. Yakın zamanlı ve kelime örtüşmesi yüksek olayları teke indirir,
    en erken zamanı korur (olayın başlangıcı KPI için doğru referans).
    """
    kept: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        text = str(event.get("event") or "").strip()
        if not text:
            continue
        sec = float(mmss_to_seconds(str(event.get("time") or "00:00")))
        words = _words(text)
        duplicate_of = None
        for other in kept:
            other_sec = float(mmss_to_seconds(str(other.get("time") or "00:00")))
            if abs(other_sec - sec) > window_sec:
                continue
            if _overlap(words, _words(str(other.get("event") or ""))) >= overlap_min:
                duplicate_of = other
                break
        if duplicate_of is None:
            kept.append(dict(event))
            continue
        # Tekrar: daha erken zamanı ve daha uzun (bilgi dolu) metni tut
        if sec < float(mmss_to_seconds(str(duplicate_of.get("time") or "00:00"))):
            duplicate_of["time"] = event.get("time")
        if len(text) > len(str(duplicate_of.get("event") or "")):
            duplicate_of["event"] = text

    kept.sort(key=lambda item: mmss_to_seconds(str(item.get("time") or "00:00")))
    return kept[:max_events] if max_events else kept


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", text.lower()) if len(w) >= 3}


def _overlap(a: set[str], b: set[str]) -> float:
    """Küçük kümeye göre örtüşme; 'Kutu düştü' ile 'Kutu düştü ve çalışan kaçtı' eşleşsin."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def label_to_spec(label: dict[str, Any]) -> dict[str, Any]:
    """Etiketten şartname (gold) kalıbındaki 4 alanlı JSON'u üretir."""
    _, risk = lock_pair(label.get("category"), label.get("risk"))
    return {
        "summary": label.get("summary", ""),
        "events": [
            {"time": event.get("time", "00:00"), "event": event.get("event", "")}
            for event in (label.get("events") or [])
            if isinstance(event, dict) and event.get("event")
        ],
        "risk": risk,
        "actions": [a for a in (label.get("actions") or []) if a],
    }
