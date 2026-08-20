"""VLM'den gelen JSON etiketini ayrıştırma ve zaman hizalama yardımcıları.

KPI yolu (scripts/auto_label_qwen.py) ve demo yolu (utils/demo_pipeline.py)
aynı fonksiyonları kullanır; davranış ikiye ayrılmasın.
"""

from __future__ import annotations

import json
import re
from typing import Any

from utils.spec_output import mmss_to_seconds


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
    window_sec: int = 3,
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
    return {
        "summary": label.get("summary", ""),
        "events": [
            {"time": event.get("time", "00:00"), "event": event.get("event", "")}
            for event in (label.get("events") or [])
            if isinstance(event, dict) and event.get("event")
        ],
        "risk": label.get("risk", "Orta"),
        "actions": [a for a in (label.get("actions") or []) if a],
    }
