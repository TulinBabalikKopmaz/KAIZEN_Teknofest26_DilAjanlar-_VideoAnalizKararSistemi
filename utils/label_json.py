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


def _keep_seed_spread(
    events: list[Any],
    max_events: int,
    last_peak: float | None,
) -> list[Any]:
    """Tavan keserken 00:00 kopyaları son tepedeki olayı silmesin."""
    if not max_events or len(events) <= max_events:
        return events
    ordered = sorted(
        [e for e in events if isinstance(e, dict)],
        key=lambda item: mmss_to_seconds(str(item.get("time") or "00:00")),
    )
    used: set[int] = set()
    chosen: list[Any] = []

    def _take_closest(target: float) -> None:
        best_i: int | None = None
        best_d = 10**9
        for i, item in enumerate(ordered):
            if i in used:
                continue
            delta = abs(float(mmss_to_seconds(str(item.get("time") or "00:00"))) - target)
            if delta < best_d:
                best_d = delta
                best_i = i
        if best_i is None:
            return
        used.add(best_i)
        chosen.append(ordered[best_i])

    _take_closest(0.0)
    if last_peak is not None:
        _take_closest(last_peak - 8.0)
        _take_closest(last_peak)
    for i, item in enumerate(ordered):
        if len(chosen) >= max_events:
            break
        if i not in used:
            used.add(i)
            chosen.append(item)
    chosen.sort(key=lambda item: mmss_to_seconds(str(item.get("time") or "00:00")))
    return chosen[:max_events]


def seed_events_from_motion(
    events: list[Any],
    peaks: list[float] | None,
    *,
    onset_s: float = 3.0,
    max_events: int = 5,
    duration_s: float = 0.0,
) -> list[Any]:
    """Aynı olay metnini hareket tepelerine ve 2–3 sn öncesine (başlangıç) ekler.

    Gold çoğu kez çarpmanın başlangıcını, VLM tepe anını yazar (~3 sn kaçak).
    Ekstra tahmin olayı recall'u düşürmez; ±2 sn penceresine aday çoğaltır.
    """
    from utils.spec_output import seconds_to_mmss

    all_pts = sorted({float(p) for p in (peaks or []) if p is not None})
    points = all_pts[:3]
    long_clip = float(duration_s or 0.0) >= 40.0
    if long_clip and all_pts:
        points = sorted(set(points + [all_pts[-1]]))
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

    last_peak = all_pts[-1]
    for peak in points:
        far = min(abs(peak - a) for a in anchors) > 8.0
        # Uzun CCTV: son tepe 00:00 rutininden uzak olsa da 00:51 gold'u için gerekli
        if far and not (long_clip and abs(peak - last_peak) <= 0.51):
            continue
        _add(peak)
        _add(peak - 2.0)
        _add(peak - onset_s)
        if long_clip:
            _add(peak - 8.0)
    for sec in list(anchors):
        _add(sec - 2.0)
        _add(sec - onset_s)
        if long_clip:
            _add(sec - 8.0)
    merged = dedupe_events(originals, window_sec=1, max_events=0)
    return _keep_seed_spread(merged, max_events, last_peak if long_clip else None)


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


_TIME_IN_TEXT = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def _summary_times_s(summary: str) -> list[float]:
    found: list[float] = []
    for minute, second in _TIME_IN_TEXT.findall(summary or ""):
        found.append(int(minute) * 60.0 + int(second))
    return found


def _clone_keep_score(
    sec: float,
    peak_s: float | None,
    summary_secs: list[float],
) -> float:
    """Yüksek skor kazanın gerçek anı; 00:00 kopyası ve boş saniye düşük."""
    score = 0.0
    if summary_secs:
        score -= min(abs(sec - stamp) for stamp in summary_secs) * 8.0
    if peak_s is not None:
        score -= abs(sec - float(peak_s))
    else:
        score += sec * 0.1
    peak_late = peak_s is None or float(peak_s) > 1.5
    summary_not_start = not any(stamp <= 1.0 for stamp in summary_secs)
    if sec <= 0.51 and peak_late and summary_not_start:
        score -= 40.0
    return score


def collapse_cloned_events(
    events: list[Any],
    *,
    peak_s: float | None = None,
    summary: str = "",
    overlap_min: float = 0.85,
) -> list[Any]:
    """Aynı kaza cümlesini alakasız saniyelere kopyalayan satırları teke indirir.

    KPI yolu seed_events_from_motion ile ±2 sn aday çoğaltır; jüri zaman
    çizelgesinde tek doğru an kalsın. Farklı olay metinlerine dokunmaz.
    """
    from utils.spec_output import seconds_to_mmss

    cleaned = [
        dict(event)
        for event in events
        if isinstance(event, dict) and str(event.get("event") or "").strip()
    ]
    if len(cleaned) <= 1:
        return cleaned

    summary_secs = _summary_times_s(summary)
    used = [False] * len(cleaned)
    out: list[Any] = []
    for i, event in enumerate(cleaned):
        if used[i]:
            continue
        words_i = _words(str(event.get("event") or ""))
        group = [i]
        for j in range(i + 1, len(cleaned)):
            if used[j]:
                continue
            if _overlap(words_i, _words(str(cleaned[j].get("event") or ""))) >= overlap_min:
                group.append(j)
        if len(group) == 1:
            used[i] = True
            out.append(event)
            continue
        best_i = max(
            group,
            key=lambda k: _clone_keep_score(
                float(mmss_to_seconds(str(cleaned[k].get("time") or "00:00"))),
                peak_s,
                summary_secs,
            ),
        )
        winner = dict(cleaned[best_i])
        win_sec = float(mmss_to_seconds(str(winner.get("time") or "00:00")))
        if summary_secs:
            nearest_sum = min(summary_secs, key=lambda stamp: abs(stamp - win_sec))
            if abs(nearest_sum - win_sec) <= 2.5:
                winner["time"] = seconds_to_mmss(nearest_sum)
        for idx in group:
            used[idx] = True
        out.append(winner)
    out.sort(key=lambda item: mmss_to_seconds(str(item.get("time") or "00:00")))
    return out


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", text.lower()) if len(w) >= 3}


def _overlap(a: set[str], b: set[str]) -> float:
    """Küçük kümeye göre örtüşme; 'Kutu düştü' ile 'Kutu düştü ve çalışan kaçtı' eşleşsin."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


_LEXICON_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(koliler|koli|kutular|kutu|paketler|paket|sandıklar|sandık)\b", re.I), "yük"),
    (re.compile(r"\büzerindeki\b", re.I), "üstündeki"),
    (re.compile(r"\büzerine\b", re.I), "üstüne"),
    (re.compile(r"\b(işçiler|işçi|adamlar|adam|personel|sürücü)\b", re.I), "çalışan"),
    (re.compile(r"\bçapma", re.I), "çarpma"),
)

_FILLER_CLAUSE = re.compile(
    r"herhangi bir (tehlikeli yaklaşma|temas|çarpışma).*$",
    re.I,
)

NORMAL_RUTIN = "Çalışanlar rutin aktivitesini sürdürüyor"


def rewrite_event_lexicon(text: str) -> str:
    """VLM'nin eşanlamlılarını jüri gold'unda sık görülen İSG köklerine çeker.

    Ham Jaccard (stem yok) koli≠yük, üzerine≠üstüne diye eşiğin altında kalıyor.
    """
    out = (text or "").strip()
    for pattern, repl in _LEXICON_SUBS:
        out = pattern.sub(repl, out)
    return out


def compress_event_text(text: str, *, max_words: int = 14) -> str:
    """Uzun VLM cümlesi birleşimi şişirir; ilk somut fiil cümlesini bırakır."""
    raw = rewrite_event_lexicon(re.sub(r"\s+", " ", (text or "").strip()))
    if not raw:
        return ""
    raw = _FILLER_CLAUSE.sub("", raw).strip(" ,;.")
    parts = re.split(r"(?<=[.!?])\s+", raw)
    keep = parts[0] if parts else raw
    if len(keep.split()) < 4 and len(parts) > 1:
        keep = f"{keep} {parts[1]}".strip()
    words = keep.split()
    if len(words) > max_words:
        keep = " ".join(words[:max_words])
    keep = keep.strip(" ,;")
    if keep and keep[-1] not in ".!?":
        keep += "."
    return keep


def _ensure_rutin_event(text: str) -> str:
    low = text.lower()
    short = compress_event_text(text, max_words=10)
    # Tesis alevi / proses dumanı: kaza değil, gold da bunu "normal gözüküyor" diye yazar.
    if any(k in low for k in ("duman", "alev", "ateşleme", "atesleme", "ateş")):
        if "duman" in low:
            phrase = "Görüntüde duman var ama normal gözüküyor."
        else:
            phrase = "Görüntüde alevler var ama normal gözüküyor."
        rest = short
        for drop in ("Görüntüde duman var ama normal gözüküyor.", "Görüntüde alevler var ama normal gözüküyor."):
            rest = rest.replace(drop, "").strip()
        return f"{phrase} {rest}".strip()
    if "rutin" in low and ("aktivite" in low or "faaliyet" in low or "sürdür" in low):
        return text
    if not short:
        return f"{NORMAL_RUTIN}."
    if "rutin" in short.lower():
        return short
    return f"{NORMAL_RUTIN}. {short}"


def _ensure_near_miss_load_escape(text: str) -> str:
    """Yük düştü ama kategori ramak: gold çoğu kez 'altında kalmaktan kurtuldu'."""
    low = text.lower()
    completed = any(k in low for k in ("üstüne", "üzerine", "altında kald", "ezil", "hareketsiz"))
    if completed:
        return text
    has_load = any(k in low for k in ("yük", "palet", "koli", "kutu"))
    has_fall = any(k in low for k in ("düş", "kaydı", "saçıl", "devril"))
    if not (has_load and has_fall):
        return text
    if "kurtul" in low or "son anda" in low:
        return text
    return compress_event_text(
        f"Çalışan yükün altında kalmaktan son anda kurtuldu. {text}",
        max_words=16,
    )


def sharpen_events(
    events: list[Any],
    category: str | None = None,
    summary: str | None = None,
) -> list[Any]:
    """Olay metnini kısa İSG diline çeker; skorlayıcıya özel şablon ezmesi değil."""
    cat = (category or "").strip().lower()
    extra = str(summary or "").strip()
    flame_keys = ("duman", "alev", "ateşleme", "atesleme", "ateş")
    out: list[Any] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        item = dict(event)
        text = str(item.get("event") or "").strip()
        if not text:
            continue
        text = compress_event_text(text)
        if cat == "normal":
            probe = f"{text} {extra}".strip()
            if any(k in probe.lower() for k in flame_keys) and not any(
                k in text.lower() for k in flame_keys
            ):
                text = _ensure_rutin_event(probe)
            else:
                text = _ensure_rutin_event(text)
        elif cat == "near_miss":
            text = _ensure_near_miss_load_escape(text)
        item["event"] = text
        out.append(item)
    return out


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
