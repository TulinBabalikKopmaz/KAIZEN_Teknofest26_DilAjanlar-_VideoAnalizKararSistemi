"""Ekran ve sunum kopyası. Şartname JSON alan adları / token'ları değişmez.

Jüri çıktısı: risk ∈ {Düşük, Orta, Yüksek}, events/summary/actions.
İç kod: category ∈ {normal, near_miss, accident}.
İnsan yüzü: rutin operasyon / ramak kala / iş kazası.
"""

from __future__ import annotations

import re
from typing import Any

from utils.spec_output import lock_pair, normalize_risk

# Kod anahtarı → saha dili (UI, slayt, konuşma)
CATEGORY_LABEL: dict[str, str] = {
    "normal": "Rutin operasyon",
    "near_miss": "Ramak kala",
    "accident": "İş kazası",
}

# Şartname token → karar cümlesi (metric'te ham "Düşük" basma)
RISK_LABEL: dict[str, str] = {
    "Düşük": "Kontrol altında",
    "Orta": "Yüksek dikkat",
    "Yüksek": "Kritik durum",
}

RISK_SUBTITLE: dict[str, str] = {
    "Düşük": "Acil müdahale gerekmez; izlemeye devam.",
    "Orta": "Kaza olmadı ama eşiğe gelindi; sahayı sıkılaştırın.",
    "Yüksek": "Fiili kaza veya zarar; derhal müdahale.",
}

# Streamlit / CSS tonu
RISK_TONE: dict[str, str] = {
    "Düşük": "ok",
    "Orta": "watch",
    "Yüksek": "critical",
}


def category_label(category: str | None) -> str:
    key = (category or "").strip().lower()
    return CATEGORY_LABEL.get(key, "Değerlendiriliyor")


def risk_label(risk: str | None) -> str:
    return RISK_LABEL.get(normalize_risk(risk), RISK_LABEL["Orta"])


def verdict(category: str | None, risk: str | None) -> dict[str, str]:
    """Kilitli çift için tek karar kartı. JSON'a yazılmaz."""
    cat, rsk = lock_pair(category, risk)
    return {
        "category_key": cat,
        "risk_key": rsk,
        "situation": category_label(cat),
        "decision": risk_label(rsk),
        "subtitle": RISK_SUBTITLE[rsk],
        "tone": RISK_TONE[rsk],
        "kicker": "Saha kararı",
        "spec_risk": rsk,
    }


def spec_footnote() -> str:
    return (
        "Jüri JSON'unda risk alanı şartname token'ıdır: Düşük, Orta, Yüksek. "
        "Ekrandaki 'Kontrol altında / Yüksek dikkat / Kritik durum' aynı skaladır."
    )


def humanize_label(label: dict[str, Any] | None, spec: dict[str, Any] | None = None) -> dict[str, str]:
    row = label or {}
    spec_row = spec or {}
    return verdict(row.get("category"), spec_row.get("risk") or row.get("risk"))


# Rutin sahnede alarm verdirebilecek görünüm (gold: alarm_gorunumlu_normal).
# Şartname JSON'una yazılmaz; karar kartı ve düz Türkçe cevap içindir.
_FLAME_KEYS = ("alev", "ateş", "ates", "yangın", "yangin")
_SMOKE_KEYS = ("duman",)
_SPARK_KEYS = ("kıvılcım", "kivilcim", "kaynak")
_STEAM_KEYS = ("buhar",)
_HEDGE_KEYS = (
    "ama normal",
    "normal gözük",
    "normal goruk",
    "normal görün",
    "normal gorun",
    "proses",
    "tesisin",
    "ateşleme işlem",
    "atesleme islem",
    "olağan",
    "olagan",
)
_ANSWER_ALREADY_COVERS = (
    "proses",
    "ama normal",
    "olağan proses",
    "olagan proses",
    "tesisin olağan",
    "kontrollü işlem",
    "kontrollu islem",
    "kaynak kıvılcım",
    "proses ateş",
    "proses duman",
)

HARD_CASE_COPY: dict[str, str] = {
    "flame": (
        "Ortamda alev görünüyor; bu makinenin olağan proses ateşi, "
        "kaçış veya zarar yok."
    ),
    "smoke": (
        "Görüntüde duman var; tesisin olağan proses dumanı, kaza veya kaçış yok."
    ),
    "spark": (
        "Kaynak kıvılcımı / ateşleme görünüyor; kontrollü işlem, acil müdahale gerekmez."
    ),
    "steam": "Buhar görünümü proses kaynaklı; saha rutin, acil müdahale gerekmez.",
    "sensor": (
        "Sensör alev/duman benzeri renk gördü; saha rutin, proses kaynaklı görünüyor."
    ),
}


def _scene_blob(label: dict[str, Any] | None, spec: dict[str, Any] | None) -> str:
    row = label or {}
    spec_row = spec or {}
    parts: list[str] = [
        str(spec_row.get("summary") or row.get("summary") or ""),
    ]
    for src in (spec_row.get("events"), row.get("events")):
        for event in src or []:
            if isinstance(event, dict):
                parts.append(str(event.get("event") or ""))
            else:
                parts.append(str(event))
    return " ".join(parts).casefold()


def _has_any(blob: str, keys: tuple[str, ...]) -> bool:
    return any(key in blob for key in keys)


def _fire_suspect(evidence: Any) -> bool:
    if evidence is None:
        return False
    if isinstance(evidence, dict):
        return bool(evidence.get("fire_suspect"))
    return bool(getattr(evidence, "fire_suspect", False))


def hard_case_note(
    label: dict[str, Any] | None,
    spec: dict[str, Any] | None = None,
    evidence: Any | None = None,
) -> dict[str, str] | None:
    """Rutin kararda görünen alev/duman sahte alarm değilse ekran cümlesi.

    Kaza / ramak kala çıktısında None döner — gerçek yangını proses diye
    yumuşatmamak için. Şartname alanlarına yazılmaz.
    """
    row = label or {}
    spec_row = spec or {}
    raw_risk = spec_row.get("risk") or row.get("risk")
    if raw_risk:
        cat, _rsk = lock_pair(row.get("category"), raw_risk)
    else:
        cat = str(row.get("category") or "").strip().lower()
    if cat != "normal":
        return None

    blob = _scene_blob(row, spec_row)
    hedged = _has_any(blob, _HEDGE_KEYS)
    cue = ""
    if _has_any(blob, _FLAME_KEYS):
        cue = "flame"
    elif _has_any(blob, _SMOKE_KEYS):
        cue = "smoke"
    elif _has_any(blob, _SPARK_KEYS):
        cue = "spark"
    elif _has_any(blob, _STEAM_KEYS):
        cue = "steam"

    sensor = _fire_suspect(evidence)
    if cue and not (hedged or sensor):
        return None
    if not cue and not sensor:
        return None
    kind = cue or "sensor"
    return {
        "kind": kind,
        "kicker": "Zor sahne",
        "text": HARD_CASE_COPY[kind],
    }


def attach_hard_case_sentence(answer: str, note: dict[str, str] | None) -> str:
    """LLM cevabına proses notunu bir kez ekler; spec JSON'una yazmaz."""
    body = (answer or "").strip()
    if not note:
        return body
    text = str(note.get("text") or "").strip()
    if not text:
        return body
    low = body.casefold()
    if text.casefold() in low:
        return body
    if _has_any(low, _ANSWER_ALREADY_COVERS):
        return body
    if not body:
        return text
    return f"{body.rstrip('.')} {text}"


def model_source(
    provider: str = "",
    model_calls: list[dict[str, Any]] | None = None,
    *,
    backup: bool = False,
) -> dict[str, str]:
    """Kararın hangi motorla geldiği. Ekranda ham teknofest/ollama kodu yok."""
    if backup:
        return {
            "kind": "backup",
            "label": "Kayıtlı yedek",
            "detail": "Canlı model değil",
            "tone": "watch",
        }
    names: list[str] = []
    models: list[str] = []
    fallback = False
    for row in model_calls or []:
        name = str(row.get("provider") or "").strip().lower()
        if name and name not in names:
            names.append(name)
        model = str(row.get("model") or "").strip()
        if model and model not in models:
            models.append(model)
        if row.get("fallback"):
            fallback = True
    if not names:
        head = (provider or "").split(":")[0].strip().lower()
        if head:
            names = [head]
    used_ollama = "ollama" in names or fallback
    used_evren = "teknofest" in names
    if used_ollama and used_evren:
        return {
            "kind": "mixed",
            "label": "Karışık",
            "detail": "EVREN düştü, Ollama devreye girdi",
            "tone": "critical",
        }
    if used_ollama:
        return {
            "kind": "ollama",
            "label": "Ollama",
            "detail": "Yerel yedek — sunum kalitesi değil",
            "tone": "critical",
        }
    if used_evren:
        detail = " · ".join(models) if models else "vlm · llm-fast"
        return {
            "kind": "evren",
            "label": "EVREN",
            "detail": detail,
            "tone": "ok",
        }
    return {
        "kind": "unknown",
        "label": "Bilinmiyor",
        "detail": provider or "kaynak yok",
        "tone": "watch",
    }


def law_support_note(rag_text: str) -> str:
    """Model aksiyonunu ezmez; çizelgenin altında kısa mevzuat teyidi."""
    blob = (rag_text or "").strip()
    if not blob:
        return ""
    found: list[str] = []
    for match in re.finditer(r"(?:madde|md\.?)\s*(\d+)", blob, re.IGNORECASE):
        token = match.group(1)
        if token not in found:
            found.append(token)
    if found:
        refs = ", ".join(f"md. {item}" for item in found[:2])
        return f"Mevzuat da benzer öneriyor ({refs})."
    return "Mevzuat da benzer saha önlemleri öneriyor."


def law_support_card(rag_text: str) -> dict[str, Any] | None:
    """Kısa kicker + madde özetleri. Şartname JSON'una yazılmaz."""
    blob = (rag_text or "").strip()
    if not blob:
        return None
    articles: list[dict[str, str]] = []
    for line in blob.splitlines():
        row = line.strip()
        if not row:
            continue
        if ": " in row:
            title, text = row.split(": ", 1)
        else:
            title, text = "Mevzuat", row
        articles.append({"title": title.strip(), "text": text.strip()})
    if not articles:
        return None
    return {
        "kicker": law_support_note(blob),
        "body": blob,
        "articles": articles,
    }


WATCH_PHASE: dict[str, dict[str, str]] = {
    "idle": {
        "kicker": "Operatör konsolu",
        "title": "Kamera bekleniyor",
        "subtitle": "Güvenlik kamerası veya webcam bağlanınca wake-up izlemeye geçer.",
        "tone": "ok",
    },
    "watching": {
        "kicker": "Canlı izleme",
        "title": "Saha kontrol altında",
        "subtitle": "Wake-up kareleri tarıyor; görsel model uyuyor. Her kare EVREN'e gitmez.",
        "tone": "ok",
    },
    "candidate": {
        "kicker": "Aday pencere",
        "title": "Hareket tetiklendi",
        "subtitle": "Kısa klip kuyruğa alındı. Akış durmadı; karar katmanı bu pencereye bakacak.",
        "tone": "watch",
    },
    "analyzing": {
        "kicker": "Karar katmanı",
        "title": "Görsel model bakıyor",
        "subtitle": "EVREN bu klibi okuyor. Uyarı kartı model dönünce güncellenir.",
        "tone": "watch",
    },
}


def watch_banner(
    phase: str,
    category: str | None = None,
    risk: str | None = None,
) -> dict[str, str]:
    """Canlı konsol kopyası. Şartname token'ı ve ham accident basılmaz."""
    key = (phase or "").strip().lower()
    if key in {"decided", "alert"}:
        v = verdict(category, risk)
        return {
            "kicker": "Saha kararı",
            "title": f"{v['situation']} · {v['decision']}",
            "subtitle": v["subtitle"],
            "tone": v["tone"],
            "situation": v["situation"],
            "decision": v["decision"],
        }
    row = dict(WATCH_PHASE.get(key) or WATCH_PHASE["idle"])
    row["situation"] = row["title"]
    row["decision"] = ""
    return row
