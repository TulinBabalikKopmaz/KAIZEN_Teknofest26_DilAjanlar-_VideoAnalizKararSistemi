"""Ekran ve sunum kopyası. Şartname JSON alan adları / token'ları değişmez.

Jüri çıktısı: risk ∈ {Düşük, Orta, Yüksek}, events/summary/actions.
İç kod: category ∈ {normal, near_miss, accident}.
İnsan yüzü: rutin operasyon / ramak kala / iş kazası.
"""

from __future__ import annotations

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
