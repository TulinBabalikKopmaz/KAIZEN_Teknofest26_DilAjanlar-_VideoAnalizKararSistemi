"""Model çıktısını ölçülebilir kanıtlarla düzeltir (cevap anahtarını ezmez).

İki katman:
1) Metin anahtar kelimeleri → risk tabanı (floor)
2) SceneEvidence (hareket / yakınlık / yangın) → risk + kategori yükseltme

Amaç: Qwen 'Orta' deyip olay metninde çarpışma yazarsa veya YOLO
kişi-araç çok yakınsa riski yükseltmek. Normal sahnede şişirmemek.
"""

from __future__ import annotations

import re
from typing import Any

from utils.scene_evidence import SceneEvidence
from utils.spec_output import normalize_risk

_RISK_RANK = {"Düşük": 0, "Orta": 1, "Yüksek": 2}
_CAT_RANK = {"normal": 0, "near_miss": 1, "accident": 2}

# Türkçe + sık İngilizce kaçaklar
HIGH_PATTERNS = [
    r"çarpt",
    r"carp",
    r"çarpış",
    r"devril",
    r"düşt",
    r"dust",
    r"ezil",
    r"yandı",
    r"yanma",
    r"yangın",
    r"ateş",
    r"alev",
    r"yaralan",
    r"hareketsiz",
    r"yere\s+seril",
    r"collision",
    r"fell\b",
    r"fire\b",
    r"burn",
]
NEAR_PATTERNS = [
    r"neredeyse",
    r"son\s+anda",
    r"kaçt",
    r"kaçış",
    r"yakınından",
    r"ramak",
    r"near\s*miss",
    r"tehlikeli\s+yaklaş",
    r"az\s+kaldı",
]


def _joined_text(label: dict[str, Any]) -> str:
    parts = [str(label.get("summary") or "")]
    for event in label.get("events") or []:
        parts.append(str(event.get("event") or ""))
    parts.extend(str(a) for a in (label.get("actions") or []))
    return " ".join(parts).lower()


def _match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _max_risk(a: str, b: str) -> str:
    a_n, b_n = normalize_risk(a), normalize_risk(b)
    return a_n if _RISK_RANK[a_n] >= _RISK_RANK[b_n] else b_n


def _max_cat(a: str, b: str) -> str:
    aa = a if a in _CAT_RANK else "normal"
    bb = b if b in _CAT_RANK else "normal"
    return aa if _CAT_RANK[aa] >= _CAT_RANK[bb] else bb


def text_risk_floor(label: dict[str, Any]) -> tuple[str, str | None]:
    """Metinden minimum risk (+ isteğe bağlı kategori)."""
    text = _joined_text(label)
    # Önce ramak kala: "neredeyse çarpıştı" Yüksek olmasın
    if _match_any(text, NEAR_PATTERNS):
        return "Orta", "near_miss"
    if _match_any(text, HIGH_PATTERNS):
        return "Yüksek", "accident"
    return "Düşük", None


def evidence_floor(evidence: SceneEvidence | None) -> tuple[str, str | None]:
    if evidence is None:
        return "Düşük", None
    if evidence.fire_suspect:
        return "Yüksek", "accident"
    # Çok yakın ≠ fiili kaza; metinde çarpışma yoksa near_miss kalsın
    if evidence.person_vehicle_very_close:
        return "Yüksek", "near_miss"
    if evidence.person_vehicle_close:
        return "Orta", "near_miss"
    return "Düşük", None


def refine_label(label: dict[str, Any], evidence: SceneEvidence | None = None) -> dict[str, Any]:
    """Kopya üzerinde risk/kategori yükseltir; özeti silmez."""
    out = dict(label)
    reasons: list[str] = []

    text_risk, text_cat = text_risk_floor(out)
    ev_risk, ev_cat = evidence_floor(evidence)

    old_risk = normalize_risk(out.get("risk"))
    new_risk = _max_risk(old_risk, _max_risk(text_risk, ev_risk))
    if new_risk != old_risk:
        reasons.append(f"risk {old_risk}→{new_risk}")
        out["risk"] = new_risk

    old_cat = out.get("category") or "normal"
    new_cat = old_cat
    text = _joined_text(out)
    near_text = _match_any(text, NEAR_PATTERNS)
    for cand in (text_cat, ev_cat):
        if cand:
            # Metin açıkça ramak kala diyorsa evidence ile accident'e zorlama
            if near_text and cand == "accident" and not _match_any(text, HIGH_PATTERNS):
                cand = "near_miss"
            new_cat = _max_cat(new_cat, cand)
    # Metin/kanıt Yüksek ama kategori normal kaldıysa düzelt
    if new_risk == "Yüksek" and new_cat == "normal":
        new_cat = "accident" if _match_any(text, HIGH_PATTERNS) else "near_miss"
    if new_risk == "Orta" and new_cat == "normal":
        new_cat = "near_miss"
    if new_cat != old_cat:
        reasons.append(f"category {old_cat}→{new_cat}")
        out["category"] = new_cat

    # Kazada aksiyon hâlâ "rutin izle" ise güçlendir
    actions = [a for a in (out.get("actions") or []) if a]
    joined = " ".join(actions).lower()
    if new_risk == "Yüksek" and (
        not actions or "rutin" in joined or "izlemeye devam" in joined
    ):
        out["actions"] = [
            "Sağlık ekibini çağır",
            "Alanı güvenlik altına al",
        ]
        reasons.append("aksiyon güçlendirildi")

    notes = str(out.get("notes") or "")
    if reasons:
        tag = "Kanıt birleştirme: " + "; ".join(reasons)
        out["notes"] = f"{notes} | {tag}".strip(" |") if notes else tag
    return out


def needs_second_look(label: dict[str, Any], evidence: SceneEvidence | None) -> bool:
    """Model düşük dedi ama sensör şüpheliyse tekrar sor."""
    if evidence is None or not evidence.suggests_second_look():
        return False
    risk = normalize_risk(label.get("risk"))
    cat = label.get("category") or "normal"
    return risk == "Düşük" or cat == "normal"
