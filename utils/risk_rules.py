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
from utils.spec_output import lock_category_risk, normalize_risk

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
    r"yangın",
    r"\bateş\b",
    r"alev",
    r"yaralan",
    r"hareketsiz",
    r"yere\s+seril",
    r"collision",
    r"fell\b",
    r"fire\b",
    r"burn",
    r"çökt",
    r"coktu",
    r"enkaz",
    r"altında\s+kald",
    r"ezdi",
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
# Fiili sonuç: "neredeyse düştü" değil, gerçekten olmuş kaza.
COMPLETED_PATTERNS = [
    r"hareketsiz",
    r"yandı",
    r"yangın",
    r"çökt",
    r"enkaz",
    r"altında\s+kald",
    r"ezil",
    r"devril",
    r"yere\s+düş",
    r"yere\s+seril",
    r"çarptı",
    r"yerde\s+yat",
]
_HEDGE_PREFIX = re.compile(r"(neredeyse|son\s+anda|az\s+kaldı)\s*$", re.IGNORECASE)
_NEG_AFTER = re.compile(
    r"^.{0,28}(gözlemlenmedi|tespit edilmedi|görülmedi|bulunmadı|yok(?:tur)?)",
    re.IGNORECASE,
)
_NEG_BEFORE = re.compile(
    r"(herhangi bir|olmadı|yok|değil).{0,48}$",
    re.IGNORECASE,
)
_ALL_CLEAR = re.compile(
    r"gözlemlenmedi|tespit edilmedi|görülmedi|kaza yok|"
    r"rutin (faaliyet|aktivite|izleme)|normal görünüm",
    re.IGNORECASE,
)


def _joined_text(label: dict[str, Any]) -> str:
    parts = [str(label.get("summary") or "")]
    for event in label.get("events") or []:
        parts.append(str(event.get("event") or ""))
    parts.extend(str(a) for a in (label.get("actions") or []))
    return " ".join(parts).lower()


def _match_any(text: str, patterns: list[str]) -> bool:
    """Negasyon içindeki eşleşmeyi yok say ('tehlikeli yaklaşma gözlemlenmedi')."""
    blob = text or ""
    for pattern in patterns:
        for match in re.finditer(pattern, blob, re.IGNORECASE):
            after = blob[match.end() : match.end() + 32]
            before = blob[max(0, match.start() - 48) : match.start()]
            if _NEG_AFTER.search(after) or _NEG_BEFORE.search(before):
                continue
            return True
    return False


def has_unhedged_accident(text: str) -> bool:
    """Tamamlanmış kaza fiili, hemen önünde 'neredeyse' hedge'i yoksa True."""
    blob = (text or "").lower()
    for pattern in COMPLETED_PATTERNS:
        for match in re.finditer(pattern, blob, re.IGNORECASE):
            prefix = blob[max(0, match.start() - 24) : match.start()]
            after = blob[match.end() : match.end() + 32]
            before = blob[max(0, match.start() - 48) : match.start()]
            if _NEG_AFTER.search(after) or _NEG_BEFORE.search(before):
                continue
            if not _HEDGE_PREFIX.search(prefix.rstrip()):
                return True
    return False


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


def _should_snap_all_clear(
    label: dict[str, Any],
    evidence: SceneEvidence | None,
    has_high: bool,
    has_near: bool,
    text: str,
) -> bool:
    """Pozitif kaza/ramak kanıtı yokken 'olay yok' metnini normalde bırak."""
    if has_high or has_near:
        return False
    if has_unhedged_accident(text):
        return False
    if evidence and (
        evidence.person_vehicle_close
        or evidence.person_vehicle_very_close
        or evidence.fire_suspect
    ):
        return False
    if not _ALL_CLEAR.search(text or ""):
        return False
    hot = (label.get("category") or "normal") in {"near_miss", "accident"}
    return hot or normalize_risk(label.get("risk")) != "Düşük"


def evidence_floor(evidence: SceneEvidence | None) -> tuple[str, str | None]:
    if evidence is None:
        return "Düşük", None
    if evidence.fire_suspect:
        return "Yüksek", "accident"
    # Çok yakın: kaza yoksa near_miss; gold sözleşmesi risk=Orta.
    # Metinde çarpışma/zarar varsa text_floor accident+Yüksek yapar.
    if evidence.person_vehicle_very_close:
        return "Orta", "near_miss"
    if evidence.person_vehicle_close:
        return "Orta", "near_miss"
    return "Düşük", None


def refine_label(label: dict[str, Any], evidence: SceneEvidence | None = None) -> dict[str, Any]:
    """Kopya üzerinde risk/kategori yükseltir; özeti silmez. Zayıf Yüksek'i budar."""
    out = dict(label)
    reasons: list[str] = []
    text = _joined_text(out)
    has_high = _match_any(text, HIGH_PATTERNS)
    has_near = _match_any(text, NEAR_PATTERNS)
    # Model near_miss deyip Yüksek dediyse bu anlaşmazlığı FA budaması ezmesin
    keep_hot_near = (
        str(label.get("category") or "") == "near_miss"
        and normalize_risk(label.get("risk")) == "Yüksek"
    )

    text_risk, text_cat = text_risk_floor(out)
    ev_risk, ev_cat = evidence_floor(evidence)

    old_risk = normalize_risk(out.get("risk"))
    new_risk = _max_risk(old_risk, _max_risk(text_risk, ev_risk))
    if new_risk != old_risk:
        reasons.append(f"risk {old_risk}→{new_risk}")
        out["risk"] = new_risk

    old_cat = out.get("category") or "normal"
    new_cat = old_cat
    for cand in (text_cat, ev_cat):
        if cand:
            new_cat = _max_cat(new_cat, cand)
    # Metin/kanıt Yüksek ama kategori normal kaldıysa düzelt
    if new_risk == "Yüksek" and new_cat == "normal":
        # Metinde kaza yoksa accident dayatma (false alarm)
        new_cat = "accident" if has_high or (evidence and evidence.fire_suspect) else "near_miss"
    if new_risk == "Orta" and new_cat == "normal":
        new_cat = "near_miss"
    if new_cat != old_cat:
        reasons.append(f"category {old_cat}→{new_cat}")
        out["category"] = new_cat

    # Zayıf Yüksek budama: metin + yangın + çok-yakın yoksa FA metriğini düşür
    weak_high = (
        normalize_risk(out.get("risk")) == "Yüksek"
        and not has_high
        and not (evidence and evidence.fire_suspect)
        and not (evidence and evidence.person_vehicle_very_close)
    )
    if weak_high and not keep_hot_near:
        if has_near or (evidence and evidence.person_vehicle_close):
            out["risk"] = "Orta"
            if (out.get("category") or "normal") == "accident":
                out["category"] = "near_miss"
            reasons.append("zayıf Yüksek→Orta (FA dampen)")
        elif not (evidence and evidence.motion_elevated):
            out["risk"] = "Düşük"
            out["category"] = "normal"
            reasons.append("zayıf Yüksek→Düşük/normal (FA dampen)")
        else:
            # Sadece hareket var: Orta/near_miss tavanı
            out["risk"] = "Orta"
            if (out.get("category") or "") == "accident":
                out["category"] = "near_miss"
            reasons.append("zayıf Yüksek→Orta (yalnız hareket)")

    # Metin tamamlanmış kazayı anlatıyorsa near_miss hedge'ini ezme
    if has_unhedged_accident(text) and (out.get("category") or "") != "accident":
        reasons.append(f"category {out.get('category')}→accident (tamamlanmış sonuç)")
        out["category"] = "accident"

    # VLM 'olay yok' deyip kuralın şişirdiği rutin saha (ateşleme, negatif cümle)
    if _should_snap_all_clear(out, evidence, has_high, has_near, text):
        reasons.append("rutin saha / olay yok → normal")
        out["risk"] = "Düşük"
        out["category"] = "normal"

    # Kaza/ramak: klip saati → orijinal saat, tepeye hizala, tepe±2sn aday ekle
    if (out.get("category") or "") in {"accident", "near_miss"} and evidence is not None:
        from utils.label_json import (
            align_events_to_motion,
            lift_clip_relative_times,
            preferred_incident_peak_s,
            seed_events_from_motion,
        )
        from utils.video_clip import clip_span

        peaks = list(evidence.motion_peaks or [])
        if evidence.motion_peak_sec is not None:
            peaks.append(float(evidence.motion_peak_sec))
        clip_start, _clip_end = clip_span(
            float(evidence.duration_sec or 0.0),
            evidence.motion_peak_sec,
            evidence.motion_peaks,
        )
        events = lift_clip_relative_times(out.get("events") or [], clip_start, peaks)
        anchor = preferred_incident_peak_s(
            peaks,
            evidence.motion_peak_sec,
            duration_s=evidence.duration_sec,
            category=str(out.get("category") or ""),
        )
        events = align_events_to_motion(events, peaks, primary_peak_s=anchor)
        events = seed_events_from_motion(events, peaks)
        if events:
            out["events"] = events

    # Anlaşmazlıkta daha ağır sinyali al (varsayılan severity_max)
    old_cat = out.get("category") or "normal"
    old_locked = normalize_risk(out.get("risk"))
    lock_category_risk(out)
    if out.get("category") != old_cat or normalize_risk(out.get("risk")) != old_locked:
        reasons.append(
            f"kilit {old_cat}/{old_locked}→{out.get('category')}/{out.get('risk')}"
        )

    # Kazada aksiyon hâlâ "rutin izle" ise güçlendir
    actions = [a for a in (out.get("actions") or []) if a]
    joined = " ".join(actions).lower()
    if normalize_risk(out.get("risk")) == "Yüksek" and (
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
    """Model sakin/ramak kala dedi ama sensör şüpheliyse tekrar sor.

    7B kazayı sık near_miss yazıyor; hareket tepe + near_miss ise bir kez daha bak.
    """
    if evidence is None or not evidence.suggests_second_look():
        return False
    risk = normalize_risk(label.get("risk"))
    cat = label.get("category") or "normal"
    if risk == "Düşük" or cat == "normal":
        return True
    # Kategori near_miss kaldıysa (risk Yüksek olsa bile) kazayı kaçırmış olabilir
    if cat == "near_miss" and evidence.motion_elevated:
        return True
    return False
