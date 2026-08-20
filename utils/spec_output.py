"""Arkadaşın pipeline çıktısını şartname / gold JSON formatına çevirir.

Gold format:
{
  "summary": "...",
  "events": [{"time": "00:15", "event": "..."}],
  "risk": "Düşük" | "Orta" | "Yüksek",
  "actions": ["..."]
}

Eski alan adları (olay_ozeti, risk_seviyesi, ...) silinmez; bu modül
onların üzerine şartname kopyasını üretir.
"""

from __future__ import annotations

import re
from typing import Any

SPEC_RISKS = ("Düşük", "Orta", "Yüksek")

# Gold etiketleme sözleşmesi: üç sınıf ↔ üç risk, aynı skala.
CATEGORY_TO_RISK = {
    "normal": "Düşük",
    "near_miss": "Orta",
    "accident": "Yüksek",
}
RISK_TO_CATEGORY = {risk: cat for cat, risk in CATEGORY_TO_RISK.items()}
_CAT_RANK = {"normal": 0, "near_miss": 1, "accident": 2}
_LOCK_POLICIES = ("severity_max", "category", "risk")

_RISK_MAP = {
    "güvenli": "Düşük",
    "guvenli": "Düşük",
    "normal": "Düşük",
    "düşük": "Düşük",
    "dusuk": "Düşük",
    "orta": "Orta",
    "kritik": "Yüksek",
    "yüksek": "Yüksek",
    "yuksek": "Yüksek",
}

_RISK_RANK = {"Düşük": 0, "Orta": 1, "Yüksek": 2}


def mmss_to_seconds(stamp: str | None) -> int:
    """'00:15' → 15. Bozuksa 0."""
    if not stamp:
        return 0
    stamp = stamp.strip()
    if re.fullmatch(r"\d{1,2}:\d{2}", stamp):
        mm, ss = stamp.split(":")
        return int(mm) * 60 + int(ss)
    try:
        return max(int(round(float(stamp))), 0)
    except ValueError:
        return 0


def seconds_to_mmss(seconds: float | int | str | None) -> str:
    """12.4 saniye → 00:12. 'Video Sonu' gibi metinler 00:00 olur."""
    if seconds is None:
        return "00:00"
    if isinstance(seconds, str):
        if re.fullmatch(r"\d{1,2}:\d{2}", seconds.strip()):
            mm, ss = seconds.strip().split(":")
            return f"{int(mm):02d}:{int(ss):02d}"
        try:
            seconds = float(seconds.replace(",", "."))
        except ValueError:
            return "00:00"
    total = max(int(round(float(seconds))), 0)
    return f"{total // 60:02d}:{total % 60:02d}"


def normalize_risk(raw: str | None) -> str:
    """Güvenli/Kritik gibi eski kelimeleri gold'daki Düşük/Orta/Yüksek'e çevirir."""
    if not raw:
        return "Orta"
    key = raw.strip().lower()
    key = key.replace(".", "").split()[0] if key else ""
    return _RISK_MAP.get(key, "Orta")


def risk_from_category(category: str | None, raw_risk: str | None = None) -> str:
    """normal→Düşük, near_miss→Orta, accident→Yüksek. Kategori yoksa ham risk."""
    _, risk = lock_pair(category, raw_risk, policy="category")
    return risk


def lock_policy_name(policy: str | None = None) -> str:
    """LOCK_POLICY env: severity_max (varsayılan) | category | risk."""
    if policy:
        key = str(policy).strip().lower().replace("-", "_")
    else:
        from utils.config import lock_policy as _env_lock

        key = _env_lock()
    aliases = {
        "max": "severity_max",
        "hotter": "severity_max",
        "category_primary": "category",
        "risk_primary": "risk",
    }
    key = aliases.get(key, key)
    return key if key in _LOCK_POLICIES else "severity_max"


def lock_pair(
    category: str | None,
    raw_risk: str | None,
    policy: str | None = None,
) -> tuple[str, str]:
    """Kategori ve riski tek karara indirger.

    severity_max: anlaşmazlıkta daha ağır sinyal (7B'de risk kazayı daha sık yakalıyordu).
    category: risk kategoriyi izler.
    risk: kategori riski izler.
    """
    risk = normalize_risk(raw_risk)
    cat = str(category or "").strip().lower()
    if cat not in _CAT_RANK:
        return RISK_TO_CATEGORY.get(risk, "normal"), risk
    mode = lock_policy_name(policy)
    if mode == "category":
        return cat, CATEGORY_TO_RISK[cat]
    if mode == "risk":
        return RISK_TO_CATEGORY[risk], risk
    rank = max(_CAT_RANK[cat], _RISK_RANK.get(risk, 0))
    inv_cat = {0: "normal", 1: "near_miss", 2: "accident"}
    inv_risk = {0: "Düşük", 1: "Orta", 2: "Yüksek"}
    return inv_cat[rank], inv_risk[rank]


def lock_category_risk(label: dict[str, Any], policy: str | None = None) -> dict[str, Any]:
    """Label dict'te kategori ve riski kilitler (yerinde)."""
    cat, risk = lock_pair(label.get("category"), label.get("risk"), policy=policy)
    label["category"] = cat
    label["risk"] = risk
    return label


def align_risk_to_category(label: dict[str, Any]) -> dict[str, Any]:
    """Eski ad: kategori birincil kilit."""
    return lock_category_risk(label, policy="category")


def parse_vlm_line(text: str | None) -> dict[str, str]:
    """'Durum Açıklaması: ... | Risk: ... | Aksiyon: ...' satırını parçalar."""
    text = (text or "").strip()
    summary = text
    risk = ""
    action = ""

    dur = re.search(r"Durum Açıklaması\s*:\s*(.+?)(?:\s*\|\s*|$)", text, re.IGNORECASE)
    if dur:
        summary = dur.group(1).strip()
    risk_m = re.search(r"Risk\s*:\s*([^|]+)", text, re.IGNORECASE)
    if risk_m:
        risk = risk_m.group(1).strip()
    act_m = re.search(r"Aksiyon\s*:\s*(.+)$", text, re.IGNORECASE)
    if act_m:
        action = act_m.group(1).strip()

    return {"summary": summary, "risk": risk, "action": action}


def pipeline_result_to_spec(
    result: dict[str, Any] | None,
    time_sec: float | int | str | None = None,
) -> dict[str, Any]:
    """Tek tetik (run_pipeline) çıktısını gold JSON'a çevirir."""
    result = result or {}
    parsed = parse_vlm_line(str(result.get("olay_ozeti") or ""))
    time_hint = time_sec if time_sec is not None else result.get("zaman_damgasi")
    event_text = parsed["summary"] or "Olay açıklaması yok"
    risk = normalize_risk(result.get("risk_seviyesi") or parsed["risk"])
    actions = [a for a in (result.get("onerilen_aksiyonlar") or []) if a]
    if not actions and parsed["action"]:
        actions = [parsed["action"]]

    return {
        "summary": event_text,
        "events": [{"time": seconds_to_mmss(time_hint), "event": event_text}],
        "risk": risk,
        "actions": actions,
    }


def _max_risk(values: list[str]) -> str:
    if not values:
        return "Düşük"
    return max(values, key=lambda r: _RISK_RANK.get(r, 0))


def incidents_to_spec(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Video boyunca biriken tetiklerin hepsini tek gold JSON yapar.

    entries öğesi: saniye, vlm_result (pipeline dict). Future çözülmüş olmalı.
    """
    if not entries:
        return {
            "summary": "Videoda kaza veya tehlikeli yaklaşma görülmedi. Rutin saha hareketi.",
            "events": [{"time": "00:00", "event": "Rutin saha hareketi, kaza yok"}],
            "risk": "Düşük",
            "actions": ["Rutin izlemeye devam et"],
        }

    events: list[dict[str, str]] = []
    summaries: list[str] = []
    risks: list[str] = []
    actions: list[str] = []

    for entry in entries:
        spec = pipeline_result_to_spec(entry.get("vlm_result") or {}, entry.get("saniye"))
        events.extend(spec["events"])
        summaries.append(spec["summary"])
        risks.append(spec["risk"])
        for act in spec["actions"]:
            if act not in actions:
                actions.append(act)

    unique_summaries = list(dict.fromkeys(summaries))
    summary = unique_summaries[0] if len(unique_summaries) == 1 else " ".join(unique_summaries[:2])
    return {
        "summary": summary,
        "events": events,
        "risk": _max_risk(risks),
        "actions": actions or ["Rutin izlemeye devam et"],
    }
