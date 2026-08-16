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
