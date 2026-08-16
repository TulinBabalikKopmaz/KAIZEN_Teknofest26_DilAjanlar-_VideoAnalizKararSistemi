"""Gold cevap anahtarı ile sistem JSON'unu karşılaştırır (KPI)."""

from __future__ import annotations

import re
from typing import Any

from utils.spec_output import mmss_to_seconds, normalize_risk

EVENT_TIME_TOLERANCE_SEC = 2
TEXT_JACCARD_MIN = 0.12


def tokens(text: str | None) -> set[str]:
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+", (text or "").lower())
    return {w for w in words if len(w) >= 2}


def jaccard(a: str | None, b: str | None) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def spec_of(row: dict[str, Any]) -> dict[str, Any]:
    """Hem gold hem pipeline spec hem _spec.json'u aynı forma getirir."""
    inner = row.get("spec") if isinstance(row.get("spec"), dict) else row
    events = []
    for event in inner.get("events") or []:
        text = (event.get("event") or "").strip()
        if not text:
            continue
        events.append({"time": event.get("time") or "00:00", "event": text})
    return {
        "video_id": row.get("video_id") or inner.get("video_id"),
        "filename": row.get("filename") or inner.get("filename"),
        "category": row.get("category") or inner.get("category"),
        "summary": (inner.get("summary") or "").strip(),
        "events": events,
        "risk": normalize_risk(inner.get("risk")),
        "actions": [a for a in (inner.get("actions") or []) if a],
    }


def identity_keys(row: dict[str, Any]) -> set[str]:
    keys = set()
    for field in ("video_id", "filename"):
        value = row.get(field)
        if value:
            keys.add(str(value))
            keys.add(Path_stem(str(value)))
    return {k for k in keys if k}


def Path_stem(name: str) -> str:
    name = name.rsplit("/", 1)[-1]
    if name.endswith(".json"):
        name = name[:-5]
    if name.endswith("_spec"):
        name = name[:-5]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


def event_hits(gold_events: list[dict], pred_events: list[dict], tol: int = EVENT_TIME_TOLERANCE_SEC) -> tuple[int, int]:
    """Gold olaylarından kaçı, ±tol saniye içinde benzer bir tahmin olayıyla eşleşti."""
    hits = 0
    for gold in gold_events:
        gsec = mmss_to_seconds(gold.get("time"))
        for pred in pred_events:
            if abs(mmss_to_seconds(pred.get("time")) - gsec) > tol:
                continue
            if jaccard(gold.get("event"), pred.get("event")) >= TEXT_JACCARD_MIN:
                hits += 1
                break
    return hits, len(gold_events)


def score_video(gold: dict[str, Any], pred: dict[str, Any] | None) -> dict[str, Any]:
    g = spec_of(gold)
    missing = pred is None
    p = spec_of(pred or {})
    hits, total = event_hits(g["events"], p["events"] if not missing else [])
    event_recall = (hits / total) if total else (1.0 if not missing else 0.0)
    critical = g.get("category") in {"accident", "near_miss"} or g["risk"] in {"Orta", "Yüksek"}
    pred_raised = p["risk"] in {"Orta", "Yüksek"} or hits > 0
    critical_hit = (not critical) or (not missing and pred_raised)
    false_alarm = g.get("category") == "normal" and p["risk"] == "Yüksek" and not missing
    summary_ok = (not missing) and (
        g["summary"] == p["summary"] or jaccard(g["summary"], p["summary"]) >= TEXT_JACCARD_MIN
    )
    action_ok = (not missing) and bool(p["actions"])
    if g["actions"] and p["actions"]:
        joined_g = " ".join(g["actions"])
        joined_p = " ".join(p["actions"])
        action_ok = joined_g == joined_p or jaccard(joined_g, joined_p) >= 0.08 or action_ok
    risk_ok = (not missing) and g["risk"] == p["risk"]

    return {
        "video_id": g.get("video_id"),
        "filename": g.get("filename"),
        "category": g.get("category"),
        "missing_pred": missing,
        "risk_gold": g["risk"],
        "risk_pred": None if missing else p["risk"],
        "risk_ok": risk_ok,
        "event_hits": hits,
        "event_total": total,
        "event_recall": round(event_recall, 3),
        "critical": critical,
        "critical_hit": critical_hit,
        "false_alarm": false_alarm,
        "summary_ok": summary_ok,
        "action_ok": action_ok,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    crit = [r for r in rows if r["critical"]]
    normals = [r for r in rows if r.get("category") == "normal"]
    return {
        "n_video": len(rows),
        "risk_accuracy": round(sum(r["risk_ok"] for r in rows) / n, 3),
        "event_recall": round(
            sum(r["event_hits"] for r in rows) / max(sum(r["event_total"] for r in rows), 1),
            3,
        ),
        "critical_recall": round(
            sum(r["critical_hit"] for r in crit) / max(len(crit), 1),
            3,
        ),
        "false_alarm_rate": round(
            sum(r["false_alarm"] for r in normals) / max(len(normals), 1),
            3,
        ),
        "summary_ok_rate": round(sum(r["summary_ok"] for r in rows) / n, 3),
        "action_ok_rate": round(sum(r["action_ok"] for r in rows) / n, 3),
        "missing_predictions": sum(r["missing_pred"] for r in rows),
    }
