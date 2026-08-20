"""Çıktı eleştirmeni — kare yok, yalnız VLM'in kendi metnini denetler.

Görüntüyü yeniden izlemez (o iş ikinci bakışın). Sadece sınıf/risk/özet
çelişince ve süre bütçesi varsa LLM'e sorar; yükseltmeye izin verir,
düşürmez. Her videoda döngüye sokmak demo süresini ve API kotasını yer.
"""

from __future__ import annotations

import json
from typing import Any

from utils.label_json import parse_json
from utils.risk_rules import has_unhedged_accident
from utils.spec_output import _CAT_RANK, lock_pair, normalize_risk

SYSTEM_PROMPT = (
    "Sen İSG etiket denetçisisin. Sana modelin kendi özeti ve olayları gelir, "
    "kare gelmez. Yeni olay uydurma. Sınıfı YALNIZCA metin fiili temas veya "
    "zarar anlatıyorsa accident yap. 'neredeyse', 'son anda', 'çok yakınından "
    "geçti', 'tehlikesi' varsa near_miss kalır. Sadece JSON döndür: "
    '{"category":"normal|near_miss|accident","raise":true|false}'
)


def needs_critic(label: dict[str, Any]) -> bool:
    """LLM eleştirmeni yalnız çelişki veya küçümsenmiş kaza metninde."""
    cat = str(label.get("category") or "normal").strip().lower()
    risk = normalize_risk(label.get("risk"))
    locked_cat, locked_risk = lock_pair(cat, risk)
    if locked_cat != cat or locked_risk != risk:
        return True
    text_parts = [str(label.get("summary") or "")]
    for event in label.get("events") or []:
        if isinstance(event, dict):
            text_parts.append(str(event.get("event") or ""))
    text = " ".join(text_parts)
    if cat != "accident" and has_unhedged_accident(text):
        return True
    return False


def apply_raise(label: dict[str, Any], new_category: str) -> dict[str, Any]:
    """Yalnız yükselt; düşürme."""
    old = str(label.get("category") or "normal").strip().lower()
    new = str(new_category or "").strip().lower()
    if new not in _CAT_RANK:
        return label
    if _CAT_RANK[new] <= _CAT_RANK.get(old, 0):
        return label
    label["category"] = new
    notes = str(label.get("notes") or "")
    tag = f"eleştirmen {old}→{new}"
    label["notes"] = f"{notes} | {tag}".strip(" |") if notes else tag
    return label


def _critic_user_prompt(label: dict[str, Any]) -> str:
    events = label.get("events") or []
    event_lines = "; ".join(
        f"{item.get('time', '')} {item.get('event', '')}".strip()
        for item in events
        if isinstance(item, dict) and item.get("event")
    )
    return (
        f"category={label.get('category')}\n"
        f"risk={label.get('risk')}\n"
        f"summary={label.get('summary')}\n"
        f"events={event_lines or '-'}\n"
        "Metin tamamlanmış kaza mı, ramak kala mı, rutin mi?"
    )


async def critique_label(label: dict[str, Any]) -> dict[str, Any]:
    """LLM ile olası yükseltme. Çağrı düşerse etiketi olduğu gibi bırakır."""
    from utils.model_client import ModelCallError, chat_llm

    try:
        result = await chat_llm(
            _critic_user_prompt(label),
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=80,
        )
        parsed = parse_json(result.text)
    except (ModelCallError, ValueError, json.JSONDecodeError):
        return label
    if parsed.get("raise") is False:
        return label
    return apply_raise(label, str(parsed.get("category") or ""))
