"""Risk Assessor — olay özetini İSG seviyelerine map eder (metin LLM'i)."""

from __future__ import annotations

from typing import Any

from agents.state import AgentState
from utils.model_client import ModelCallError, chat_llm
from utils.risk_rules import text_risk_floor
from utils.spec_output import normalize_risk

SYSTEM_PROMPT: str = (
    "Sen kıdemli bir İş Sağlığı ve Güvenliği (İSG) uzmanısın. "
    "Gelen olay özetini analiz et. Sadece şu 4 seviyeden birini tek kelime olarak söyle: "
    "'Güvenli', 'Düşük', 'Orta', 'Kritik'."
)

_FLOOR_TO_LEVEL = {"Düşük": "Düşük", "Orta": "Orta", "Yüksek": "Kritik"}
_RANK = {"Düşük": 0, "Orta": 1, "Yüksek": 2}


async def risk_assessor_tool(state: AgentState) -> dict[str, Any]:
    """Analiz sonucuna göre Güvenli / Düşük / Orta / Kritik risk atar.

    LLM kararını alır; metinde çarpışma/düşme/yanma varsa Kritik altına düşürmez.
    """
    print("\n--- [2] Risk Assessor (LLM) Çalışıyor ---")

    analysis = state.get("analysis_result", {})
    event_text = analysis.get("event", "Bilinmeyen olay")

    try:
        result = await chat_llm(
            f"Olay Özeti: {event_text}\nRisk seviyesi nedir?",
            system=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=10,
        )
        risk = result.text.strip().replace(".", "").replace(",", "")
    except ModelCallError as exc:
        print(f"Risk Assessor LLM hatası: {exc}")
        risk = "Bilinmiyor"

    # Kural katmanı: LLM metni küçümsese bile anahtar kelime tabanı uygula
    floor, _ = text_risk_floor({"summary": event_text, "events": [], "actions": []})
    if _RANK[floor] > _RANK.get(normalize_risk(risk), 0):
        mapped = _FLOOR_TO_LEVEL[floor]
        print(f"Kural tabanı: LLM={risk} → en az {mapped} (metin kanıtı)")
        risk = mapped

    print(f"Uzman Ajanın Kararı: {risk}")
    return {"risk_level": risk}
