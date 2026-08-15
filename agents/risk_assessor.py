"""Risk Assessor — olay özetini İSG seviyelerine map eder."""

from __future__ import annotations

from typing import Any

import requests

from agents.state import AgentState
from utils.config import API_BASE_URL, MODEL_NAME


def risk_assessor_tool(state: AgentState) -> dict[str, Any]:
    """Analiz sonucuna göre Güvenli / Düşük / Orta / Kritik risk atar."""
    print("\n--- [2] Risk Assessor (LLM) Çalışıyor ---")

    analysis = state.get("analysis_result", {})
    event_text = analysis.get("event", "Bilinmeyen olay")

    endpoint = f"{API_BASE_URL.rstrip('/')}/v1/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Sen kıdemli bir İş Sağlığı ve Güvenliği (İSG) uzmanısın. "
                    "Gelen olay özetini analiz et. Sadece şu 4 seviyeden birini tek kelime olarak söyle: "
                    "'Güvenli', 'Düşük', 'Orta', 'Kritik'."
                ),
            },
            {
                "role": "user",
                "content": f"Olay Özeti: {event_text}\nRisk seviyesi nedir?",
            },
        ],
        "max_tokens": 10,
        "temperature": 0.1,
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        risk = response.json()["choices"][0]["message"]["content"].strip()
        risk = risk.replace(".", "").replace(",", "")
    except Exception as e:
        print(f"API Hatası (Risk Assessor): {e}")
        risk = "Bilinmiyor"

    print(f"Uzman Ajanın Kararı: {risk}")
    return {"risk_level": risk}
