"""Action Recommender — risk seviyesine göre acil müdahale adımları üretir."""

from __future__ import annotations

from typing import Any

import requests

from agents.state import AgentState
from utils.config import API_BASE_URL, MODEL_NAME


def action_recommender_tool(state: AgentState) -> dict[str, Any]:
    """Risk ve olay özetine göre önerilen aksiyon listesini döner."""
    print("\n--- [3] Action Recommender (LLM) Çalışıyor ---")

    analysis = state.get("analysis_result", {})
    event_text = analysis.get("event", "")
    risk = state.get("risk_level", "Normal")

    endpoint = f"{API_BASE_URL.rstrip('/')}/v1/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Sen bir endüstriyel kriz yönetim koordinatörüsün. "
                    "Yanıtların her zaman sahadaki ekiplere yönelik, net, kısa ve eyleme dönük "
                    "emir kipleri (Örn: 'Alanı tahliye et') içermelidir."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Olay: {event_text}\n"
                    f"Risk Seviyesi: {risk}\n"
                    "Sahada anında uygulanması gereken en kritik 3 acil müdahale adımını "
                    "kısa maddeler halinde yaz."
                ),
            },
        ],
        "max_tokens": 150,
        "temperature": 0.2,
    }

    headers = {"Content-Type": "application/json"}
    actions: list[str] = []

    if risk.lower() in ["güvenli", "normal"]:
        actions = [
            "Sistemi standart şekilde izlemeye devam et",
            "Periyodik kontrolleri sürdür",
        ]
    else:
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            action_text = response.json()["choices"][0]["message"]["content"].strip()
            actions = [
                line.strip("- *1234567890. ")
                for line in action_text.split("\n")
                if line.strip()
            ]
        except Exception as e:
            print(f"API Hatası (Action Recommender): {e}")
            actions = ["Sistem operatörünü manuel inceleme için uyar!"]

    return {"recommended_actions": actions}
