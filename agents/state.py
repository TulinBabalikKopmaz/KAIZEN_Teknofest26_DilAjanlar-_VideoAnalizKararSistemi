"""LangGraph ajan durum (state) tanımı."""

from __future__ import annotations

from typing import List, TypedDict


class AgentState(TypedDict):
    keyframes: List[str]
    analysis_result: dict
    risk_level: str
    recommended_actions: List[str]
    rag_context: str
    trigger_reason: str
    # Demoda jürinin verdiği serbest metin soru; boşsa standart İSG taraması yapılır
    user_prompt: str
