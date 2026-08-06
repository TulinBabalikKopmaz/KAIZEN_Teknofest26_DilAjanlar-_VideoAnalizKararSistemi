"""LangGraph ajan durum (state) tanımı."""

from __future__ import annotations

from typing import List, TypedDict


class AgentState(TypedDict):
    keyframes: List[str]
    analysis_result: dict
    risk_level: str
    recommended_actions: List[str]
