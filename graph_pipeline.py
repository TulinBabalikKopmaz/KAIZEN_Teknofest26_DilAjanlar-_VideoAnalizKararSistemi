"""İSG LangGraph multi-agent pipeline (Video Analyzer → Risk → Action)."""

from __future__ import annotations

import asyncio
import glob
import json
from typing import Any

from langgraph.graph import END, StateGraph

from agents.action_recommender import action_recommender_tool
from agents.risk_assessor import risk_assessor_tool
from agents.state import AgentState
from agents.video_analyzer import second_look_tool, video_analyzer_tool
from utils.spec_output import pipeline_result_to_spec


def _after_analyzer(state: AgentState) -> str:
    """Sakin tarama + tetik varsa ikinci bakış; aksi halde risk ajanına."""
    if state.get("second_look_done"):
        return "risk_assessor"
    event = str((state.get("analysis_result") or {}).get("event") or "").lower()
    trigger = (state.get("trigger_reason") or "").strip()
    sakin = any(token in event for token in ("güvenli", "guvenli", "rutin", "olağan", "olagan"))
    if trigger and sakin:
        return "second_look"
    return "risk_assessor"


def build_workflow() -> Any:
    """Video Analyzer → (gerekirse ikinci bakış) → Risk Assessor → Action Recommender."""
    workflow = StateGraph(AgentState)

    workflow.add_node("video_analyzer", video_analyzer_tool)
    workflow.add_node("second_look", second_look_tool)
    workflow.add_node("risk_assessor", risk_assessor_tool)
    workflow.add_node("action_recommender", action_recommender_tool)

    workflow.set_entry_point("video_analyzer")
    workflow.add_conditional_edges(
        "video_analyzer",
        _after_analyzer,
        {"second_look": "second_look", "risk_assessor": "risk_assessor"},
    )
    workflow.add_edge("second_look", "risk_assessor")
    workflow.add_edge("risk_assessor", "action_recommender")
    workflow.add_edge("action_recommender", END)

    return workflow.compile()


async def run_pipeline(
    keyframes: list[str] | None = None,
    trigger_reason: str = "",
    user_prompt: str = "",
) -> dict[str, Any]:
    """Pipeline'ı çalıştırır ve final JSON çıktısını döner.

    user_prompt: demoda jürinin verdiği serbest metin soru (boş olabilir).
    """
    if keyframes is None:
        keyframes = sorted(glob.glob("keyframes/*.jpg"))

    initial_state: AgentState = {
        "keyframes": keyframes,
        "analysis_result": {},
        "risk_level": "",
        "recommended_actions": [],
        "rag_context": "",
        "trigger_reason": trigger_reason,
        "user_prompt": user_prompt,
    }

    app = build_workflow()
    final_state = await app.ainvoke(initial_state)

    legacy = {
        "zaman_damgasi": final_state["analysis_result"].get("timestamp"),
        "olay_ozeti": final_state["analysis_result"].get("event"),
        "risk_seviyesi": final_state["risk_level"],
        "onerilen_aksiyonlar": final_state["recommended_actions"],
        "isg_kanun_maddeleri": final_state.get("rag_context", ""),
        "tetik_sebebi": final_state.get("trigger_reason", trigger_reason),
        "kullanici_promptu": final_state.get("user_prompt", user_prompt),
    }
    spec = pipeline_result_to_spec(legacy)
    return {**legacy, "spec": spec}


if __name__ == "__main__":
    print("LangGraph ajan pipeline başlatılıyor...")
    output = asyncio.run(run_pipeline())
    print("\n=== FINAL AJAN ÇIKTI (JSON) ===")
    print(json.dumps(output, indent=4, ensure_ascii=False))
