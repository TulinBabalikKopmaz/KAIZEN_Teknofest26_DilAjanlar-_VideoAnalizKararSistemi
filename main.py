"""İSG multi-agent pipeline — LangGraph workflow giriş noktası."""

from __future__ import annotations

import glob
import json
from typing import Any

from langgraph.graph import END, StateGraph

from agents.action_recommender import action_recommender_tool
from agents.risk_assessor import risk_assessor_tool
from agents.state import AgentState
from agents.video_analyzer import video_analyzer_tool


def build_workflow() -> Any:
    """Video Analyzer → Risk Assessor → Action Recommender grafını derler."""
    workflow = StateGraph(AgentState)

    workflow.add_node("video_analyzer", video_analyzer_tool)
    workflow.add_node("risk_assessor", risk_assessor_tool)
    workflow.add_node("action_recommender", action_recommender_tool)

    workflow.set_entry_point("video_analyzer")
    workflow.add_edge("video_analyzer", "risk_assessor")
    workflow.add_edge("risk_assessor", "action_recommender")
    workflow.add_edge("action_recommender", END)

    return workflow.compile()


def run_pipeline(keyframes: list[str] | None = None) -> dict[str, Any]:
    """Pipeline'ı çalıştırır ve final JSON çıktısını döner."""
    if keyframes is None:
        keyframes = sorted(glob.glob("keyframes/*.jpg"))

    initial_state: AgentState = {
        "keyframes": keyframes,
        "analysis_result": {},
        "risk_level": "",
        "recommended_actions": [],
    }

    app = build_workflow()
    final_state = app.invoke(initial_state)

    return {
        "zaman_damgasi": final_state["analysis_result"].get("timestamp"),
        "olay_ozeti": final_state["analysis_result"].get("event"),
        "risk_seviyesi": final_state["risk_level"],
        "onerilen_aksiyonlar": final_state["recommended_actions"],
    }


if __name__ == "__main__":
    print("Gerçek VLM Entegrasyonlu Ajan Başlatılıyor...")
    output = run_pipeline()
    print("\n=== FINAL AJAN ÇIKTI (JSON) ===")
    print(json.dumps(output, indent=4, ensure_ascii=False))
