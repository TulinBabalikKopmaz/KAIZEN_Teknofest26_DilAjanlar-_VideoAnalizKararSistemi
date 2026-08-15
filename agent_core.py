"""
Geriye dönük uyumluluk katmanı.

Yeni giriş noktası: main.py
Modüler yapı: agents/, utils/
"""

from agents.action_recommender import action_recommender_tool
from agents.risk_assessor import risk_assessor_tool
from agents.state import AgentState
from agents.video_analyzer import video_analyzer_tool
from utils.image import encode_image

__all__ = [
    "AgentState",
    "encode_image",
    "video_analyzer_tool",
    "risk_assessor_tool",
    "action_recommender_tool",
]

if __name__ == "__main__":
    import asyncio
    import json

    from graph_pipeline import run_pipeline

    print("Gerçek VLM Entegrasyonlu Ajan Başlatılıyor...")
    print("(Not: E2E giriş noktası main.py; LangGraph için graph_pipeline.py)")
    output = asyncio.run(run_pipeline())
    print("\n=== FINAL AJAN ÇIKTI (JSON) ===")
    print(json.dumps(output, indent=4, ensure_ascii=False))
