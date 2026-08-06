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
    from main import run_pipeline
    import json

    print("Gerçek VLM Entegrasyonlu Ajan Başlatılıyor...")
    print("(Not: Tercih edilen giriş noktası artık main.py)")
    output = run_pipeline()
    print("\n=== FINAL AJAN ÇIKTI (JSON) ===")
    print(json.dumps(output, indent=4, ensure_ascii=False))
