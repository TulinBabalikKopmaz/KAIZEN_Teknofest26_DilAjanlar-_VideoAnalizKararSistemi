"""
RAG entegreli Action Recommender smoke test.

Video Analyzer adımını atlar; mock AgentState ile doğrudan
action_recommender_tool'u çalıştırır.

Kullanım (proje kökünden):
    python tools/test_rag.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.action_recommender import (  # noqa: E402
    RETRIEVAL_K,
    action_recommender_tool,
    build_retriever,
)
from agents.state import AgentState  # noqa: E402

MOCK_VIDEO_ANALYSIS: str = (
    "Bir işçi baret ve emniyet kemeri takmadan "
    "5 metre yükseklikteki iskelede çalışıyor."
)
MOCK_RISK_LEVEL: str = "Kritik Risk"


def build_mock_state() -> AgentState:
    """
    Kullanıcı senaryosundaki mock girdiyi AgentState'e map eder.

    Not: Pipeline'da olay özeti `analysis_result.event` altında tutulur;
    test girdilerindeki `video_analysis` alanı buraya taşınır.
    """
    return {
        "keyframes": [],
        "analysis_result": {
            "timestamp": "mock-test",
            "event": MOCK_VIDEO_ANALYSIS,
        },
        "risk_level": MOCK_RISK_LEVEL,
        "recommended_actions": [],
        "rag_context": "",
        "trigger_reason": "",
        "user_prompt": "",
    }


def fetch_retrieved_articles(event_text: str, risk_level: str) -> list[dict[str, Any]]:
    """ChromaDB similarity_search sonuçlarını rapor için toplar."""
    query = f"{event_text} Risk seviyesi: {risk_level}".strip()
    try:
        retriever = build_retriever()
        docs = retriever.vectorstore.similarity_search(query, k=RETRIEVAL_K)
    except Exception as exc:
        return [{"error": str(exc)}]

    articles: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        articles.append(
            {
                "sira": idx,
                "kaynak": doc.metadata.get("filename") or doc.metadata.get("source", "bilinmiyor"),
                "metin": doc.page_content.strip(),
            }
        )
    return articles


def print_report(
    mock_input: dict[str, str],
    retrieved_articles: list[dict[str, Any]],
    tool_result: dict[str, Any],
) -> None:
    """Test çıktısını okunaklı JSON olarak terminale basar."""
    report = {
        "test": "RAG + Action Recommender",
        "mock_girdi": mock_input,
        "rag_kanun_maddeleri": retrieved_articles,
        "onerilen_aksiyonlar": tool_result.get("recommended_actions", []),
        "ham_tool_ciktisi": tool_result,
    }

    print("\n" + "=" * 60)
    print("  RAG ACTION RECOMMENDER TEST SONUCU")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 60)

    print("\n--- Özet ---")
    print(f"Senaryo : {mock_input['video_analysis']}")
    print(f"Risk    : {mock_input['risk_level']}")
    print(f"RAG hit : {len(retrieved_articles)} madde")
    for article in retrieved_articles:
        if "error" in article:
            print(f"  ! Hata: {article['error']}")
            continue
        preview = article["metin"][:160].replace("\n", " ")
        print(f"  [{article['sira']}] {article['kaynak']}: {preview}...")
    print("Aksiyonlar:")
    for i, action in enumerate(tool_result.get("recommended_actions", []), start=1):
        print(f"  {i}. {action}")


async def run_test() -> dict[str, Any]:
    """Mock state ile RAG retrieval + action_recommender_tool'u çalıştırır."""
    mock_input = {
        "video_analysis": MOCK_VIDEO_ANALYSIS,
        "risk_level": MOCK_RISK_LEVEL,
    }
    state = build_mock_state()

    print("Mock AgentState hazırlandı (Video Analyzer atlandı).")
    print(json.dumps(mock_input, indent=2, ensure_ascii=False))

    print("\n[1/2] ChromaDB'den ilgili kanun maddeleri getiriliyor...")
    retrieved_articles = fetch_retrieved_articles(MOCK_VIDEO_ANALYSIS, MOCK_RISK_LEVEL)

    print("[2/2] action_recommender_tool çağrılıyor...")
    tool_result = await action_recommender_tool(state)

    print_report(mock_input, retrieved_articles, tool_result)
    return tool_result


def main() -> None:
    asyncio.run(run_test())


if __name__ == "__main__":
    main()
