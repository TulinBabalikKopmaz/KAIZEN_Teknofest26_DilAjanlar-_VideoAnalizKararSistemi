"""Action Recommender — RAG destekli acil müdahale önerileri üretir."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import aiohttp
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from agents.state import AgentState
from utils.config import API_BASE_URL, MODEL_NAME

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR: Path = PROJECT_ROOT / "dataset" / "vector_db"
EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME: str = "isg_mevzuat"
RETRIEVAL_K: int = 2

OUTPUT_RULES: str = (
    "Çıktılarında ASLA HTML tag'leri (<br>, <p>, <ul>, <li> vb.) veya markdown formatı "
    "(**, *, #, -, ```) kullanma. Sadece düz metin yaz. "
    "Aşırı resmi, uzatılmış yasal jargondan kaçın. Doğrudan, sade ve net bir dil kullan. "
    "Çıktın maksimum 2 veya 3 kısa, vurucu ve eyleme geçirilebilir cümleden oluşsun. "
    "Yanıtların profesyonel, net ve Türkçe dil kurallarına uygun olsun."
)

BASE_SYSTEM_PROMPT: str = (
    "Sen kıdemli bir İş Sağlığı ve Güvenliği (İSG) uzmanısın. "
    "Sahadaki ekiplere yönelik kısa, net ve eyleme dönük acil müdahale adımları üret. "
    f"{OUTPUT_RULES}"
)


def build_retriever() -> VectorStoreRetriever:
    """ChromaDB (isg_mevzuat) üzerinde similarity retriever oluşturur."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


def retrieve_isg_context(query: str) -> str:
    """
    Tehlike özeti / risk ile ChromaDB'de similarity_search (k=2) yapar.

    Veritabanına ulaşılamazsa boş string döner (standart prompta düşülür).
    """
    try:
        retriever = build_retriever()
        vectorstore = retriever.vectorstore
        docs = vectorstore.similarity_search(query, k=RETRIEVAL_K)
        if not docs:
            print("RAG: İlgili kanun maddesi bulunamadı, standart prompt kullanılacak.")
            return ""

        retrieved_texts = "\n\n".join(doc.page_content.strip() for doc in docs if doc.page_content)
        print(f"RAG: {len(docs)} kanun maddesi getirildi.")
        return retrieved_texts
    except Exception as exc:
        print(f"RAG Hatası (veritabanı atlanıyor): {exc}")
        return ""


def build_system_prompt(retrieved_texts: str) -> str:
    """RAG sonuçlarını system prompt'a ekler; yoksa temel promptu döner."""
    if not retrieved_texts:
        return BASE_SYSTEM_PROMPT

    return (
        f"{BASE_SYSTEM_PROMPT} "
        f"Referans İSG kanun maddeleri (yalnızca bilgi kaynağı): {retrieved_texts} "
        "Verilen kanun maddelerini ASLA birebir kopyalama. Onları sentezle ve sadece referans ver "
        "(Örn: 'Madde 25 uyarınca...'). "
        "Kanun metnini alıntılama, cümle bölme veya yapıştırma; madde numarasını anıp sahaya "
        "uygulanacak net emir ver. "
        "Örnek Format: 'Madde 25 uyarınca hayati tehlike tespit edilmiştir. "
        "İskeledeki çalışmayı derhal durdurun ve alanı tahliye edin.'"
    )


def _parse_actions(action_text: str) -> list[str]:
    """LLM çıktısını madde listesine çevirir; olası HTML kalıntılarını temizler."""
    cleaned = re.sub(r"<[^>]+>", " ", action_text)
    cleaned = cleaned.replace("&nbsp;", " ")
    return [
        line.strip("- *1234567890. ")
        for line in cleaned.split("\n")
        if line.strip()
    ]


async def action_recommender_tool(state: AgentState) -> dict[str, Any]:
    """Risk ve olay özetine göre RAG destekli aksiyon listesi üretir."""
    print("\n--- [3] Action Recommender (LLM + RAG) Çalışıyor ---")

    analysis = state.get("analysis_result", {})
    event_text = analysis.get("event", "")
    risk = state.get("risk_level", "Normal")

    actions: list[str] = []

    if risk.lower() in ["güvenli", "normal"]:
        return {
            "recommended_actions": [
                "Sistemi standart şekilde izlemeye devam et",
                "Periyodik kontrolleri sürdür",
            ]
        }

    rag_query = f"{event_text} Risk seviyesi: {risk}".strip()
    retrieved_texts = retrieve_isg_context(rag_query)
    system_prompt = build_system_prompt(retrieved_texts)

    endpoint = f"{API_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Olay: {event_text}\n"
                    f"Risk Seviyesi: {risk}\n"
                    "Bu olay için sahaya yönelik acil müdahale yaz.\n"
                    "Kurallar:\n"
                    "1) Kanun maddelerini ASLA birebir kopyalama; sentezle ve sadece referans ver "
                    "(Örn: Madde 25 uyarınca...).\n"
                    "2) Aşırı resmi yasal jargondan kaçın; doğrudan, sade ve net dil kullan.\n"
                    "3) Çıktı en fazla 2 veya 3 kısa, vurucu, eyleme geçirilebilir cümle olsun.\n"
                    "4) HTML veya markdown kullanma; yalnızca düz Türkçe metin yaz.\n"
                    "5) Örnek Format: Madde 25 uyarınca hayati tehlike tespit edilmiştir. "
                    "İskeledeki çalışmayı derhal durdurun ve alanı tahliye edin."
                ),
            },
        ],
        "max_tokens": 180,
        "temperature": 0.1,
    }
    headers = {"Content-Type": "application/json"}

    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                action_text = data["choices"][0]["message"]["content"].strip()
                actions = _parse_actions(action_text)
    except Exception as e:
        print(f"API Hatası (Action Recommender): {e}")
        actions = ["Sistem operatörünü manuel inceleme için uyar!"]

    return {"recommended_actions": actions}
