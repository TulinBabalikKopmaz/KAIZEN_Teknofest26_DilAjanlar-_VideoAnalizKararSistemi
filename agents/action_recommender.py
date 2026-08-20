"""Action Recommender — RAG destekli acil müdahale önerileri üretir."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from agents.state import AgentState
from utils.model_client import ModelCallError, chat_llm

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


def build_retriever() -> Any:
    """ChromaDB (isg_mevzuat) üzerinde similarity retriever oluşturur.

    langchain paketleri ağır; demo makinesinde kurulu olmayabilir. Bu yüzden
    import fonksiyon içinde: RAG yoksa sistem standart prompt ile devam eder.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

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
            ],
            "rag_context": "",
        }

    rag_query = f"{event_text} Risk seviyesi: {risk}".strip()
    # Chroma + embedding senkron çalışıyor; event loop'u bloklamasın
    retrieved_texts = await asyncio.to_thread(retrieve_isg_context, rag_query)
    system_prompt = build_system_prompt(retrieved_texts)

    question = (state.get("user_prompt") or "").strip()
    question_block = f"Operatörün sorusu: {question}\n" if question else ""

    try:
        result = await chat_llm(
            f"{question_block}"
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
            "İskeledeki çalışmayı derhal durdurun ve alanı tahliye edin.",
            system=system_prompt,
            temperature=0.1,
            max_tokens=180,
        )
        actions = _parse_actions(result.text.strip())
    except ModelCallError as exc:
        print(f"Action Recommender LLM hatası: {exc}")
        actions = ["Sistem operatörünü manuel inceleme için uyar!"]

    return {
        "recommended_actions": actions,
        "rag_context": retrieved_texts,
    }
