"""EVREN bge-m3-embed ile kısa İSG mevzuat getirimi.

Karar sınıfını ve model aksiyon listesini değiştirmez; ekranda küçük teyit notu
için pasaj döner. rerank / sparse / ColBERT yok (ölçümde R@1 düşüyor).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from utils.model_client import ModelCallError, embed_texts

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "rag" / "evren_bge_m3_index.json"

# Kısa resmi özetler; tam kanun metni değil. Getirme sorgusu için yeter.
_CHUNKS: list[dict[str, str]] = [
    {
        "id": "6331-4",
        "source": "6331 sayılı İSG Kanunu md. 4",
        "text": (
            "Madde 4. İşveren, çalışanların işle ilgili sağlık ve güvenliğini sağlamakla "
            "yükümlüdür. Riskleri belirler, önler, eğitim ve araç gereç sağlar."
        ),
    },
    {
        "id": "6331-13",
        "source": "6331 sayılı İSG Kanunu md. 13",
        "text": (
            "Madde 13. Ciddi ve yakın tehlike halinde çalışanlar çalışmayı durdurabilir. "
            "Hayati tehlike giderilmeden üretim zorlanamaz; tahliye ve müdahale önceliklidir."
        ),
    },
    {
        "id": "6331-30-acil",
        "source": "6331 sayılı İSG Kanunu md. 30",
        "text": (
            "Madde 30. İşveren acil durumları önceden değerlendirir. Yangın, patlama, "
            "çökme ve yaralanmada tahliye, ilkyardım ve ilgili ekiplere haber verilir."
        ),
    },
    {
        "id": "yukseklik",
        "source": "Yapı İşlerinde İSG Yönetmeliği — düşme",
        "text": (
            "Madde 5 ve ekler. Yüksekten düşme riskinde çalışma durdurulur, alan kapatılır, "
            "yaralıya ilkyardım ve sağlık ekibi çağrılır. Korkuluk ve kişisel koruyucu şarttır."
        ),
    },
    {
        "id": "forklift",
        "source": "İş Ekipmanları Kullanımında Sağlık ve Güvenlik Yönetmeliği",
        "text": (
            "Madde 6. Forklift ve benzeri araçlarda yaya mesafesi korunur, hız düşürülür, "
            "yaya yolu ayrılır. Ramak kalada trafik durdurulur, alan işaretlenir."
        ),
    },
]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    na = math.sqrt(sum(a * a for a in left))
    nb = math.sqrt(sum(b * b for b in right))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _load_cache() -> dict[str, Any] | None:
    if not CACHE_PATH.is_file():
        return None
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    items = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(items, list) or len(items) != len(_CHUNKS):
        return None
    return raw


async def _ensure_index() -> list[dict[str, Any]]:
    cached = _load_cache()
    if cached:
        return list(cached["items"])
    vectors = await embed_texts([chunk["text"] for chunk in _CHUNKS])
    items = []
    for chunk, vector in zip(_CHUNKS, vectors):
        items.append({**chunk, "vector": vector})
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps({"model": "bge-m3-embed", "items": items}, ensure_ascii=False),
        encoding="utf-8",
    )
    return items


def retrieve_mevzuat_lexical(query: str, *, k: int = 2) -> str:
    """Ağ yokken anahtar kelime ile aynı kısa pasajlar. Aksiyon listesini değiştirmez."""
    blob = (query or "").casefold()
    if not blob.strip():
        return ""
    keys = {
        "6331-13": ("düş", "yaralan", "hareketsiz", "iskele", "yükseklik", "kaza"),
        "6331-30-acil": ("alev", "yangın", "patlama", "duman", "itfaiye", "ateş"),
        "forklift": ("forklift", "yaya", "araç", "ramak"),
        "yukseklik": ("düş", "iskele", "yükseklik"),
        "6331-4": ("rutin", "proses", "izle", "operasyon"),
    }
    scored: list[tuple[int, dict[str, str]]] = []
    for chunk in _CHUNKS:
        hits = sum(1 for word in keys.get(chunk["id"], ()) if word in blob)
        if hits:
            scored.append((hits, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    picked = [chunk for _hits, chunk in scored[:k]]
    if not picked:
        picked = _CHUNKS[:1]
    parts = []
    for row in picked:
        parts.append(f"{row['source']}: {row['text']}")
    return "\n".join(parts)


async def retrieve_mevzuat(query: str, *, k: int = 2) -> str:
    """En yakın 1–2 pasaj. Boş veya hata: '' (aksiyon listesi aynı kalır)."""
    text = (query or "").strip()
    if not text:
        return ""
    try:
        items = await _ensure_index()
        qvec = (await embed_texts([text]))[0]
    except (ModelCallError, IndexError, OSError) as exc:
        print(f"  [rag] EVREN embed atlandı: {exc}")
        return retrieve_mevzuat_lexical(text, k=k)
    ranked = sorted(items, key=lambda row: _cosine(qvec, list(row.get("vector") or [])), reverse=True)
    picked = [row for row in ranked[: max(1, k)] if _cosine(qvec, list(row.get("vector") or [])) > 0.15]
    if not picked:
        return retrieve_mevzuat_lexical(text, k=k)
    parts = []
    for row in picked:
        source = str(row.get("source") or "")
        body = str(row.get("text") or "").strip()
        parts.append(f"{source}: {body}" if source else body)
    return "\n".join(parts)
