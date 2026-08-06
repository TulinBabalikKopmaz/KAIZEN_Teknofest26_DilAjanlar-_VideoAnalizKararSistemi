"""
İSG mevzuatı için RAG (Retrieval-Augmented Generation) vektör veritabanı kurucu.

dataset/isg_mevzuat/ altındaki PDF/TXT dosyalarını okur, chunk'lar,
multilingual embedding üretir ve ChromaDB'ye kalıcı olarak yazar.

Kullanım (proje kökünden):
    python tools/rag_builder.py

Gerekli paketler:
    pip install langchain-community langchain-text-splitters \\
                langchain-huggingface langchain-chroma chromadb \\
                sentence-transformers pypdf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MEVZUAT_DIR: Path = PROJECT_ROOT / "dataset" / "isg_mevzuat"
VECTOR_DB_DIR: Path = PROJECT_ROOT / "dataset" / "vector_db"

EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME: str = "isg_mevzuat"
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")


class NoDocumentsFoundError(FileNotFoundError):
    """isg_mevzuat klasöründe işlenebilir doküman yoksa fırlatılır."""


def ensure_directories() -> None:
    """dataset/isg_mevzuat ve dataset/vector_db klasörlerini oluşturur."""
    MEVZUAT_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def find_document_paths(directory: Path = MEVZUAT_DIR) -> list[Path]:
    """Klasördeki .pdf / .txt dosyalarını sıralı liste olarak döner."""
    paths: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(directory.glob(f"*{ext}"))
        paths.extend(directory.glob(f"*{ext.upper()}"))

    unique = {path.resolve(): path for path in paths}
    return sorted(unique.values(), key=lambda p: p.name.lower())


def load_single_document(path: Path) -> list[Document]:
    """Tek bir PDF veya TXT dosyasını LangChain Document listesine çevirir."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {path.name}")

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("filename", path.name)
    return docs


def load_documents(directory: Path = MEVZUAT_DIR) -> list[Document]:
    """
    isg_mevzuat altındaki tüm PDF/TXT dosyalarını yükler.

    Raises:
        NoDocumentsFoundError: Klasör boşsa veya hiç dosya okunamazsa.
    """
    paths = find_document_paths(directory)
    if not paths:
        raise NoDocumentsFoundError(
            f"Doküman bulunamadı: {directory}\n"
            "Lütfen .pdf veya .txt formatında İSG mevzuat dosyalarını bu klasöre koyun."
        )

    documents: list[Document] = []
    failed: list[str] = []

    for path in paths:
        try:
            loaded = load_single_document(path)
            if not loaded:
                failed.append(f"{path.name} (boş içerik)")
                continue
            documents.extend(loaded)
            print(f"  ✓ {path.name} → {len(loaded)} sayfa/parça")
        except Exception as exc:
            failed.append(f"{path.name} ({exc})")
            print(f"  ✗ {path.name}: {exc}")

    if not documents:
        detail = "; ".join(failed) if failed else "bilinmeyen neden"
        raise NoDocumentsFoundError(
            f"Hiçbir doküman başarıyla okunamadı ({directory}).\nDetay: {detail}"
        )

    if failed:
        print(f"Uyarı: {len(failed)} dosya atlandı.")

    return documents


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Metinleri RecursiveCharacterTextSplitter ile chunk'lara böler."""
    if not documents:
        raise ValueError("Bölünecek doküman listesi boş.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))
    if not chunks:
        raise ValueError("Chunk üretilemedi; doküman içerikleri boş olabilir.")
    return chunks


def build_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Türkçe destekli HuggingFace embedding modelini yükler."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def persist_vector_store(
    chunks: Sequence[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: Path = VECTOR_DB_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Chunk embedding'lerini ChromaDB'ye kalıcı olarak yazar."""
    persist_directory.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=list(chunks),
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )
    return vectorstore


def build_rag_pipeline(
    mevzuat_dir: Path = MEVZUAT_DIR,
    vector_db_dir: Path = VECTOR_DB_DIR,
) -> Chroma:
    """
    Uçtan uca RAG index pipeline'ı:
    klasör hazırla → yükle → chunk'la → embed → Chroma'ya kaydet.
    """
    ensure_directories()

    print(f"Kaynak klasör : {mevzuat_dir}")
    print(f"Vektör DB     : {vector_db_dir}")
    print("-" * 50)

    print("[1/4] Dokümanlar yükleniyor...")
    documents = load_documents(mevzuat_dir)
    print(f"      Toplam yüklenen birim: {len(documents)}")

    print("[2/4] Metinler chunk'lanıyor...")
    chunks = split_documents(documents)
    print(f"      Chunk sayısı: {len(chunks)} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"[3/4] Embedding modeli yükleniyor ({EMBEDDING_MODEL})...")
    embeddings = build_embeddings()

    print("[4/4] ChromaDB'ye yazılıyor...")
    vectorstore = persist_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        persist_directory=vector_db_dir,
    )
    print(f"      Koleksiyon: {COLLECTION_NAME}")
    print(f"      Kayıt yolu: {vector_db_dir}")

    return vectorstore


def main() -> None:
    print("İSG RAG Builder — Mevzuat → ChromaDB")
    print("-" * 50)

    try:
        vectorstore = build_rag_pipeline()
    except NoDocumentsFoundError as exc:
        print(f"\n[HATA] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[HATA] RAG pipeline başarısız: {exc}")
        sys.exit(1)

    try:
        count = vectorstore._collection.count()  # noqa: SLF001
        print("-" * 50)
        print(f"Tamamlandı. Vektör sayısı: {count}")
    except Exception:
        print("-" * 50)
        print("Tamamlandı. Vektör veritabanı kaydedildi.")


if __name__ == "__main__":
    main()
