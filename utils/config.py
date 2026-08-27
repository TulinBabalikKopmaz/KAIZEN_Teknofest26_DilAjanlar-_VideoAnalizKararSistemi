"""Ortak API ve model ayarları; tüm değerler ortam değişkeninden okunur.

PROVIDER seçenekleri:
    teknofest : yarışmanın ortak H200 API'si (VLM ve LLM ayrı modeller)
    ollama    : yerel geliştirme ve demo yedeği
    mock       : model olmadan pipeline testi (CI / hızlı deneme)

Değerler import anında sabitlenmez; fonksiyonlar her çağrıda env'i okur, böylece
demo sırasında .env veya os.environ değişince yeniden başlatmak gerekmez.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _ROOT / ".env"


def _bindings_from_env_file(path: Path) -> dict[str, str]:
    """python-dotenv kaçırsa bile KEY=value satırlarını oku (utf-8, BOM'suz)."""
    if not path.is_file():
        return {}
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if key.lower().startswith("export "):
            key = key[7:].strip()
        value = value.strip().strip("'").strip('"').strip()
        if key:
            found[key] = value
    return found


_FILE_VARS = _bindings_from_env_file(_ENV_PATH)
load_dotenv(_ENV_PATH, encoding="utf-8", override=True)

PROVIDERS: tuple[str, ...] = ("teknofest", "ollama", "mock")
ROLES: tuple[str, ...] = ("vlm", "llm")

# EVREN ortak API alias'ları (https://evren-teknofest.ssyz.org.tr/)
DEFAULT_TEKNOFEST_BASE: str = "https://evren-llmapi.ssyz.org.tr/v1"
DEFAULT_VLM_MODEL: str = "vlm"
DEFAULT_LLM_MODEL: str = "llm-fast"
DEFAULT_OLLAMA_VLM: str = "qwen2.5vl:7b"
DEFAULT_OLLAMA_LLM: str = "qwen2.5:7b"


def _clean_env_value(raw: str | None) -> str:
    text = (raw or "").strip().strip("'").strip('"').strip()
    if text.lower().startswith("bearer "):
        text = text[7:].strip()
    return text


def _env(name: str, default: str = "") -> str:
    """Dosya + ortam. API anahtarında sk-evren- varsa sahte/kısa kabuk değeri ezilmesin."""
    file_vars = _bindings_from_env_file(_ENV_PATH)
    from_os = _clean_env_value(os.getenv(name))
    from_file = _clean_env_value(file_vars.get(name, ""))
    if "API_KEY" in name or name.endswith("_KEY"):
        if from_file.startswith("sk-evren-") and not from_os.startswith("sk-evren-"):
            return from_file
        return from_os or from_file or default
    return from_os or from_file or default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on", "evet"}


def provider() -> str:
    """Aktif sağlayıcı. Bilinmeyen değer verilirse ollama'ya düşer."""
    value = _env("PROVIDER", "ollama").lower()
    return value if value in PROVIDERS else "ollama"


def fallback_provider() -> str:
    """Ana sağlayıcı çökerse denenecek yedek ('' = yedek yok)."""
    default = "ollama" if provider() == "teknofest" else ""
    value = _env("FALLBACK_PROVIDER", default).lower()
    if value in {"", "none", "yok"}:
        return ""
    if value not in PROVIDERS or value == provider():
        return ""
    return value


def ollama_reachable(timeout_s: float = 0.4) -> bool:
    """Yerel Ollama dinliyor mu? Kapalıyken yedeğe düşmek yalnızca süreyi yakar."""
    from urllib.parse import urlparse
    import socket

    parsed = urlparse(_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


@dataclass(frozen=True)
class ModelEndpoint:
    """Tek bir model servisinin adresi ve kimliği."""

    provider: str
    role: str
    base_url: str
    model: str
    api_key: str = ""
    native_ollama: bool = False

    @property
    def chat_url(self) -> str:
        """OpenAI uyumlu (veya Ollama native) sohbet adresi."""
        base = self.base_url.rstrip("/")
        if self.native_ollama:
            return f"{base.removesuffix('/v1')}/api/chat"
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def _teknofest_endpoint(role: str) -> ModelEndpoint:
    """EVREN ortak API. Tek base URL, model alias ile VLM/LLM ayrılır."""
    shared_url = _env("TEKNOFEST_BASE_URL") or _env("EVREN_BASE_URL", DEFAULT_TEKNOFEST_BASE)
    shared_key = _env("TEKNOFEST_API_KEY") or _env("EVREN_API_KEY")
    if role == "vlm":
        return ModelEndpoint(
            provider="teknofest",
            role="vlm",
            base_url=_env("VLM_BASE_URL", shared_url),
            model=_env("VLM_MODEL", DEFAULT_VLM_MODEL),
            api_key=_env("VLM_API_KEY", shared_key),
        )
    return ModelEndpoint(
        provider="teknofest",
        role="llm",
        # LLM adresi verilmezse VLM ile aynı endpoint varsayılır
        base_url=_env("LLM_BASE_URL", _env("VLM_BASE_URL", shared_url)),
        model=_env("LLM_MODEL", DEFAULT_LLM_MODEL),
        api_key=_env("LLM_API_KEY", _env("VLM_API_KEY", shared_key)),
    )


def _ollama_endpoint(role: str) -> ModelEndpoint:
    base_url = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    vlm_model = _env("OLLAMA_MODEL", DEFAULT_OLLAMA_VLM)
    model = vlm_model if role == "vlm" else _env("OLLAMA_LLM_MODEL", vlm_model)
    return ModelEndpoint(
        provider="ollama",
        role=role,
        base_url=base_url,
        model=model,
        api_key="ollama",
        native_ollama=True,
    )


def endpoint(role: str, prov: str | None = None) -> ModelEndpoint:
    """role='vlm'|'llm' için aktif (veya verilen) sağlayıcının adresini döner."""
    if role not in ROLES:
        raise ValueError(f"Bilinmeyen rol: {role}")
    active = (prov or provider()).lower()
    if active == "teknofest":
        return _teknofest_endpoint(role)
    if active == "mock":
        model = DEFAULT_VLM_MODEL if role == "vlm" else DEFAULT_LLM_MODEL
        return ModelEndpoint(provider="mock", role=role, base_url="mock://", model=model)
    return _ollama_endpoint(role)


def vlm_endpoint(prov: str | None = None) -> ModelEndpoint:
    return endpoint("vlm", prov)


def llm_endpoint(prov: str | None = None) -> ModelEndpoint:
    return endpoint("llm", prov)


def request_timeout() -> float:
    """Tek model çağrısı için saniye cinsinden üst sınır."""
    default = 1800.0 if provider() == "teknofest" else 120.0
    return _env_float("REQUEST_TIMEOUT", default)


def max_retries() -> int:
    return max(1, _env_int("MAX_RETRIES", 3))


def max_image_side() -> int:
    """Kareler bu kenar uzunluğuna küçültülüp gönderilir (token ve süre tasarrufu)."""
    return _env_int("MAX_IMAGE_SIDE", 768)


def vlm_concurrency() -> int:
    """Aynı anda kaç VLM çağrısı açılabilir."""
    return max(1, _env_int("VLM_CONCURRENCY", 3))


def ollama_num_ctx() -> int:
    return _env_int("OLLAMA_NUM_CTX", 16384)


def demo_fast_mode() -> bool:
    """Süre bütçesi kritikse ağır adımları (YOLO, ikinci bakış, RAG) kısar."""
    return _env_bool("DEMO_FAST_MODE", False)


def demo_max_frames() -> int:
    default = 6 if demo_fast_mode() else 8
    return max(2, _env_int("DEMO_MAX_FRAMES", default))


def embed_model() -> str:
    """EVREN getirme alias'ı. rerank kullanma — R@1 düşer."""
    return _env("EMBED_MODEL", "bge-m3-embed")


def lock_policy() -> str:
    """Kategori↔risk kilidi: severity_max | category | risk."""
    return _env("LOCK_POLICY", "severity_max")


def label_critic_llm() -> bool:
    """VLM metni ile sınıf çelişince ucuz LLM eleştirmeni (kare yok)."""
    return _env_bool("LABEL_CRITIC_LLM", True)


def describe() -> str:
    """Log ve demo ekranı için tek satır özet."""
    vlm, llm = vlm_endpoint(), llm_endpoint()
    fb = fallback_provider() or "yok"
    key = vlm.api_key if provider() == "teknofest" else ""
    if provider() != "teknofest":
        key_state = "—"
    elif key.startswith("sk-evren-"):
        key_state = "ok"
    elif not key:
        key_state = "EKSIK"
    else:
        key_state = "format?"
    return (
        f"PROVIDER={provider()} (yedek: {fb}) | "
        f"VLM={vlm.model} @ {vlm.base_url or '-'} | "
        f"LLM={llm.model} @ {llm.base_url or '-'} | "
        f"EVREN_KEY={key_state}"
    )


# Eski kod yolları için geriye dönük isimler (yeni kod endpoint() kullanmalı)
API_BASE_URL: str = vlm_endpoint().base_url
MODEL_NAME: str = vlm_endpoint().model
