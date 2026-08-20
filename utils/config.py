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

from dotenv import load_dotenv

load_dotenv()

PROVIDERS: tuple[str, ...] = ("teknofest", "ollama", "mock")
ROLES: tuple[str, ...] = ("vlm", "llm")

# Yarışmada verilecek ortak modeller (env ile ezilebilir)
DEFAULT_VLM_MODEL: str = "Qwen3-VL-27B-Instruct"
DEFAULT_LLM_MODEL: str = "gemma-3-27b-it"
DEFAULT_OLLAMA_VLM: str = "qwen2.5vl:7b"
DEFAULT_OLLAMA_LLM: str = "qwen2.5:7b"


def _env(name: str, default: str = "") -> str:
    """Boş / sadece boşluk içeren değerleri yok sayar."""
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


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
    """Ortak API. Tek endpoint iki model senaryosu da desteklenir."""
    shared_url = _env("TEKNOFEST_BASE_URL")
    shared_key = _env("TEKNOFEST_API_KEY", "teknofest")
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
    return _env_float("REQUEST_TIMEOUT", 120.0)


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


def describe() -> str:
    """Log ve demo ekranı için tek satır özet."""
    vlm, llm = vlm_endpoint(), llm_endpoint()
    fb = fallback_provider() or "yok"
    return (
        f"PROVIDER={provider()} (yedek: {fb}) | "
        f"VLM={vlm.model} @ {vlm.base_url or '-'} | "
        f"LLM={llm.model} @ {llm.base_url or '-'}"
    )


# Eski kod yolları için geriye dönük isimler (yeni kod endpoint() kullanmalı)
API_BASE_URL: str = vlm_endpoint().base_url
MODEL_NAME: str = vlm_endpoint().model
