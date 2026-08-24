"""VLM ve LLM çağrıları için tek arayüz: async, retry, yedek sağlayıcı, süre logu.

API şekli değişirse sadece bu dosya değişir; ajanlar ve demo yolu chat_vlm /
chat_llm imzasını görür.

    result = await chat_vlm("Bu karede ne oluyor?", image_paths=[frame])
    # EVREN resmi alias `vlm` JPEG kabul etmez; kısa mp4 klibi gönder:
    result = await chat_vlm("Bu videoda ne oluyor?", video_path=clip_mp4)
    print(result.text, result.latency_s)
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import aiohttp

from utils.config import (
    ModelEndpoint,
    endpoint,
    fallback_provider,
    max_image_side,
    max_retries,
    ollama_num_ctx,
    request_timeout,
    vlm_concurrency,
)
from utils.image import encode_image_b64

RETRY_STATUS: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class ModelCallError(RuntimeError):
    """Tüm denemeler ve yedek sağlayıcı tükendikten sonra atılır."""


@dataclass
class ChatResult:
    """Tek çağrının çıktısı ve ölçümü."""

    text: str
    role: str
    provider: str
    model: str
    latency_s: float
    attempts: int = 1
    fallback_used: bool = False
    note: str = ""

    def as_log_row(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "latency_s": round(self.latency_s, 2),
            "attempts": self.attempts,
            "fallback": self.fallback_used,
        }


CALL_LOG: list[ChatResult] = []
_SEMAPHORES: dict[tuple[int, str], asyncio.Semaphore] = {}


def reset_call_log() -> None:
    CALL_LOG.clear()


def call_log_rows() -> list[dict[str, Any]]:
    return [item.as_log_row() for item in CALL_LOG]


def total_model_seconds() -> float:
    return sum(item.latency_s for item in CALL_LOG)


def _semaphore(role: str) -> asyncio.Semaphore:
    """Çalışan event loop başına eşzamanlılık sınırı."""
    loop_key = (id(asyncio.get_running_loop()), role)
    sem = _SEMAPHORES.get(loop_key)
    if sem is None:
        limit = vlm_concurrency() if role == "vlm" else max(2, vlm_concurrency())
        sem = asyncio.Semaphore(limit)
        _SEMAPHORES[loop_key] = sem
    return sem


def _images_b64(image_paths: Sequence[Path | str]) -> list[str]:
    side = max_image_side()
    return [encode_image_b64(path, max_side=side) for path in image_paths]


def _openai_payload(
    ep: ModelEndpoint,
    prompt: str,
    images: list[str],
    system: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    video_b64: str | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    # EVREN: `vlm` yalnız video_url; JPEG karışınca HTTP 400. İkisini birden gönderme.
    if video_b64:
        content.append(
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            }
        )
    else:
        for b64 in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    use_list = bool(video_b64 or images)
    messages.append({"role": "user", "content": content if use_list else prompt})

    payload: dict[str, Any] = {
        "model": ep.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _ollama_payload(
    ep: ModelEndpoint,
    prompt: str,
    images: list[str],
    system: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    video_b64: str | None = None,
) -> dict[str, Any]:
    del video_b64  # Ollama native chat kare listesi kullanır; video_url yok.
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        user_message["images"] = images
    messages.append(user_message)

    payload: dict[str, Any] = {
        "model": ep.model,
        "stream": False,
        "messages": messages,
        "options": {
            "num_ctx": ollama_num_ctx(),
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    if json_mode:
        payload["format"] = "json"
    return payload


def _extract_text(ep: ModelEndpoint, data: dict[str, Any]) -> str:
    if ep.native_ollama:
        return str((data.get("message") or {}).get("content") or "")
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def _mock_text(role: str, json_mode: bool) -> str:
    if role == "llm" and json_mode:
        return json.dumps(
            {
                "answer": (
                    "Mock cevap: 00:03'te forklift çalışanın çok yakınından geçti, "
                    "temas olmadı ama ramak kala durumu var."
                ),
                "actions": ["Forklift trafiğini durdur", "Yaya yolunu şeritle ayır"],
            },
            ensure_ascii=False,
        )
    if role == "vlm":
        return json.dumps(
            {
                "category": "near_miss",
                "summary": "Forklift çalışanın çok yakınından geçti. Neredeyse temas edeceklerdi.",
                "events": [
                    {
                        "time": "00:03",
                        "event": "Forklift çalışanın çok yakınından geçti.",
                        "event_type": "near_miss",
                        "severity": "orta",
                    }
                ],
                "risk": "Orta",
                "actions": ["Forklift trafiğini durdur", "Yaya yolunu işaretle"],
            },
            ensure_ascii=False,
        )
    return "Mock yanıt: sahada tehlikeli yaklaşma var, forklift trafiğini durdurun."


async def _post_once(
    ep: ModelEndpoint,
    payload: dict[str, Any],
    timeout_s: float,
) -> str:
    headers = {"Content-Type": "application/json"}
    if ep.api_key:
        headers["Authorization"] = f"Bearer {ep.api_key}"

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(ep.chat_url, headers=headers, json=payload) as response:
            body = await response.text()
            if response.status >= 400:
                retryable = response.status in RETRY_STATUS
                message = f"HTTP {response.status} ({ep.label}): {body[:400]}"
                raise _HttpError(message, retryable=retryable)
            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                raise _HttpError(
                    f"Geçersiz JSON yanıt ({ep.label}): {body[:200]}", retryable=True
                ) from exc
    return _extract_text(ep, data)


class _HttpError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


async def _call_provider(
    ep: ModelEndpoint,
    prompt: str,
    images: list[str],
    *,
    system: str | None,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    timeout_s: float,
    attempts_allowed: int,
    video_b64: str | None = None,
) -> tuple[str, int]:
    """Tek sağlayıcıda retry + backoff. (metin, deneme sayısı) döner."""
    if ep.provider == "mock":
        await asyncio.sleep(0.05)
        return _mock_text(ep.role, json_mode), 1

    # Resmi doküman: uzun video isteğini otomatik tekrarlama. Prefix cache de bozulur.
    if video_b64 and not ep.native_ollama:
        attempts_allowed = 1

    send_video = video_b64 if not ep.native_ollama else None
    send_images = images if not send_video else []
    build = _ollama_payload if ep.native_ollama else _openai_payload
    payload = build(
        ep, prompt, send_images, system, temperature, max_tokens, json_mode, send_video
    )

    last_error: Exception | None = None
    for attempt in range(1, attempts_allowed + 1):
        try:
            text = await _post_once(ep, payload, timeout_s)
            if not text.strip():
                raise _HttpError(f"Boş yanıt ({ep.label})", retryable=True)
            return text, attempt
        except _HttpError as exc:
            last_error = exc
            if not exc.retryable:
                break
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_error = RuntimeError(f"Bağlantı hatası ({ep.label}): {exc}")
        if attempt < attempts_allowed:
            wait_s = min(2.0 * attempt, 6.0) + random.uniform(0, 0.4)
            print(f"  [model] {ep.label} hata, {wait_s:.1f}s sonra tekrar: {last_error}")
            await asyncio.sleep(wait_s)

    raise ModelCallError(str(last_error or f"{ep.label} çağrısı başarısız"))


async def _chat(
    role: str,
    prompt: str,
    image_paths: Sequence[Path | str] = (),
    *,
    video_path: Path | str | None = None,
    system: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 768,
    json_mode: bool = False,
    timeout_s: float | None = None,
    allow_fallback: bool = True,
) -> ChatResult:
    images = _images_b64(image_paths) if image_paths else []
    timeout = timeout_s or request_timeout()
    attempts_allowed = max_retries()
    primary = endpoint(role)
    video_b64: str | None = None
    if video_path and primary.provider != "mock":
        from utils.video_clip import encode_video_b64

        video_b64 = await asyncio.to_thread(encode_video_b64, video_path)

    started = perf_counter()
    async with _semaphore(role):
        try:
            text, attempts = await _call_provider(
                primary,
                prompt,
                images,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                timeout_s=timeout,
                attempts_allowed=attempts_allowed,
                video_b64=video_b64 if primary.provider == "teknofest" else None,
            )
            result = ChatResult(
                text=text,
                role=role,
                provider=primary.provider,
                model=primary.model,
                latency_s=perf_counter() - started,
                attempts=attempts,
            )
        except ModelCallError as primary_error:
            fb_name = fallback_provider() if allow_fallback else ""
            if not fb_name:
                raise
            fb = endpoint(role, fb_name)
            print(f"  [model] {primary.label} düştü, yedeğe geçiliyor: {fb.label}")
            text, attempts = await _call_provider(
                fb,
                prompt,
                images,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                timeout_s=timeout,
                attempts_allowed=attempts_allowed,
                video_b64=video_b64 if fb.provider == "teknofest" else None,
            )
            result = ChatResult(
                text=text,
                role=role,
                provider=fb.provider,
                model=fb.model,
                latency_s=perf_counter() - started,
                attempts=attempts,
                fallback_used=True,
                note=f"birincil hata: {primary_error}",
            )

    CALL_LOG.append(result)
    return result


async def chat_vlm(
    prompt: str,
    image_paths: Sequence[Path | str] = (),
    *,
    video_path: Path | str | None = None,
    system: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 768,
    json_mode: bool = False,
    timeout_s: float | None = None,
) -> ChatResult:
    """Görsel veya kısa video klibi + metin (Video Analyzer yolu).

    PROVIDER=teknofest iken resmi `vlm` alias'ı JPEG reddeder; `video_path` ver.
    Kareler yedek (Ollama) için durur; ikinci bakışta aynı klip baytlarını tekrar gönder.
    """
    return await _chat(
        "vlm",
        prompt,
        image_paths,
        video_path=video_path,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        timeout_s=timeout_s,
    )


async def chat_llm(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 400,
    json_mode: bool = False,
    timeout_s: float | None = None,
) -> ChatResult:
    """Sadece metin isteği (Risk Assessor, Action Recommender, final cevap)."""
    return await _chat(
        "llm",
        prompt,
        (),
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        timeout_s=timeout_s,
    )


def _run_sync(coro: Any) -> Any:
    """Streamlit / CLI gibi senkron bağlamlar için."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Zaten bir event loop içindesiniz; async sürümü kullanın.")


def chat_vlm_sync(prompt: str, image_paths: Sequence[Path | str] = (), **kwargs: Any) -> ChatResult:
    return _run_sync(chat_vlm(prompt, image_paths, **kwargs))


def chat_llm_sync(prompt: str, **kwargs: Any) -> ChatResult:
    return _run_sync(chat_llm(prompt, **kwargs))


async def ping(
    role: str,
    image_path: Path | str | None = None,
    video_path: Path | str | None = None,
) -> ChatResult:
    """Endpoint doğrulama: kısa istek, düşük token, yedek kapalı."""
    if role == "vlm":
        if video_path:
            return await _chat(
                "vlm",
                "Bu kısa videoda ne oluyor? Tek cümle Türkçe cevap ver.",
                (),
                video_path=video_path,
                max_tokens=64,
                timeout_s=request_timeout(),
                allow_fallback=False,
            )
        return await _chat(
            "vlm",
            "Bu karede kaç kişi var? Tek cümle Türkçe cevap ver.",
            [image_path] if image_path else (),
            max_tokens=64,
            timeout_s=min(request_timeout(), 60.0),
            allow_fallback=False,
        )
    return await _chat(
        "llm",
        "Tek kelimeyle cevap ver: iskelede korkuluk yoksa risk seviyesi nedir?",
        (),
        max_tokens=16,
        timeout_s=min(request_timeout(), 60.0),
        allow_fallback=False,
    )
