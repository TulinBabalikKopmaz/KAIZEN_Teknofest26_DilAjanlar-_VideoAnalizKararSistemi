"""Model istemcisi ve konfigürasyonun kontrolleri. Ağ / model gerektirmez."""

from __future__ import annotations

import asyncio
import os
import unittest

from utils import config
from utils.model_client import chat_llm, chat_vlm, call_log_rows, reset_call_log

TOUCHED_KEYS = (
    "PROVIDER",
    "FALLBACK_PROVIDER",
    "TEKNOFEST_BASE_URL",
    "TEKNOFEST_API_KEY",
    "EVREN_API_KEY",
    "EVREN_BASE_URL",
    "VLM_BASE_URL",
    "VLM_MODEL",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_LLM_MODEL",
)


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in TOUCHED_KEYS}
        for key in TOUCHED_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_base_url_without_v1_is_normalized(self) -> None:
        os.environ["PROVIDER"] = "teknofest"
        os.environ["TEKNOFEST_BASE_URL"] = "https://gpu.example.com"
        self.assertEqual(
            config.vlm_endpoint().chat_url, "https://gpu.example.com/v1/chat/completions"
        )

    def test_base_url_with_v1_is_not_duplicated(self) -> None:
        os.environ["PROVIDER"] = "teknofest"
        os.environ["VLM_BASE_URL"] = "https://gpu.example.com/v1/"
        self.assertEqual(
            config.vlm_endpoint().chat_url, "https://gpu.example.com/v1/chat/completions"
        )

    def test_llm_falls_back_to_shared_endpoint(self) -> None:
        os.environ["PROVIDER"] = "teknofest"
        os.environ["TEKNOFEST_BASE_URL"] = "https://gpu.example.com/v1"
        os.environ["VLM_MODEL"] = "qwen3-vl"
        os.environ["LLM_MODEL"] = "gemma3"
        vlm, llm = config.vlm_endpoint(), config.llm_endpoint()
        self.assertEqual(vlm.chat_url, llm.chat_url)
        self.assertEqual((vlm.model, llm.model), ("qwen3-vl", "gemma3"))

    def test_evren_defaults_use_official_aliases(self) -> None:
        os.environ["PROVIDER"] = "teknofest"
        vlm, llm = config.vlm_endpoint(), config.llm_endpoint()
        self.assertEqual(vlm.model, "vlm")
        self.assertEqual(llm.model, "llm-fast")
        self.assertIn("evren-llmapi.ssyz.org.tr", vlm.chat_url)

    def test_ollama_uses_native_chat_endpoint(self) -> None:
        os.environ["PROVIDER"] = "ollama"
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434/v1"
        endpoint = config.vlm_endpoint()
        self.assertTrue(endpoint.native_ollama)
        self.assertEqual(endpoint.chat_url, "http://127.0.0.1:11434/api/chat")

    def test_ollama_llm_defaults_to_vlm_model(self) -> None:
        os.environ["PROVIDER"] = "ollama"
        os.environ["OLLAMA_MODEL"] = "qwen2.5vl:7b"
        self.assertEqual(config.llm_endpoint().model, "qwen2.5vl:7b")
        os.environ["OLLAMA_LLM_MODEL"] = "gemma2:9b"
        self.assertEqual(config.llm_endpoint().model, "gemma2:9b")

    def test_fallback_only_when_different_provider(self) -> None:
        os.environ["PROVIDER"] = "teknofest"
        self.assertEqual(config.fallback_provider(), "ollama")
        os.environ["FALLBACK_PROVIDER"] = "teknofest"
        self.assertEqual(config.fallback_provider(), "")
        os.environ["FALLBACK_PROVIDER"] = "none"
        self.assertEqual(config.fallback_provider(), "")

    def test_unknown_provider_falls_back_to_ollama(self) -> None:
        os.environ["PROVIDER"] = "uzayli-model"
        self.assertEqual(config.provider(), "ollama")


class MockCallTests(unittest.TestCase):
    def setUp(self) -> None:
        self._provider = os.environ.get("PROVIDER")
        os.environ["PROVIDER"] = "mock"
        reset_call_log()

    def tearDown(self) -> None:
        if self._provider is None:
            os.environ.pop("PROVIDER", None)
        else:
            os.environ["PROVIDER"] = self._provider

    def test_vlm_and_llm_calls_are_logged(self) -> None:
        vlm = asyncio.run(chat_vlm("kare yorumla"))
        llm = asyncio.run(chat_llm("risk nedir", json_mode=True))
        self.assertIn("summary", vlm.text)
        self.assertIn("answer", llm.text)
        rows = call_log_rows()
        self.assertEqual([row["role"] for row in rows], ["vlm", "llm"])
        self.assertTrue(all(row["provider"] == "mock" for row in rows))


if __name__ == "__main__":
    unittest.main()
