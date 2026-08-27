"""EVREN RAG yardımcısı — ağ / anahtar gerektirmez."""

from __future__ import annotations

import unittest

from utils.evren_rag import _cosine, retrieve_mevzuat_lexical


class EvrenRagTests(unittest.TestCase):
    def test_identical_vectors_are_one(self) -> None:
        self.assertAlmostEqual(_cosine([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors_are_zero(self) -> None:
        self.assertAlmostEqual(_cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_lexical_fallback_returns_madde(self) -> None:
        text = retrieve_mevzuat_lexical("Sahada alev ve duman var, patlama oldu")
        self.assertIn("Madde", text)


if __name__ == "__main__":
    unittest.main()
