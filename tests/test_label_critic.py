"""Metin eleştirmeni: yalnız yükseltir, kare istemez."""

from __future__ import annotations

import unittest

from agents.label_critic import apply_raise, needs_critic


class LabelCriticTests(unittest.TestCase):
    def test_needs_critic_on_mismatched_pair(self) -> None:
        self.assertTrue(
            needs_critic({"category": "near_miss", "risk": "Yüksek", "summary": "x", "events": []})
        )
        self.assertFalse(
            needs_critic(
                {
                    "category": "near_miss",
                    "risk": "Orta",
                    "summary": "Forklift çalışanın çok yakınından geçti",
                    "events": [{"time": "00:02", "event": "çok yakınından geçti"}],
                }
            )
        )

    def test_apply_raise_does_not_lower(self) -> None:
        label = {"category": "accident", "risk": "Yüksek", "notes": ""}
        out = apply_raise(label, "near_miss")
        self.assertEqual(out["category"], "accident")
        out = apply_raise({"category": "near_miss", "notes": ""}, "accident")
        self.assertEqual(out["category"], "accident")


if __name__ == "__main__":
    unittest.main()
