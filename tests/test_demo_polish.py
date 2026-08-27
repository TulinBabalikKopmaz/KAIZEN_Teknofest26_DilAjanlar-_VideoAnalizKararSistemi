"""Ekran cilası: eski oturum, proses alevi, kopya zaman, mevzuat notu."""

from __future__ import annotations

import unittest

from utils.demo_pipeline import DemoResult, polish_demo_result


class PolishDemoTests(unittest.TestCase):
    def test_process_flame_card_follows_answer_not_stale_accident(self) -> None:
        raw = DemoResult(
            video="x.mp4",
            user_prompt="",
            answer=(
                "Bu videoda bir iş kazası bulunmamaktadır. "
                "Alevler tesisin normal prosesidir."
            ),
            spec={
                "summary": "Alevler görünüyor",
                "events": [
                    {"time": "00:00", "event": "Alevler yükseliyor"},
                    {"time": "00:07", "event": "Alevler yükseliyor"},
                ],
                "risk": "Yüksek",
                "actions": ["Acil tıbbi yardım çağırın"],
            },
            label={
                "category": "accident",
                "risk": "Yüksek",
                "summary": "Alevler görünüyor",
                "events": [
                    {"time": "00:00", "event": "Alevler yükseliyor"},
                    {"time": "00:07", "event": "Alevler yükseliyor"},
                ],
            },
            frames=[],
            evidence={},
            timings={},
        )
        out = polish_demo_result(raw)
        self.assertEqual(out.label["category"], "normal")
        self.assertEqual(out.spec["risk"], "Düşük")
        self.assertEqual(len(out.spec["events"]), 1)
        self.assertTrue(out.law_note)
        self.assertIn("Mevzuat", out.law_note)

    def test_coerce_old_object_without_law_note(self) -> None:
        class Old:
            video = "a.mp4"
            user_prompt = ""
            answer = "Rutin"
            spec = {"summary": "x", "events": [], "risk": "Düşük", "actions": []}
            label = {"category": "normal"}
            frames = []
            evidence = {}
            timings = {}
            model_calls = []
            provider = "teknofest"
            total_s = 1.0
            fast_mode = False
            out_dir = ""
            warnings = []

        out = DemoResult.coerce(Old())
        self.assertEqual(out.law_note, "")
        polished = polish_demo_result(Old())
        self.assertTrue(hasattr(polished, "law_note"))


if __name__ == "__main__":
    unittest.main()
