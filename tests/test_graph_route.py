"""LangGraph yönlendirme: ikinci bakış koşulu."""

from __future__ import annotations

import unittest

from graph_pipeline import _after_analyzer


class GraphRouteTests(unittest.TestCase):
    def test_calm_with_trigger_goes_second_look(self) -> None:
        state = {
            "analysis_result": {"event": "Durum Açıklaması: Güvenli ortam | Risk: Güvenli"},
            "trigger_reason": "hareket tepe",
        }
        self.assertEqual(_after_analyzer(state), "second_look")  # type: ignore[arg-type]

    def test_critical_skips_second_look(self) -> None:
        state = {
            "analysis_result": {"event": "Risk: Kritik | forklift devrildi"},
            "trigger_reason": "hareket tepe",
        }
        self.assertEqual(_after_analyzer(state), "risk_assessor")  # type: ignore[arg-type]

    def test_already_done(self) -> None:
        state = {
            "analysis_result": {"event": "Güvenli"},
            "trigger_reason": "x",
            "second_look_done": True,
        }
        self.assertEqual(_after_analyzer(state), "risk_assessor")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
