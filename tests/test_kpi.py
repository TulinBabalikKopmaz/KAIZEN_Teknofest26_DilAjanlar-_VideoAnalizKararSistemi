"""KPI skorlayıcısının küçük kontrolleri. Model gerektirmez."""

from __future__ import annotations

import unittest

from utils.kpi import aggregate, event_hits, score_video


class KpiTests(unittest.TestCase):
    def test_self_match_is_perfect(self) -> None:
        gold = {
            "video_id": "a1",
            "filename": "a1.mp4",
            "category": "accident",
            "summary": "Forklift devrildi yerde hareketsiz kişi var",
            "events": [{"time": "00:15", "event": "Forklift devrildi"}],
            "risk": "Yüksek",
            "actions": ["Sağlık ekibini çağır"],
        }
        row = score_video(gold, gold)
        self.assertTrue(row["risk_ok"])
        self.assertEqual(row["event_hits"], 1)
        self.assertTrue(row["critical_hit"])
        self.assertTrue(row["summary_ok"])

    def test_event_within_two_seconds(self) -> None:
        hits, total = event_hits(
            [{"time": "00:15", "event": "Forklift devrildi"}],
            [{"time": "00:16", "event": "forklift devrilmesi"}],
        )
        self.assertEqual(total, 1)
        self.assertEqual(hits, 1)

    def test_missed_accident(self) -> None:
        gold = {
            "category": "accident",
            "summary": "Kaza oldu",
            "events": [{"time": "00:10", "event": "İskele çöktü"}],
            "risk": "Yüksek",
            "actions": ["Sağlık ekibini çağır"],
        }
        pred = {
            "summary": "Rutin çalışma",
            "events": [{"time": "00:00", "event": "Rutin saha hareketi"}],
            "risk": "Düşük",
            "actions": ["İzlemeye devam et"],
        }
        row = score_video(gold, pred)
        self.assertFalse(row["critical_hit"])
        self.assertFalse(row["risk_ok"])

    def test_aggregate_counts(self) -> None:
        rows = [
            score_video(
                {"category": "normal", "summary": "rutin depo", "events": [{"time": "00:00", "event": "rutin"}], "risk": "Düşük", "actions": ["izle"]},
                {"summary": "rutin depo", "events": [{"time": "00:00", "event": "rutin"}], "risk": "Düşük", "actions": ["izle"]},
            )
        ]
        summary = aggregate(rows)
        self.assertEqual(summary["n_video"], 1)
        self.assertEqual(summary["risk_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
