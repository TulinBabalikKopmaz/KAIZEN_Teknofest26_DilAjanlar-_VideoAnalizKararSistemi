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

    def test_stemmed_event_text_still_matches(self) -> None:
        hits, total = event_hits(
            [{"time": "00:32", "event": "Forklift çapması sonucu yük dolu raflar devriliyor."}],
            [{"time": "00:34", "event": "Raf sistemi çöktü ve yükler yere devrildi."}],
        )
        self.assertEqual(total, 1)
        self.assertEqual(hits, 1)

    def test_box_and_load_are_the_same_token(self) -> None:
        hits, total = event_hits(
            [{"time": "00:06", "event": "Kamyondaki yük çalışanın üstüne düşüyor."}],
            [{"time": "00:04", "event": "Kamyonun arkasından bir koli düşerek çalışanın üzerine indi."}],
        )
        self.assertEqual(total, 1)
        self.assertEqual(hits, 1)

    def test_rutin_phrase_matches_generic_normal(self) -> None:
        hits, total = event_hits(
            [{"time": "00:00", "event": "Çalışanlar rutin aktivitesini sürdürüyor."}],
            [{"time": "00:00", "event": "Çalışanlar rutin aktivitesini sürdürüyor. Çalışan fabrikada yürüyor."}],
        )
        self.assertEqual(hits, 1)

    def test_synonym_event_text_still_matches(self) -> None:
        hits, total = event_hits(
            [{"time": "00:03", "event": "Çalışan arabanın altında kaldı."}],
            [{"time": "00:02", "event": "Araç çalışanın üzerine çarptı."}],
        )
        self.assertEqual(total, 1)
        self.assertEqual(hits, 1)

    def test_unrelated_event_text_does_not_match(self) -> None:
        hits, total = event_hits(
            [{"time": "00:00", "event": "Rutin saha yürüyüşü"}],
            [{"time": "00:00", "event": "Depo rafı çöktü"}],
        )
        self.assertEqual(hits, 0)

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

    def test_pred_risk_follows_category_even_if_model_mismatched(self) -> None:
        gold = {
            "category": "accident",
            "summary": "Çalışan yere düştü",
            "events": [{"time": "00:10", "event": "Çalışan yere düştü"}],
            "risk": "Yüksek",
            "actions": ["Sağlık ekibini çağır"],
        }
        pred = {
            "category": "accident",
            "summary": "Çalışan yere düştü",
            "events": [{"time": "00:10", "event": "Çalışan yere düştü"}],
            "risk": "Orta",
            "actions": ["Sağlık ekibini çağır"],
        }
        row = score_video(gold, pred)
        self.assertTrue(row["risk_ok"])
        self.assertEqual(row["risk_pred"], "Yüksek")

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
