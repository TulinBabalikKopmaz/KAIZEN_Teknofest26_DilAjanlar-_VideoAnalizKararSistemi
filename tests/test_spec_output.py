"""Şartname JSON çevirisinin küçük kontrolleri. Model gerektirmez.

    python -m unittest tests.test_spec_output
"""

from __future__ import annotations

import unittest

from utils.spec_output import (
    incidents_to_spec,
    normalize_risk,
    parse_vlm_line,
    pipeline_result_to_spec,
    risk_from_category,
    seconds_to_mmss,
)


class SpecOutputTests(unittest.TestCase):
    def test_seconds_to_mmss(self) -> None:
        self.assertEqual(seconds_to_mmss(15), "00:15")
        self.assertEqual(seconds_to_mmss(75.4), "01:15")
        self.assertEqual(seconds_to_mmss("00:09"), "00:09")

    def test_normalize_risk(self) -> None:
        self.assertEqual(normalize_risk("Güvenli"), "Düşük")
        self.assertEqual(normalize_risk("Kritik"), "Yüksek")
        self.assertEqual(normalize_risk("Orta"), "Orta")

    def test_risk_from_category_lock(self) -> None:
        self.assertEqual(risk_from_category("accident", "Düşük"), "Yüksek")
        self.assertEqual(risk_from_category("near_miss", "Yüksek"), "Orta")
        self.assertEqual(risk_from_category("normal", "Orta"), "Düşük")
        self.assertEqual(risk_from_category(None, "Kritik"), "Yüksek")

    def test_lock_pair_policies(self) -> None:
        from utils.spec_output import lock_pair

        self.assertEqual(
            lock_pair("near_miss", "Yüksek", policy="severity_max"),
            ("accident", "Yüksek"),
        )
        self.assertEqual(
            lock_pair("near_miss", "Yüksek", policy="category"),
            ("near_miss", "Orta"),
        )
        self.assertEqual(
            lock_pair("accident", "Orta", policy="risk"),
            ("near_miss", "Orta"),
        )
        self.assertEqual(
            lock_pair("accident", "Orta", policy="severity_max"),
            ("accident", "Yüksek"),
        )

    def test_pipeline_matches_gold_keys(self) -> None:
        spec = pipeline_result_to_spec(
            {
                "zaman_damgasi": "00:15",
                "olay_ozeti": (
                    "Durum Açıklaması: Forklift devrildi | Risk: Kritik | "
                    "Aksiyon: Sağlık ekibini çağır"
                ),
                "risk_seviyesi": "Kritik",
                "onerilen_aksiyonlar": ["Sağlık ekibini çağır", "Alanı güvenlik altına al"],
            }
        )
        self.assertEqual(set(spec), {"summary", "events", "risk", "actions"})
        self.assertEqual(spec["risk"], "Yüksek")
        self.assertEqual(spec["events"][0]["time"], "00:15")
        self.assertEqual(spec["events"][0]["event"], "Forklift devrildi")

    def test_empty_video_is_normal(self) -> None:
        spec = incidents_to_spec([])
        self.assertEqual(spec["risk"], "Düşük")
        self.assertEqual(spec["events"][0]["time"], "00:00")


if __name__ == "__main__":
    unittest.main()
