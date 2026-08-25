"""Ekran kopyası şartname token'ını değiştirmez.

    python -m unittest tests.test_display
"""

from __future__ import annotations

import unittest

from utils.display import (
    category_label,
    humanize_label,
    risk_label,
    spec_footnote,
    verdict,
)


class DisplayTests(unittest.TestCase):
    def test_internal_keys_stay_english(self) -> None:
        v = verdict("accident", "Yüksek")
        self.assertEqual(v["category_key"], "accident")
        self.assertEqual(v["risk_key"], "Yüksek")
        self.assertEqual(v["spec_risk"], "Yüksek")
        self.assertEqual(v["situation"], "İş kazası")
        self.assertEqual(v["decision"], "Kritik durum")

    def test_normal_is_not_low_risk_copy(self) -> None:
        v = verdict("normal", "Düşük")
        self.assertEqual(v["situation"], "Rutin operasyon")
        self.assertEqual(v["decision"], "Kontrol altında")
        self.assertNotIn("Düşük", v["decision"])
        self.assertEqual(v["tone"], "ok")

    def test_near_miss(self) -> None:
        v = verdict("near_miss", "Orta")
        self.assertEqual(v["situation"], "Ramak kala")
        self.assertEqual(v["decision"], "Yüksek dikkat")
        self.assertEqual(v["tone"], "watch")

    def test_lock_pair_used(self) -> None:
        v = verdict("accident", "Düşük")
        self.assertEqual(v["category_key"], "accident")
        self.assertEqual(v["spec_risk"], "Yüksek")

    def test_helpers(self) -> None:
        self.assertEqual(category_label("near_miss"), "Ramak kala")
        self.assertEqual(risk_label("Kritik"), "Kritik durum")
        self.assertIn("Düşük", spec_footnote())

    def test_humanize_from_label_and_spec(self) -> None:
        out = humanize_label(
            {"category": "normal", "risk": "Orta"},
            {"risk": "Düşük"},
        )
        self.assertEqual(out["situation"], "Rutin operasyon")
        self.assertEqual(out["spec_risk"], "Düşük")


if __name__ == "__main__":
    unittest.main()
