"""Sınıf metrikleri (accuracy / P / R / F1)."""

from __future__ import annotations

import unittest

from utils.classification import class_report


class ClassReportTests(unittest.TestCase):
    def test_perfect(self) -> None:
        pairs = [("Düşük", "Düşük"), ("Yüksek", "Yüksek")]
        report = class_report(pairs, ["Düşük", "Orta", "Yüksek"])
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["per_class"]["Düşük"]["recall"], 1.0)

    def test_skips_missing_pred(self) -> None:
        pairs = [("Düşük", "Düşük"), ("Yüksek", None)]
        report = class_report(pairs, ["Düşük", "Orta", "Yüksek"])
        self.assertEqual(report["n"], 1)

    def test_false_positive_hurts_precision(self) -> None:
        pairs = [("Düşük", "Yüksek"), ("Yüksek", "Yüksek")]
        report = class_report(pairs, ["Düşük", "Orta", "Yüksek"])
        self.assertEqual(report["per_class"]["Yüksek"]["precision"], 0.5)
        self.assertEqual(report["per_class"]["Yüksek"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
