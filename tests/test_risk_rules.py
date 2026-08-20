"""Risk birleştirme kurallarının küçük kontrolleri. Model gerektirmez."""

from __future__ import annotations

import unittest

from utils.risk_rules import needs_second_look, refine_label, text_risk_floor
from utils.scene_evidence import SceneEvidence


class RiskRuleTests(unittest.TestCase):
    def test_text_collision_raises_high(self) -> None:
        risk, cat = text_risk_floor(
            {"summary": "Motosiklet forklifte çarptı", "events": [], "actions": []}
        )
        self.assertEqual(risk, "Yüksek")
        self.assertEqual(cat, "accident")

    def test_refine_raises_near_miss_from_text(self) -> None:
        label = {
            "summary": "Motosiklet forkliftin önünden geçti",
            "events": [{"time": "00:03", "event": "neredeyse çarpışacaktı"}],
            "risk": "Düşük",
            "category": "normal",
            "actions": ["Rutin izlemeye devam et"],
        }
        out = refine_label(label, None)
        self.assertEqual(out["risk"], "Orta")
        self.assertEqual(out["category"], "near_miss")

    def test_fire_evidence_raises(self) -> None:
        label = {
            "summary": "Garajda tamir",
            "events": [{"time": "00:00", "event": "Araç tamiri"}],
            "risk": "Düşük",
            "category": "normal",
            "actions": ["Rutin izlemeye devam et"],
        }
        ev = SceneEvidence(fire_suspect=True, fire_like_ratio_max=0.09)
        out = refine_label(label, ev)
        self.assertEqual(out["risk"], "Yüksek")
        self.assertEqual(out["category"], "accident")

    def test_risk_snaps_to_category_lock(self) -> None:
        """Anlaşmazlıkta daha ağır sinyal kazanır (severity_max)."""
        accident = refine_label(
            {
                "summary": "Çalışan yere düştü",
                "events": [{"time": "00:04", "event": "Çalışan yere düştü"}],
                "risk": "Orta",
                "category": "accident",
                "actions": ["Rutin izlemeye devam et"],
            },
            None,
        )
        self.assertEqual(accident["category"], "accident")
        self.assertEqual(accident["risk"], "Yüksek")

        hotter_risk = refine_label(
            {
                "summary": "Forklift çalışanın çok yakınından geçti",
                "events": [{"time": "00:03", "event": "çok yakınından geçti"}],
                "risk": "Yüksek",
                "category": "near_miss",
                "actions": ["Alarm ver"],
            },
            None,
        )
        self.assertEqual(hotter_risk["category"], "accident")
        self.assertEqual(hotter_risk["risk"], "Yüksek")

    def test_unhedged_fall_overrides_near_miss(self) -> None:
        from utils.risk_rules import has_unhedged_accident

        self.assertTrue(has_unhedged_accident("Çalışan yere düştü"))
        self.assertFalse(has_unhedged_accident("neredeyse yere düştü"))
        out = refine_label(
            {
                "summary": "Çalışan yere düştü ve hareketsiz yatıyor",
                "events": [{"time": "00:04", "event": "Çalışan yere düştü"}],
                "risk": "Orta",
                "category": "near_miss",
                "actions": ["Alarm"],
            },
            None,
        )
        self.assertEqual(out["category"], "accident")
        self.assertEqual(out["risk"], "Yüksek")

    def test_very_close_is_near_miss_orta_not_yuksek(self) -> None:
        label = {
            "summary": "Saha hareketi",
            "events": [{"time": "00:00", "event": "Araç geçişi"}],
            "risk": "Düşük",
            "category": "normal",
            "actions": ["Rutin izlemeye devam et"],
        }
        ev = SceneEvidence(person_vehicle_very_close=True)
        out = refine_label(label, ev)
        self.assertEqual(out["category"], "near_miss")
        self.assertEqual(out["risk"], "Orta")

    def test_second_look_trigger(self) -> None:
        label = {"risk": "Düşük", "category": "normal"}
        ev = SceneEvidence(person_vehicle_close=True)
        self.assertTrue(needs_second_look(label, ev))
        self.assertFalse(needs_second_look({"risk": "Yüksek", "category": "accident"}, ev))

    def test_second_look_on_undercalled_near_miss(self) -> None:
        ev = SceneEvidence(motion_elevated=True, person_vehicle_close=True)
        self.assertTrue(
            needs_second_look({"risk": "Orta", "category": "near_miss"}, ev)
        )

    def test_collapse_text_raises_high(self) -> None:
        risk, cat = text_risk_floor(
            {"summary": "İskele çöktü çalışan enkaz altında kaldı", "events": [], "actions": []}
        )
        self.assertEqual(risk, "Yüksek")
        self.assertEqual(cat, "accident")


if __name__ == "__main__":
    unittest.main()
