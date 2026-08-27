"""Risk birleştirme kurallarının küçük kontrolleri. Model gerektirmez."""

from __future__ import annotations

import unittest

from utils.risk_rules import (
    needs_second_look,
    refine_label,
    scene_is_process_routine,
    text_risk_floor,
)
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

    def test_negated_near_miss_stays_normal(self) -> None:
        label = {
            "summary": "Çalışanlar atölyede rutin işlerini sürdürüyor. "
            "Herhangi bir kaza veya tehlikeli yaklaşma gözlemlenmedi.",
            "events": [{"time": "00:00", "event": "Çalışanlar yürüyor"}],
            "risk": "Orta",
            "category": "near_miss",
            "actions": ["Rutin izlemeye devam et"],
        }
        out = refine_label(label, SceneEvidence())
        self.assertEqual(out["category"], "normal")
        self.assertEqual(out["risk"], "Düşük")

    def test_atesleme_is_not_a_fire(self) -> None:
        label = {
            "summary": "Çalışan sahada duruyor. Ateşleme işlemi devam ediyor. "
            "İşletme alanı normal görünümde.",
            "events": [{"time": "00:00", "event": "Ateşleme işlemi devam ediyor."}],
            "risk": "Yüksek",
            "category": "accident",
            "actions": ["Sağlık ekibini çağır"],
        }
        out = refine_label(label, SceneEvidence())
        self.assertEqual(out["category"], "normal")
        self.assertEqual(out["risk"], "Düşük")

    def test_process_flame_phrase_stays_normal(self) -> None:
        label = {
            "summary": "Çalışan sahada duruyor. Ateşleme işlemi devam ediyor.",
            "events": [
                {"time": "00:00", "event": "Görüntüde alevler var ama normal gözüküyor."}
            ],
            "risk": "Yüksek",
            "category": "accident",
            "actions": ["Sağlık ekibini çağır"],
        }
        out = refine_label(label, SceneEvidence())
        self.assertEqual(out["category"], "normal")
        self.assertEqual(out["risk"], "Düşük")

    def test_process_flame_normal_proses_stays_normal(self) -> None:
        """VLM proses alevi yazar; kural 'alev' deyince kaza yapmasın."""
        label = {
            "summary": (
                "Bu videoda bir iş kazası veya yaralanma olayı bulunmamaktadır. "
                "Görüntülerdeki alevler ve duman, tesisin yüksek sıcaklıkta metal "
                "işleme prosesinin normal bir parçasıdır ve kontrol altındadır."
            ),
            "events": [
                {
                    "time": "00:07",
                    "event": "Alevler ve duman tesisin normal prosesinden kaynaklanıyor.",
                }
            ],
            "risk": "Düşük",
            "category": "normal",
            "actions": ["Rutin izlemeye devam et"],
        }
        risk, cat = text_risk_floor(label)
        self.assertEqual(risk, "Düşük")
        self.assertIsNone(cat)
        out = refine_label(
            label,
            SceneEvidence(fire_suspect=True, fire_like_ratio_max=0.12),
        )
        self.assertEqual(out["category"], "normal")
        self.assertEqual(out["risk"], "Düşük")

    def test_answer_denies_accident_counts_as_process_routine(self) -> None:
        self.assertTrue(
            scene_is_process_routine(
                {
                    "category": "accident",
                    "summary": "Alevler görünüyor",
                    "events": [{"time": "00:00", "event": "Alevler yükseliyor"}],
                },
                {"risk": "Yüksek", "summary": "Alevler görünüyor"},
                "Bu videoda bir iş kazası bulunmamaktadır. Alevler tesisin normal prosesidir.",
            )
        )

    def test_real_fire_with_escape_stays_accident(self) -> None:
        label = {
            "summary": "Forklift alev aldı. Çalışanlar kaçıyor.",
            "events": [{"time": "00:04", "event": "Makine alev aldı, çalışanlar kaçıyor."}],
            "risk": "Düşük",
            "category": "normal",
            "actions": ["Rutin izlemeye devam et"],
        }
        out = refine_label(label, SceneEvidence(fire_suspect=True))
        self.assertEqual(out["category"], "accident")
        self.assertEqual(out["risk"], "Yüksek")

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
        ev = SceneEvidence(person_vehicle_very_close=True, motion_elevated=True)
        out = refine_label(label, ev)
        self.assertEqual(out["category"], "near_miss")
        self.assertEqual(out["risk"], "Orta")

    def test_second_look_trigger(self) -> None:
        label = {"risk": "Düşük", "category": "normal"}
        ev = SceneEvidence(person_vehicle_close=True, motion_elevated=True)
        self.assertTrue(needs_second_look(label, ev))
        self.assertFalse(needs_second_look({"risk": "Yüksek", "category": "accident"}, ev))
        idle = SceneEvidence(person_vehicle_very_close=True, motion_elevated=False)
        self.assertFalse(needs_second_look(label, idle))

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

    def test_short_fall_seeds_later_peak_not_early_wobble(self) -> None:
        label = {
            "summary": "Çalışan merdivenden düştü",
            "events": [{"time": "00:01", "event": "Çalışan merdivenden yere düştü"}],
            "risk": "Yüksek",
            "category": "accident",
            "actions": ["Sağlık ekibini çağır"],
        }
        ev = SceneEvidence(
            duration_sec=6.0,
            motion_peak_sec=1.0,
            motion_peaks=[1.0, 3.5, 5.5],
        )
        out = refine_label(label, ev)
        times = {item["time"] for item in (out.get("events") or [])}
        self.assertTrue("00:04" in times or "00:03" in times)

    def test_busy_yard_without_motion_stays_normal(self) -> None:
        label = {
            "summary": "Forklift çalışanın çok yakınından geçti. Neredeyse temas edeceklerdi.",
            "events": [{"time": "00:00", "event": "Forklift çalışanın çok yakınından geçti."}],
            "risk": "Orta",
            "category": "near_miss",
            "actions": ["Mesafeyi artırın"],
        }
        ev = SceneEvidence(
            person_count_max=5,
            vehicle_count_max=2,
            person_vehicle_very_close=True,
            motion_elevated=False,
        )
        out = refine_label(label, ev)
        self.assertEqual(out["category"], "normal")
        self.assertEqual(out["risk"], "Düşük")
        self.assertIn("rutin", (out["events"][0]["event"] or "").lower())

    def test_load_drop_workers_standing_is_near_miss(self) -> None:
        label = {
            "summary": "Çalışanlar kamyonun arkasından yük çıkarken yük yere düştü. Çalışanlar yere düşen yükün etrafında duruyor.",
            "events": [
                {"time": "00:00", "event": "Yük kamyonun arkasından düşerek yere saçıldı."}
            ],
            "risk": "Yüksek",
            "category": "accident",
            "actions": ["Sağlık ekibini çağır"],
        }
        out = refine_label(label, None)
        self.assertEqual(out["category"], "near_miss")
        self.assertEqual(out["risk"], "Orta")
        self.assertIn("kurtul", (out["events"][0]["event"] or "").lower())

    def test_worker_fell_with_load_stays_accident(self) -> None:
        label = {
            "summary": "Çalışan yükü kamyonun arkasından çekerken dengesini kaybetti ve yere düştü.",
            "events": [
                {"time": "00:14", "event": "Çalışan yükü çekerken dengesini kaybetti ve yere düştü."}
            ],
            "risk": "Yüksek",
            "category": "accident",
            "actions": ["Sağlık ekibini çağır"],
        }
        out = refine_label(label, None)
        self.assertEqual(out["category"], "accident")
        self.assertEqual(out["risk"], "Yüksek")


if __name__ == "__main__":
    unittest.main()
