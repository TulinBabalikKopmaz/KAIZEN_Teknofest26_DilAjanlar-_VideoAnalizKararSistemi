"""Ekran kopyası şartname token'ını değiştirmez.

    python -m unittest tests.test_display
"""

from __future__ import annotations

import unittest

from utils.display import (
    attach_hard_case_sentence,
    category_label,
    hard_case_note,
    humanize_label,
    law_support_note,
    model_source,
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

    def test_process_flame_is_hard_case_on_normal(self) -> None:
        note = hard_case_note(
            {
                "category": "normal",
                "summary": "Görüntüde alevler var ama normal gözüküyor.",
                "events": [
                    {"time": "00:00", "event": "Görüntüde alevler var ama normal gözüküyor."}
                ],
            },
            {"risk": "Düşük"},
        )
        assert note is not None
        self.assertEqual(note["kind"], "flame")
        self.assertEqual(note["kicker"], "Zor sahne")
        self.assertIn("proses", note["text"].casefold())
        self.assertNotIn("Düşük", note["text"])
        self.assertNotIn("accident", note["text"])

    def test_process_flame_normal_proses_is_hard_case(self) -> None:
        note = hard_case_note(
            {
                "category": "normal",
                "summary": (
                    "Görüntülerdeki alevler ve duman tesisin yüksek sıcaklıkta "
                    "metal işleme prosesinin normal bir parçasıdır."
                ),
                "events": [
                    {
                        "time": "00:07",
                        "event": "Alevler ve duman tesisin normal prosesinden kaynaklanıyor.",
                    }
                ],
            },
            {"risk": "Düşük"},
        )
        assert note is not None
        self.assertEqual(note["kind"], "flame")
        self.assertEqual(note["kicker"], "Zor sahne")

    def test_process_smoke_is_hard_case_on_normal(self) -> None:
        note = hard_case_note(
            {
                "category": "normal",
                "summary": "Çalışanlar rutin aktivitesini sürdürüyor. Duman var ama normal gözüküyor.",
                "events": [{"time": "00:00", "event": "Görüntüde duman var ama normal gözüküyor."}],
            },
            {"risk": "Düşük"},
        )
        assert note is not None
        self.assertEqual(note["kind"], "smoke")

    def test_hard_case_without_risk_field_trusts_normal_category(self) -> None:
        note = hard_case_note(
            {
                "category": "normal",
                "summary": "Görüntüde alevler var ama normal gözüküyor.",
                "events": [{"time": "00:00", "event": "Görüntüde alevler var ama normal gözüküyor."}],
            }
        )
        assert note is not None
        self.assertEqual(note["kind"], "flame")

    def test_hard_case_skips_real_fire_accident(self) -> None:
        self.assertIsNone(
            hard_case_note(
                {
                    "category": "accident",
                    "summary": "Forklift alev aldı. Çalışan alevlerin arasında kaldı.",
                    "events": [{"time": "00:02", "event": "Forklift alev aldı."}],
                },
                {"risk": "Yüksek"},
            )
        )

    def test_hard_case_skips_plain_routine(self) -> None:
        self.assertIsNone(
            hard_case_note(
                {
                    "category": "normal",
                    "summary": "Çalışanlar rutin aktivitesini sürdürüyor. Normal gözüküyor.",
                    "events": [{"time": "00:00", "event": "Çalışanlar rutin aktivitesini sürdürüyor."}],
                }
            )
        )

    def test_unhedged_flame_on_normal_is_not_claimed(self) -> None:
        self.assertIsNone(
            hard_case_note(
                {
                    "category": "normal",
                    "summary": "Fabrikadan alevler yükseliyor.",
                    "events": [{"time": "00:01", "event": "Binadan alevler yükseliyor."}],
                }
            )
        )

    def test_sensor_fire_on_normal_is_hard_case(self) -> None:
        note = hard_case_note(
            {
                "category": "normal",
                "summary": "Çalışanlar rutin aktivitesini sürdürüyor.",
                "events": [{"time": "00:00", "event": "Çalışanlar rutin aktivitesini sürdürüyor."}],
            },
            {"risk": "Düşük"},
            {"fire_suspect": True},
        )
        assert note is not None
        self.assertEqual(note["kind"], "sensor")

    def test_sensor_fire_on_accident_is_not_process_note(self) -> None:
        self.assertIsNone(
            hard_case_note(
                {"category": "accident", "summary": "Makine alev alıyor."},
                {"risk": "Yüksek"},
                {"fire_suspect": True},
            )
        )

    def test_attach_hard_case_once(self) -> None:
        note = {
            "kind": "flame",
            "kicker": "Zor sahne",
            "text": "Ortamda alev görünüyor; bu makinenin olağan proses ateşi, kaçış veya zarar yok.",
        }
        first = attach_hard_case_sentence("İş kazası yok. Saha kontrol altında.", note)
        self.assertIn("proses ateşi", first)
        self.assertEqual(attach_hard_case_sentence(first, note), first)
        skipped = attach_hard_case_sentence(
            "Rutin operasyon. Alev proses kaynaklı, alarm yok.",
            note,
        )
        self.assertEqual(skipped, "Rutin operasyon. Alev proses kaynaklı, alarm yok.")

    def test_model_source_names_evren_not_teknofest(self) -> None:
        src = model_source(
            "teknofest:vlm",
            [{"provider": "teknofest", "model": "vlm", "fallback": False}],
        )
        self.assertEqual(src["kind"], "evren")
        self.assertEqual(src["label"], "EVREN")
        self.assertNotIn("teknofest", src["label"].casefold())
        self.assertNotIn("ollama", src["label"].casefold())

    def test_model_source_flags_ollama_and_fallback(self) -> None:
        ollama = model_source(
            "ollama:qwen2.5vl:7b",
            [{"provider": "ollama", "model": "qwen2.5vl:7b"}],
        )
        self.assertEqual(ollama["kind"], "ollama")
        self.assertEqual(ollama["tone"], "critical")
        mixed = model_source(
            "teknofest:vlm",
            [
                {"provider": "teknofest", "model": "vlm", "fallback": False},
                {"provider": "ollama", "model": "qwen2.5:7b", "fallback": True},
            ],
        )
        self.assertEqual(mixed["kind"], "mixed")
        backup = model_source("teknofest:vlm", backup=True)
        self.assertEqual(backup["kind"], "backup")
        self.assertIn("yedek", backup["label"].casefold())

    def test_law_note_is_small_and_keeps_model_actions_untouched(self) -> None:
        self.assertEqual(law_support_note(""), "")
        note = law_support_note("Madde 13. Ciddi ve yakın tehlike halinde çalışmayı durdurun.")
        self.assertIn("Mevzuat da benzer öneriyor", note)
        self.assertIn("md. 13", note)
        self.assertNotIn("durdurun", note)


if __name__ == "__main__":
    unittest.main()
