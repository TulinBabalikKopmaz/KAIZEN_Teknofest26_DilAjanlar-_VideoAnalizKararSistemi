"""Olay listesi temizliği ve JSON ayrıştırma kontrolleri."""

from __future__ import annotations

import unittest

from utils.label_json import (
    align_events_to_motion,
    dedupe_events,
    label_to_spec,
    lift_clip_relative_times,
    seed_events_from_motion,
    snap_events_to_frame_times,
)


class DedupeEventTests(unittest.TestCase):
    def test_repeated_event_collapses_to_earliest(self) -> None:
        events = [
            {"time": "00:01", "event": "Kutu düşme tehlikesi."},
            {"time": "00:02", "event": "Kutu düşme tehlikesi."},
            {"time": "00:03", "event": "Kutu düşme tehlikesi."},
        ]
        out = dedupe_events(events)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["time"], "00:01")

    def test_longer_text_wins_when_merging(self) -> None:
        events = [
            {"time": "00:04", "event": "Yük düştü"},
            {"time": "00:05", "event": "Yük düştü ve çalışan son anda kaçtı"},
        ]
        out = dedupe_events(events)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["time"], "00:04")
        self.assertIn("son anda", out[0]["event"])

    def test_different_events_are_kept(self) -> None:
        events = [
            {"time": "00:02", "event": "Forklift çalışanın çok yakınından geçti"},
            {"time": "00:09", "event": "Çalışan yüksekten yere düştü"},
        ]
        self.assertEqual(len(dedupe_events(events)), 2)

    def test_far_apart_same_text_is_kept(self) -> None:
        events = [
            {"time": "00:02", "event": "Forklift çalışanın çok yakınından geçti"},
            {"time": "00:12", "event": "Forklift çalışanın çok yakınından geçti"},
        ]
        self.assertEqual(len(dedupe_events(events)), 2)

    def test_event_count_is_capped_and_sorted(self) -> None:
        events = [
            {"time": "00:09", "event": "Çalışan yerde hareketsiz"},
            {"time": "00:01", "event": "Forklift hızla yaklaştı"},
            {"time": "00:05", "event": "Forklift devrildi"},
            {"time": "00:07", "event": "Yük yere saçıldı"},
        ]
        out = dedupe_events(events, max_events=3)
        self.assertEqual([item["time"] for item in out], ["00:01", "00:05", "00:07"])

    def test_empty_and_malformed_entries_are_dropped(self) -> None:
        events = [{"time": "00:01", "event": ""}, "bozuk", {"time": "00:02", "event": "Çalışan düştü"}]
        out = dedupe_events(events)
        self.assertEqual(len(out), 1)

    def test_pipeline_order_snap_then_dedupe(self) -> None:
        """Kareye yapıştırma iki olayı aynı saniyeye getirirse tek olay kalmalı."""
        events = [
            {"time": "00:02", "event": "Kutu düşme tehlikesi"},
            {"time": "00:03", "event": "Kutu düşme tehlikesi"},
        ]
        snapped = snap_events_to_frame_times(events, ["00:00", "00:02", "00:04"])
        out = dedupe_events(snapped)
        self.assertEqual(len(out), 1)
        spec = label_to_spec({"summary": "x", "events": out, "risk": "Orta", "actions": ["y"]})
        self.assertEqual(len(spec["events"]), 1)

    def test_zero_timestamp_snaps_to_motion_peak(self) -> None:
        events = [{"time": "00:00", "event": "Raf çöktü"}]
        out = align_events_to_motion(events, [34.0])
        self.assertEqual(out[0]["time"], "00:34")

    def test_zero_timestamp_prefers_primary_peak_not_earliest(self) -> None:
        events = [{"time": "00:00", "event": "Raf çöktü"}]
        out = align_events_to_motion(
            events, [2.0, 34.0], primary_peak_s=34.0
        )
        self.assertEqual(out[0]["time"], "00:34")

    def test_short_accident_prefers_later_motion_peak(self) -> None:
        from utils.label_json import preferred_incident_peak_s

        # 1. sn sallanma, 3.5 sn düşme (merdiven klibi)
        self.assertEqual(
            preferred_incident_peak_s(
                [1.0, 3.5, 5.5], 1.0, duration_s=6.0, category="accident"
            ),
            3.5,
        )
        # Uzun depo videosunda asıl tepeye dokunma
        self.assertEqual(
            preferred_incident_peak_s(
                [1.0, 32.0], 32.0, duration_s=62.0, category="accident"
            ),
            32.0,
        )

    def test_onset_seed_covers_two_seconds_before_peak(self) -> None:
        """Gold 00:18, VLM/tepe 00:21 → 2 sn önceki aday ±2 sn içinde yakalar."""
        events = [{"time": "00:21", "event": "Çalışanın üzerine yük düştü."}]
        out = seed_events_from_motion(events, [21.0])
        times = {item["time"] for item in out}
        self.assertTrue("00:19" in times or "00:21" in times)
        # 00:21 ve 00:19 pencere=2 ile birleşirse erken zaman (00:19) kalır
        self.assertIn(out[0]["time"], {"00:19", "00:21"})

    def test_repeat_near_miss_peaks_stay_separate(self) -> None:
        events = [{"time": "00:03", "event": "Forklift çalışanın çok yakınından geçti."}]
        out = seed_events_from_motion(events, [3.0, 6.0, 9.0])
        times = [item["time"] for item in out]
        self.assertGreaterEqual(len(times), 2)

    def test_clip_relative_time_lifts_to_original(self) -> None:
        events = [{"time": "00:10", "event": "Raf çöktü"}]
        out = lift_clip_relative_times(events, clip_start_s=24.0, peaks=[34.0])
        self.assertEqual(out[0]["time"], "00:34")

    def test_original_clock_is_not_shifted(self) -> None:
        events = [{"time": "00:34", "event": "Raf çöktü"}]
        out = lift_clip_relative_times(events, clip_start_s=24.0, peaks=[34.0])
        self.assertEqual(out[0]["time"], "00:34")

    def test_spec_risk_follows_category(self) -> None:
        spec = label_to_spec(
            {
                "category": "accident",
                "summary": "Çalışan düştü",
                "events": [{"time": "00:04", "event": "Çalışan düştü"}],
                "risk": "Düşük",
                "actions": ["Sağlık ekibini çağır"],
            }
        )
        self.assertEqual(spec["risk"], "Yüksek")


if __name__ == "__main__":
    unittest.main()
