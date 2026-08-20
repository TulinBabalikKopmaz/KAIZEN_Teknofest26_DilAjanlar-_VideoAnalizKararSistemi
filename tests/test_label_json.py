"""Olay listesi temizliği ve JSON ayrıştırma kontrolleri."""

from __future__ import annotations

import unittest

from utils.label_json import dedupe_events, label_to_spec, snap_events_to_frame_times


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


if __name__ == "__main__":
    unittest.main()
