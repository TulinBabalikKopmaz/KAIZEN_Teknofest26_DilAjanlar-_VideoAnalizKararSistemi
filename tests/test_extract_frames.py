"""Kare seçiminin küçük kontrolleri. Model / video dosyası gerektirmez."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_frames import pick_times, seconds_to_mmss, target_frame_count
from auto_label_qwen import parse_json


class ExtractFrameTests(unittest.TestCase):
    def test_short_clip_gets_dense_frames(self) -> None:
        self.assertEqual(target_frame_count(5.0, 6), 6)
        self.assertEqual(target_frame_count(60.0, 6), 6)

    def test_short_clip_covers_each_second(self) -> None:
        stamps = [seconds_to_mmss(t) for t in pick_times(4.63, 6, [(2.1, 40.0)])]
        self.assertIn("00:02", stamps)
        self.assertGreaterEqual(len(stamps), 5)
        self.assertEqual(stamps[0], "00:00")

    def test_long_clip_keeps_motion_peak(self) -> None:
        times = pick_times(30.0, 6, [(12.4, 50.0), (1.0, 2.0)])
        nearest = min(times, key=lambda t: abs(t - 12.4))
        self.assertLess(abs(nearest - 12.0), 1.0)
        self.assertLessEqual(times[0], 0.2)
        self.assertGreaterEqual(times[-1], 29.0)

    def test_unique_mmss_stamps(self) -> None:
        stamps = [seconds_to_mmss(t) for t in pick_times(5.0, 6, [(0.2, 10.0), (0.4, 12.0)])]
        self.assertEqual(len(stamps), len(set(stamps)))
        self.assertIn("00:03", stamps)

    def test_window_limits_frames_to_wake_up_range(self) -> None:
        times = pick_times(120.0, 6, [(12.0, 50.0), (95.0, 60.0)], window=(90.0, 110.0))
        self.assertTrue(all(89.9 <= t <= 110.1 for t in times), times)
        self.assertLessEqual(len(times), 6)
        nearest = min(times, key=lambda t: abs(t - 95.0))
        self.assertLess(abs(nearest - 95.0), 1.5)

    def test_repairs_truncated_json(self) -> None:
        parsed = parse_json(
            '{\n  "category": "accident",\n  "summary": "Kişi düştü",\n  "risk": "Yüksek",\n  "actions": ["Sağlık çağır"],\n  "events": [{"time": "00:02", "event": "düştü", "severity": "düşük'
        )
        self.assertEqual(parsed["risk"], "Yüksek")
        self.assertTrue(parsed["events"])


if __name__ == "__main__":
    unittest.main()
