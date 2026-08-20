"""Wake-up katmanı: tepe seçimi, uzun video penceresi, odak kareleri."""

from __future__ import annotations

import unittest

from utils.demo_pipeline import _pick_focus_frames, wake_window
from utils.scene_evidence import SceneEvidence, top_motion_peaks


class TopPeaksTests(unittest.TestCase):
    def test_peaks_are_spread_apart(self) -> None:
        motion = [(float(t), 1.0) for t in range(30)]
        motion[5] = (5.0, 40.0)
        motion[6] = (6.0, 38.0)  # aynı olayın devamı, ayrı tepe sayılmamalı
        motion[20] = (20.0, 30.0)
        peaks = top_motion_peaks(motion, count=3, min_gap=2.0)
        self.assertIn(5.0, peaks)
        self.assertIn(20.0, peaks)
        self.assertNotIn(6.0, peaks)

    def test_sorted_by_time_not_score(self) -> None:
        motion = [(1.0, 5.0), (10.0, 50.0), (20.0, 30.0)]
        self.assertEqual(top_motion_peaks(motion, count=3, min_gap=2.0), [1.0, 10.0, 20.0])

    def test_zero_scores_ignored(self) -> None:
        self.assertEqual(top_motion_peaks([(1.0, 0.0), (2.0, 0.0)]), [])


class WakeWindowTests(unittest.TestCase):
    def test_short_video_has_no_window(self) -> None:
        evidence = SceneEvidence(motion_peak_sec=5.0, motion_peaks=[5.0])
        self.assertIsNone(wake_window(evidence, 30.0))

    def test_window_covers_all_peaks(self) -> None:
        evidence = SceneEvidence(motion_peak_sec=40.0, motion_peaks=[30.0, 40.0, 50.0])
        start, end = wake_window(evidence, 120.0)
        self.assertLessEqual(start, 30.0)
        self.assertGreaterEqual(end, 50.0)

    def test_window_span_is_capped(self) -> None:
        evidence = SceneEvidence(motion_peak_sec=10.0, motion_peaks=[10.0, 300.0])
        start, end = wake_window(evidence, 400.0)
        self.assertLessEqual(end - start, 45.0 + 0.01)

    def test_falls_back_to_single_peak(self) -> None:
        evidence = SceneEvidence(motion_peak_sec=80.0, motion_peaks=[])
        start, end = wake_window(evidence, 200.0)
        self.assertLessEqual(start, 80.0)
        self.assertGreaterEqual(end, 80.0)

    def test_no_motion_uses_second_half(self) -> None:
        start, end = wake_window(SceneEvidence(), 200.0)
        self.assertEqual((start, end), (100.0, 200.0))


class FocusFrameTests(unittest.TestCase):
    def _frames(self, times: list[float]) -> list[dict]:
        return [{"t_sec": t, "time": f"00:{int(t):02d}"} for t in times]

    def test_focus_uses_all_peaks(self) -> None:
        frames = self._frames([0, 2, 4, 6, 8, 10])
        evidence = SceneEvidence(motion_peak_sec=2.0, motion_peaks=[2.0, 10.0])
        picked = [f["t_sec"] for f in _pick_focus_frames(frames, evidence)]
        self.assertIn(2.0, picked)
        self.assertIn(10.0, picked)

    def test_few_frames_returned_as_is(self) -> None:
        frames = self._frames([0, 1])
        self.assertEqual(_pick_focus_frames(frames, SceneEvidence()), frames)

    def test_no_peak_falls_back_to_last_frames(self) -> None:
        frames = self._frames([0, 1, 2, 3, 4])
        picked = [f["t_sec"] for f in _pick_focus_frames(frames, SceneEvidence())]
        self.assertEqual(picked, [2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
