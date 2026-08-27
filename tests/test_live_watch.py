"""Canlı izleme: wake-up ve EVREN klibi. Jüri pipeline'ına dokunmaz.

    python -m unittest tests.test_live_watch
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.live_watch import (
    Incident,
    LiveHub,
    LiveStatus,
    MotionWakeUp,
    StreamConfig,
    attach_live_support,
    _incident_prompt,
    encode_jpeg,
    lock_live_label,
    looks_like_jpeg,
    pick_event_time,
    select_even_frames,
)
from utils.video_clip import frames_to_clip


class LiveWatchTests(unittest.TestCase):
    def test_motion_wakeup_fires_on_large_frame_change(self) -> None:
        wake = MotionWakeUp(8.0)
        dark = np.zeros((90, 160, 3), dtype=np.uint8)
        bright = np.full((90, 160, 3), 255, dtype=np.uint8)
        self.assertEqual(wake.score(dark), 0.0)
        score = wake.score(bright)
        self.assertGreater(score, 8.0)
        self.assertTrue(wake.triggered(score))

    def test_frames_to_clip_writes_mp4(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            paths = []
            for index in range(3):
                path = folder / f"f{index}.jpg"
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, :] = (index * 40, 20, 80)
                cv2.imwrite(str(path), frame)
                paths.append(path)
            dest = folder / "clip.mp4"
            out = frames_to_clip(paths, dest, fps=6.0, hold=2)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_status_snapshot_uses_watch_banner(self) -> None:
        hub = LiveHub()
        hub.status = LiveStatus(phase="candidate", motion_score=22.0)
        snap = hub.snapshot()
        self.assertEqual(snap["phase"], "candidate")
        self.assertIn("Aday", snap["banner"]["kicker"] + snap["banner"]["title"])
        self.assertNotIn("Risk: Düşük", str(snap["banner"]))
        self.assertNotIn("accident", str(snap["banner"]))

    def test_preview_jpeg_is_complete_bytes_not_a_partial_file(self) -> None:
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[:, :] = (40, 80, 120)
        data = encode_jpeg(frame, quality=80)
        self.assertTrue(looks_like_jpeg(data))
        self.assertFalse(looks_like_jpeg(data[:40]))
        hub = LiveHub()
        hub.set_preview_jpeg(data[:40])
        self.assertEqual(hub.latest_jpeg(), b"")
        hub.set_preview_jpeg(data)
        self.assertEqual(hub.latest_jpeg()[:2], b"\xff\xd8")

    def test_live_window_covers_seconds_after_trigger(self) -> None:
        cfg = StreamConfig()
        self.assertGreaterEqual(cfg.clip_frames - cfg.pre_frames, 8)
        self.assertEqual(cfg.vlm_frames, 12)
        after_s = (cfg.clip_frames - cfg.pre_frames) / cfg.sample_fps
        self.assertGreaterEqual(after_s, 1.8)

    def test_select_even_frames_keeps_span(self) -> None:
        rows = [{"time": f"00:{i:02d}", "path": str(i)} for i in range(16)]
        picked = select_even_frames(rows, 12)
        self.assertEqual(len(picked), 12)
        self.assertEqual(picked[0]["time"], "00:00")
        self.assertEqual(picked[-1]["time"], "00:15")

    def test_live_prompt_asks_for_outcome_not_first_frame(self) -> None:
        incident = Incident(trigger_t=7.2, motion_score=20.0, frames=[
            {"time": "00:06"},
            {"time": "00:07"},
            {"time": "00:09"},
        ])
        text = _incident_prompt(incident)
        self.assertIn("SON KARE", text)
        self.assertIn("00:07", text)
        self.assertNotIn("video_label_prompt", text)

    def test_live_lock_keeps_accident_without_yolo_dampen(self) -> None:
        locked = lock_live_label(
            {
                "category": "accident",
                "risk": "Yüksek",
                "summary": "Çalışan yüksekten düştü ve yerde hareketsiz.",
                "events": [{"time": "00:08", "event": "Çalışan yere düştü"}],
                "actions": ["Sağlık ekibini çağır"],
            }
        )
        self.assertEqual(locked["category"], "accident")
        self.assertEqual(locked["risk"], "Yüksek")
        upgraded = lock_live_label(
            {
                "category": "normal",
                "risk": "Düşük",
                "summary": "Çalışan yüksekten düştü.",
                "events": [{"time": "00:08", "event": "Çalışan yere düştü"}],
                "actions": [],
            }
        )
        self.assertEqual(upgraded["category"], "accident")
        self.assertEqual(upgraded["risk"], "Yüksek")

    def test_live_support_fills_event_time_and_law_note(self) -> None:
        spec = {
            "summary": "Çalışan yüksekten düştü ve yerde hareketsiz.",
            "events": [
                {"time": "00:06", "event": "Çalışan merdivende"},
                {"time": "00:12", "event": "Çalışan yere düştü"},
            ],
            "risk": "Yüksek",
            "actions": ["Sağlık ekibini çağır", "Alanı güvenlik altına al"],
        }
        self.assertEqual(pick_event_time(spec, "00:12"), "00:12")
        self.assertEqual(pick_event_time(spec, "00:09"), "00:12")
        support = attach_live_support(spec, "00:12")
        self.assertEqual(support["event_time"], "00:12")
        self.assertIn("Mevzuat", support["law_note"])
        self.assertIn("Madde", support["law_detail"])
        hub = LiveHub()
        hub.mark_decided(
            {
                "spec": spec,
                "label": {"category": "accident", "risk": "Yüksek"},
                "trigger_time": "00:12",
                "event_time": support["event_time"],
                "law_note": support["law_note"],
                "law_detail": support["law_detail"],
                "latency_s": 12.0,
                "provider": "teknofest",
            }
        )
        snap = hub.snapshot()
        self.assertEqual(snap["event_time"], "00:12")
        self.assertTrue(snap["law_note"])
        self.assertNotIn("accident", snap["banner"]["title"])
        self.assertIn("İş kazası", snap["banner"]["title"])


if __name__ == "__main__":
    unittest.main()
