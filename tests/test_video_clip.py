"""EVREN kısa klip aralığı — ffmpeg / ağ yok."""

from __future__ import annotations

import unittest

from utils.config import ModelEndpoint
from utils.model_client import _openai_payload
from utils.video_clip import CLIP_MAX_S, clip_span


class ClipSpanTests(unittest.TestCase):
    def test_short_video_is_sent_whole(self) -> None:
        self.assertEqual(clip_span(30.0, 10.0, [10.0]), (0.0, 30.0))

    def test_long_video_pads_peak(self) -> None:
        start, end = clip_span(180.0, 60.0, [60.0])
        self.assertAlmostEqual(start, 50.0)
        self.assertAlmostEqual(end, 80.0)

    def test_caps_at_max_span(self) -> None:
        start, end = clip_span(180.0, 50.0, [40.0, 90.0])
        self.assertLessEqual(end - start, CLIP_MAX_S + 0.01)
        self.assertGreaterEqual(end - start, 2.0)


class VideoPayloadTests(unittest.TestCase):
    def test_video_b64_uses_video_url_not_images(self) -> None:
        ep = ModelEndpoint(
            provider="teknofest",
            role="vlm",
            base_url="https://example.com/v1",
            model="vlm",
        )
        payload = _openai_payload(ep, "izle", ["imagedata"], None, 0.1, 64, False, "AAAA")
        content = payload["messages"][0]["content"]
        types = [part["type"] for part in content]
        self.assertEqual(types, ["text", "video_url"])
        self.assertTrue(content[1]["video_url"]["url"].startswith("data:video/mp4;base64,"))


if __name__ == "__main__":
    unittest.main()
