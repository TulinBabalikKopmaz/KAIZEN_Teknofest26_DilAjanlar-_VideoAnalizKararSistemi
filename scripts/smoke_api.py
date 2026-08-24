#!/usr/bin/env python3
"""Model endpoint doğrulama: LLM ping + (EVREN'de) kısa mp4 VLM pingi.

    python scripts/smoke_api.py --provider teknofest --role llm
    python scripts/smoke_api.py --provider teknofest --role vlm --video data/videos/.../clip.mp4

Çıkış kodu 0 = seçilen roller cevap verdi. API anahtarı yazdırılmaz.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import config  # noqa: E402
from utils.demo_pipeline import use_utf8_stdout  # noqa: E402
from utils.model_client import ChatResult, ModelCallError, ping  # noqa: E402


def _synthetic_frame() -> Path:
    import cv2
    import numpy as np

    img = np.full((360, 640, 3), 60, dtype=np.uint8)
    cv2.rectangle(img, (120, 120), (200, 300), (200, 200, 200), -1)
    cv2.rectangle(img, (360, 200), (560, 320), (40, 120, 220), -1)
    cv2.putText(img, "SMOKE TEST 00:03", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    dest = Path(tempfile.gettempdir()) / "isg_smoke_frame.jpg"
    cv2.imwrite(str(dest), img)
    return dest


def _find_smoke_video() -> Path | None:
    root = ROOT / "data" / "videos"
    if not root.exists():
        return None
    videos = sorted(root.rglob("*.mp4"), key=lambda p: p.stat().st_size)
    return videos[0] if videos else None


def _report(result: ChatResult) -> None:
    print(f"  provider : {result.provider}")
    print(f"  model    : {result.model}")
    print(f"  latency  : {result.latency_s:.2f}s (attempts: {result.attempts})")
    preview = " ".join(result.text.split())[:200]
    print(f"  reply    : {preview or '(empty)'}")


async def run(
    role_filter: str,
    image: Path | None,
    video: Path | None,
) -> int:
    failures = 0
    roles = ("vlm", "llm") if role_filter == "both" else (role_filter,)
    for role in roles:
        ep = config.endpoint(role)
        print(f"\n[{role.upper()}] {ep.chat_url}")
        try:
            if role == "vlm":
                _report(await ping("vlm", image if not video else None, video))
            else:
                _report(await ping("llm"))
            print("  status   : OK")
        except ModelCallError as exc:
            failures += 1
            print(f"  status   : FAIL -> {exc}")
    return failures


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=config.PROVIDERS, default="")
    parser.add_argument("--role", choices=("both", "vlm", "llm"), default="both")
    parser.add_argument("--image", type=Path, default=None, help="Local/Ollama VLM JPEG")
    parser.add_argument("--video", type=Path, default=None, help="EVREN vlm: short mp4")
    args = parser.parse_args()

    if args.provider:
        os.environ["PROVIDER"] = args.provider

    video = args.video
    image = args.image
    need_vlm = args.role in {"both", "vlm"}
    if need_vlm and config.provider() == "teknofest":
        video = video or _find_smoke_video()
        if video is None or not video.exists():
            raise SystemExit(
                "EVREN vlm JPEG kabul etmez. --video ile kısa bir mp4 verin "
                "(data/videos altında da aranır)."
            )
        print(f"VLM video={video.name} ({video.stat().st_size / 1e6:.1f} MB)")
        image = None
    elif need_vlm:
        image = image or _synthetic_frame()
        if not image.exists():
            raise SystemExit(f"Frame not found: {image}")

    print(config.describe())
    print(
        f"timeout={config.request_timeout():.0f}s  retry={config.max_retries()}  "
        f"vlm_media={'video' if video else ('image' if image else '-')}"
    )

    failures = asyncio.run(run(args.role, image, video))
    if failures:
        raise SystemExit(f"\n{failures} model(s) did not answer. Check .env (do not paste the key).")
    print("\nAll selected models answered.")


if __name__ == "__main__":
    main()
