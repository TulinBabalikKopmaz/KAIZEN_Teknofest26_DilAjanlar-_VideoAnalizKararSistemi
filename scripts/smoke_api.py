#!/usr/bin/env python3
"""Model endpoint doğrulama: tek kare VLM + tek metin LLM pingi.

Endpoint bilgileri geldiği gün ilk çalıştırılacak script:

    python scripts/smoke_api.py --provider teknofest
    python scripts/smoke_api.py --image data/frames/.../00-03.jpg

Çıkış kodu 0 = her iki model cevap verdi.
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
    """Kare verilmezse basit bir test görüntüsü üretir."""
    import cv2
    import numpy as np

    img = np.full((360, 640, 3), 60, dtype=np.uint8)
    cv2.rectangle(img, (120, 120), (200, 300), (200, 200, 200), -1)  # kişi benzeri
    cv2.rectangle(img, (360, 200), (560, 320), (40, 120, 220), -1)  # araç benzeri
    cv2.putText(img, "SMOKE TEST 00:03", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    dest = Path(tempfile.gettempdir()) / "isg_smoke_frame.jpg"
    cv2.imwrite(str(dest), img)
    return dest


def _report(result: ChatResult) -> None:
    print(f"  sağlayıcı : {result.provider}")
    print(f"  model     : {result.model}")
    print(f"  gecikme   : {result.latency_s:.2f} sn (deneme: {result.attempts})")
    preview = " ".join(result.text.split())[:200]
    print(f"  yanıt     : {preview or '(boş)'}")


async def run(role_filter: str, image: Path) -> int:
    failures = 0
    roles = ("vlm", "llm") if role_filter == "both" else (role_filter,)
    for role in roles:
        ep = config.endpoint(role)
        print(f"\n[{role.upper()}] {ep.chat_url}")
        try:
            _report(await ping(role, image if role == "vlm" else None))
            print("  durum     : OK")
        except ModelCallError as exc:
            failures += 1
            print(f"  durum     : BAŞARISIZ -> {exc}")
    return failures


def main() -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=config.PROVIDERS,
        default="",
        help="Env'deki PROVIDER'ı geçici olarak ezer",
    )
    parser.add_argument("--role", choices=("both", "vlm", "llm"), default="both")
    parser.add_argument("--image", type=Path, default=None, help="VLM pingi için kare")
    args = parser.parse_args()

    if args.provider:
        os.environ["PROVIDER"] = args.provider

    image = args.image or _synthetic_frame()
    if not image.exists():
        raise SystemExit(f"Kare bulunamadı: {image}")

    print(config.describe())
    print(f"timeout={config.request_timeout():.0f}s  retry={config.max_retries()}  kare={image}")

    failures = asyncio.run(run(args.role, image))
    if failures:
        raise SystemExit(f"\n{failures} model yanıt vermedi. .env ayarlarını kontrol edin.")
    print("\nTüm modeller yanıt verdi.")


if __name__ == "__main__":
    main()
