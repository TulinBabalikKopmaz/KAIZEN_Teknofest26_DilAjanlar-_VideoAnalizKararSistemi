"""Görüntü yardımcı fonksiyonları."""

from __future__ import annotations

import base64
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Görüntü dosyasını base64 string'e çevirir."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_b64(path: str | Path, max_side: int | None = None) -> str:
    """Kareyi base64'e çevirir; max_side verilirse önce küçültür.

    Küçültme token ve süre tasarrufu için; büyük CCTV kareleri 768 px'e indiğinde
    VLM doğruluğu belirgin düşmüyor ama çağrı gözle görülür hızlanıyor.
    """
    data = Path(path).read_bytes()
    if max_side and max_side > 0:
        data = _resize_jpeg(data, max_side)
    return base64.b64encode(data).decode("ascii")


def _resize_jpeg(data: bytes, max_side: int) -> bytes:
    """cv2 ile uzun kenarı max_side'a indirir; başarısızsa orijinali döner."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return data

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return data
    height, width = img.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return data
    resized = cv2.resize(img, (int(width * scale), int(height * scale)))
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else data
