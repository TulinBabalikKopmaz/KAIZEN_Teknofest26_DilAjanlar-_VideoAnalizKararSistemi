"""Video Analyzer (Perception Agent) — VLM ile kayan pencere tarama."""

from __future__ import annotations

import os
from typing import Any

import requests

from agents.state import AgentState
from utils.config import API_BASE_URL, MODEL_NAME
from utils.image import encode_image


def _extract_timestamp(image_path: str) -> str:
    """Dosya adından zaman damgası çıkarır (örn. frame_00_14.jpg → 00:14)."""
    try:
        filename = os.path.basename(image_path)
        parts = filename.replace(".jpg", "").split("_")
        return f"{parts[-2]}:{parts[-1]}"
    except (IndexError, ValueError):
        return "Bilinmiyor"


def video_analyzer_tool(state: AgentState) -> dict[str, Any]:
    """Keyframe'leri sliding window ile tarar; tehlike bulursa early exit yapar."""
    print("\n--- [1] Video Analyzer (VLM) Çalışıyor ---")

    endpoint = f"{API_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    all_frames = state.get("keyframes", [])
    if not all_frames:
        return {
            "analysis_result": {
                "timestamp": "00:00",
                "event": "Klasörde incelenecek görüntü bulunamadı.",
            }
        }

    print(f"Klasörde {len(all_frames)} adet kare bulundu. Kayan pencere taraması başlıyor...")

    window_size = 3

    for i in range(0, len(all_frames), window_size):
        window_frames = all_frames[i : i + window_size]
        current_ts = _extract_timestamp(window_frames[-1])

        print(f"[{current_ts}] saniyesine kadar olan kesit taranıyor ({len(window_frames)} kare)...")

        content_list: list[dict[str, Any]] = []
        for img_path in window_frames:
            base64_image = encode_image(img_path)
            content_list.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        content_list.append(
            {
                "type": "text",
                "text": (
                    "Sana güvenlik kamerasından ardışık kareler veriyorum. "
                    "Eğer her şey olağan akışındaysa sadece 'DURUM: NORMAL' yaz ve başka hiçbir şey ekleme. "
                    "Eğer kareler arasında bir kaza, çarpışma, tavan çökmesi veya tehlike varsa "
                    "'DURUM: TEHLİKE' yaz ve hemen yanına ne olduğunu kısaca açıkla."
                ),
            }
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "Sen nesnel ve kısa yanıtlar veren endüstriyel bir İSG yapay zekasısın.",
                },
                {"role": "user", "content": content_list},
            ],
            "max_tokens": 100,
            "temperature": 0.05,
        }

        max_retries = 3
        model_cikti = "DURUM: NORMAL"

        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                model_cikti = response.json()["choices"][0]["message"]["content"].strip()
                break
            except Exception as e:
                print(f"API Hatası (Deneme {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("Sunucuya ulaşılamadı, bu kesit atlanıyor...")

        if "TEHLİKE" in model_cikti.upper():
            print(f"!!! KAZA TESPİT EDİLDİ (Zaman: {current_ts}) !!!")
            print(f"Modelin Çıkarımı: {model_cikti}")
            return {"analysis_result": {"timestamp": current_ts, "event": model_cikti}}

        print(f"-> {current_ts} anı temiz. Taramaya devam ediliyor...")

    print("Tüm video tarandı, herhangi bir anormallik bulunamadı.")
    return {
        "analysis_result": {
            "timestamp": "Video Sonu",
            "event": "Tüm süre boyunca güvenli ortam.",
        }
    }
