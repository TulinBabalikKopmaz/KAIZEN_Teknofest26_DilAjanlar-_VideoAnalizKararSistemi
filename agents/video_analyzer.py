"""Video Analyzer (Perception Agent) — VLM ile kritik an / ramak kala analizi."""

from __future__ import annotations

import os
import re
from typing import Any

import requests

from agents.state import AgentState
from utils.config import API_BASE_URL, MODEL_NAME
from utils.image import encode_image

VLM_SYSTEM_PROMPT: str = """\
Rolün: Sen alanında uzman bir İş Sağlığı ve Güvenliği (İSG) denetçisisin. \
Sana verilen karelerde kaza, ramak kala veya rutin çalışmayı ayırt edeceksin. \
Çıktın Türkçe olacak.

Kritik Kurallar:

1) METİNLERİ YOK SAY: Yeşil/mavi kutular, "ID#7", "car spd=5.0", "conf=0.8" gibi \
takip yazılarını görmezden gel. Raporda bunlardan bahsetme. Ancak Sistem Notu'ndaki \
zaman damgasını (MM:SS) kullan.

2) FİZİKSEL SAHNE: İnsanların ne yaptığına, kaçış var mı, yük/palet devriliyor mu, \
çarpışma, düşme, yanma, yerde hareketsiz kişi var mı ona bak.

3) RAMAK KALA: Yaralanma olmasa bile son anda kaçış veya tehlikeli yaklaşma ramak kaladır. \
Risk en az Orta, çok yakınsa Kritik.

4) UYDURMA: Görmediğin makine, uçan nesne veya kaza uydurma.

5) SAHTE ALARM: Net tehlike yoksa, işçiler rutin çalışıyorsa kaza zorlama. \
O zaman: 'Güvenli ortam, rutin çalışma' ve Risk: Güvenli.

6) GÖRÜNÜR KAZAYI KAÇIRMA: İlk kare sakin olsa bile sonraki karede çarpışma, devrilme, \
düşme, yangın/yanma veya yerde kişi varsa bu sahte alarm değildir. Risk: Kritik. \
Kararını sakin kareye göre değil, olayın olduğu kareye göre ver.

7) ZAMAN: Olay zamanını Sistem Notu'ndan al. Çıktıda Zaman alanını MUTLAKA MM:SS yaz \
(örnek: 00:15). Başka format yasak.

8) ÖZET: Durum Açıklaması MAKSİMUM 1-2 cümle ve en fazla 15 kelime olsun. \
Dümdüz rapor dili; yorum yok. Örnek: 'Çalışan elinde yük varken yüksekten düştü.'

Çıktı Formatı (TEK SATIR):
Zaman: [MM:SS] | Durum Açıklaması: [en fazla 15 kelime] | Risk: [Güvenli, Düşük, Orta, Kritik] | Aksiyon: [acil önlem]
"""


def _extract_timestamp(image_path: str) -> str:
    """Dosya adından zaman damgası çıkarır (örn. frame_00_14.jpg / 00-14.jpg → 00:14)."""
    try:
        filename = os.path.basename(image_path)
        stem = os.path.splitext(filename)[0]
        # KPI kareleri: 00-15.jpg
        m = re.fullmatch(r"(\d{2})-(\d{2})", stem)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
        parts = stem.replace(".jpg", "").split("_")
        return f"{parts[-2]}:{parts[-1]}"
    except (IndexError, ValueError):
        return "Bilinmiyor"


def _mmss_to_seconds(mmss: str) -> int | None:
    try:
        minutes, seconds = mmss.split(":")
        return int(minutes) * 60 + int(seconds)
    except Exception:
        return None


def _build_user_prompt(trigger_reason: str, frame_notes: list[str]) -> str:
    """Tetik sebebini ve kare zaman notlarını VLM kullanıcı mesajına gömer."""
    reason = trigger_reason.strip() or "Dinamik olay tetiklendi"
    notes = "\n".join(frame_notes) if frame_notes else ""
    return (
        f"Tetiklenme sebebi (Wake-Up sensörü): {reason}\n\n"
        f"{notes}\n\n"
        "Bu kareler bir tetik penceresinden alınmıştır; ilk kare sakin görünebilir.\n"
        "Üzerindeki bounding box, ID, spd, conf yazılarını yok say.\n"
        "Sistem Notu'ndaki zamanı kullan: çıktıda Zaman: MM:SS yaz (ör. 00:15).\n"
        "Durum Açıklaması en fazla 15 kelime, 1-2 kısa cümle, düz rapor dili; yorum yapma.\n"
        "Tüm karelere bak: kaçış, yük/palet devrilmesi, çarpışma, düşme, yanma, ramak kala.\n"
        "Görünür kaza varsa küçümseme. Net tehlike yoksa kaza uydurma.\n"
        "Türkçe, tek satır:\n"
        "Zaman: MM:SS | Durum Açıklaması: ... | Risk: Güvenli/Düşük/Orta/Kritik | Aksiyon: ..."
    )


def _is_critical_finding(model_cikti: str) -> bool:
    """VLM çıktısında tehlike / ramak kala / yüksek risk var mı?"""
    upper = model_cikti.upper()
    if any(
        key in upper
        for key in (
            "TEHLİKE",
            "TEHLIKE",
            "RAMAK KALA",
            "NEAR MISS",
            "KRİTİK",
            "KRITIK",
            "ORTA",
        )
    ):
        # "Risk: Güvenli" içinde yanlış pozitif olmasın diye Risk satırını kontrol et
        if "RISK:" in upper or "RİSK:" in upper:
            risk_part = upper.split("RISK:")[-1] if "RISK:" in upper else upper.split("RİSK:")[-1]
            risk_token = risk_part.split("|")[0].strip()
            if risk_token.startswith("GÜVENLİ") or risk_token.startswith("GUVENLI"):
                return False
            if risk_token.startswith("DÜŞÜK") or risk_token.startswith("DUSUK"):
                return False
            return True
        return "GÜVENLİ" not in upper and "GUVENLI" not in upper
    return False


def video_analyzer_tool(state: AgentState) -> dict[str, Any]:
    """
    Keyframe'leri VLM ile analiz eder; trigger_reason bağlamını kullanır.

    State alanları:
        keyframes: analiz edilecek görüntü yolları
        trigger_reason: wake-up tetik sebebi (ani hareket, aşırı hız vb.)
    """
    print("\n--- [1] Video Analyzer (VLM) Çalışıyor ---")

    endpoint = f"{API_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    trigger_reason = state.get("trigger_reason", "") or ""
    if trigger_reason:
        print(f"Tetik sebebi: {trigger_reason}")

    all_frames = state.get("keyframes", [])
    if not all_frames:
        return {
            "analysis_result": {
                "timestamp": "00:00",
                "event": "Klasörde incelenecek görüntü bulunamadı.",
            }
        }

    print(f"Klasörde {len(all_frames)} adet kare bulundu. Analiz başlıyor...")

    window_size = 3

    for i in range(0, len(all_frames), window_size):
        window_frames = all_frames[i : i + window_size]
        current_ts = _extract_timestamp(window_frames[-1])
        frame_notes: list[str] = []
        for img_path in window_frames:
            ts = _extract_timestamp(img_path)
            sec = _mmss_to_seconds(ts)
            if sec is None:
                frame_notes.append(
                    f"[Sistem Notu: Bu görsel zaman damgası {ts} olan bir kritik andan alınmıştır.]"
                )
            else:
                frame_notes.append(
                    f"[Sistem Notu: Bu görsel videonun {sec}. saniyesinden alınmıştır. Zaman: {ts}]"
                )

        user_prompt = _build_user_prompt(trigger_reason, frame_notes)

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

        content_list.append({"type": "text", "text": user_prompt})

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {"role": "user", "content": content_list},
            ],
            "max_tokens": 180,
            "temperature": 0.05,
        }

        max_retries = 3
        model_cikti = (
            f"Zaman: {current_ts} | Durum Açıklaması: Olağan çalışma | "
            "Risk: Güvenli | Aksiyon: İzlemeye devam et"
        )

        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=45)
                response.raise_for_status()
                model_cikti = response.json()["choices"][0]["message"]["content"].strip()
                model_cikti = " ".join(model_cikti.split())
                break
            except Exception as e:
                print(f"API Hatası (Deneme {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("Sunucuya ulaşılamadı, bu kesit atlanıyor...")

        if _is_critical_finding(model_cikti):
            print(f"!!! KRİTİK / RAMAK KALA TESPİT EDİLDİ (Zaman: {current_ts}) !!!")
            print(f"Modelin Çıkarımı: {model_cikti}")
            return {"analysis_result": {"timestamp": current_ts, "event": model_cikti}}

        print(f"-> {current_ts} anı temiz. Taramaya devam ediliyor...")
        print(f"   VLM: {model_cikti}")

    print("Tüm kareler tarandı, belirgin anormallik bulunamadı.")
    return {
        "analysis_result": {
            "timestamp": "00:00",
            "event": (
                "Zaman: 00:00 | Durum Açıklaması: Güvenli ortam | "
                "Risk: Güvenli | Aksiyon: İzlemeye devam et"
            ),
        }
    }
