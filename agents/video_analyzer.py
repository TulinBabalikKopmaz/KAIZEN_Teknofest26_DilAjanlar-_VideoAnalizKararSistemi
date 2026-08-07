"""Video Analyzer (Perception Agent) — VLM ile kritik an / ramak kala analizi."""

from __future__ import annotations

import os
from typing import Any

import requests

from agents.state import AgentState
from utils.config import API_BASE_URL, MODEL_NAME
from utils.image import encode_image

VLM_SYSTEM_PROMPT: str = """\
Rolün: Sen alanında uzman, çok tecrübeli bir İş Sağlığı ve Güvenliği (İSG) denetçisisin. \
Amacın sana verilen fotoğraftaki kaza veya "Ramak Kala" (Near Miss) durumlarını tespit etmektir.

Kritik Kurallar (BUNLARA KESİNLİKLE UY):

1) METİNLERİ YOK SAY: Fotoğrafın üzerinde yeşil/mavi kutular ve "ID#7", "car spd=5.0", \
"conf=0.8" gibi teknik takip yazıları görebilirsin. Bunları KESİNLİKLE görmezden gel. \
Raporunda bu ID'lerden veya metinlerden ASLA bahsetme.

2) FİZİKSEL ORTAMA ODAKLAN: İnsanların ne yaptığına, bir şeyden kaçıp kaçmadıklarına, \
devrilen veya düşen devasa yükler/paletler olup olmadığına odaklan.

3) RAMAK KALA (NEAR MISS): İnsanlar yaralanmamış olsa bile, devrilen bir yükten son anda \
kaçmışlarsa bu bir 'Ramak Kala' olayıdır ve hayati tehlike (Kritik Risk) olarak raporlanmalıdır.

4) HALÜSİNASYON YAPMA: Sadece fotoğrafta net olarak gördüğün fiziksel eylemleri mantıklı \
bir dille açıkla. Uçan arabalar veya var olmayan makineler uydurma.

5) SAHTE ALARM KAÇIŞ RAMPASI (ÖNEMLİ): Sana gönderilen fotoğrafların çoğu sensörlerin \
yanılması sonucu oluşan SAHTE ALARMLAR olabilir. Görevini yapmak için zorla kaza bulmaya \
çalışma. Eğer işçiler normal şekilde çalışıyorsa, kimse kaçmıyorsa veya devrilen bir şey \
YOKSA; KESİNLİKLE hikaye uydurma. Sadece 'Güvenli ortam, rutin çalışma yapılıyor' de ve \
Risk seviyesini 'Güvenli' olarak işaretle.

6) BAĞLAM: Görsel sensör tetiklemesiyle gelmiş olabilir; yine de kanıt yoksa Güvenli de. \
Statik karede net tehlike yoksa abartma.

Çıktı Formatı (TEK SATIR, başka format kullanma):
Durum Açıklaması: [Olayın net, kısa ve profesyonel özeti] | Risk: [Güvenli, Düşük, Orta, Kritik] | Aksiyon: [İSG kanununa göre alınması gereken acil önlem]
"""


def _extract_timestamp(image_path: str) -> str:
    """Dosya adından zaman damgası çıkarır (örn. frame_00_14.jpg → 00:14)."""
    try:
        filename = os.path.basename(image_path)
        parts = filename.replace(".jpg", "").split("_")
        return f"{parts[-2]}:{parts[-1]}"
    except (IndexError, ValueError):
        return "Bilinmiyor"


def _build_user_prompt(trigger_reason: str) -> str:
    """Tetik sebebini VLM kullanıcı mesajına gömer."""
    reason = trigger_reason.strip() or "Dinamik olay tetiklendi"
    return (
        f"Tetiklenme sebebi (Wake-Up sensörü): {reason}\n\n"
        "Bu fotoğraf/kare(ler) kritik bir andan alınmıştır.\n"
        "Üzerindeki bounding box, ID, spd, conf yazılarını tamamen yok say.\n"
        "Sadece fiziksel sahneyi incele: kaçış, yük/palet devrilmesi, çarpışma, ramak kala.\n"
        "Tek satırda şu formatta yanıt ver:\n"
        "Durum Açıklaması: ... | Risk: Güvenli/Düşük/Orta/Kritik | Aksiyon: ..."
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
    user_prompt = _build_user_prompt(trigger_reason)

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

        content_list.append({"type": "text", "text": user_prompt})

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {"role": "user", "content": content_list},
            ],
            "max_tokens": 220,
            "temperature": 0.05,
        }

        max_retries = 3
        model_cikti = (
            "Durum Açıklaması: Olağan çalışma | Risk: Güvenli | Aksiyon: İzlemeye devam et"
        )

        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=45)
                response.raise_for_status()
                model_cikti = response.json()["choices"][0]["message"]["content"].strip()
                # Model bazen ID/spd metinlerine kayabiliyor; tekrar uyarı eklemeden temizle
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
            "timestamp": "Video Sonu",
            "event": "Durum Açıklaması: Güvenli ortam | Risk: Güvenli | Aksiyon: İzlemeye devam et",
        }
    }
