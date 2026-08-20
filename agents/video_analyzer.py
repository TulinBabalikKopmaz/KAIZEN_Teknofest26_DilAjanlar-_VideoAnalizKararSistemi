"""Video Analyzer (Perception Agent) — VLM ile kritik an / ramak kala analizi."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from agents.state import AgentState
from utils.model_client import ModelCallError, chat_vlm

WINDOW_SIZE: int = 3
SAFE_LINE: str = "Durum Açıklaması: Olağan çalışma | Risk: Güvenli | Aksiyon: İzlemeye devam et"

VLM_SYSTEM_PROMPT: str = """\
Rolün: Sen alanında uzman bir İş Sağlığı ve Güvenliği (İSG) denetçisisin. \
Sana verilen karelerde kaza, ramak kala veya rutin çalışmayı ayırt edeceksin. \
Çıktın Türkçe olacak.

Kritik Kurallar:

1) METİNLERİ YOK SAY: Yeşil/mavi kutular, "ID#7", "car spd=5.0", "conf=0.8" gibi \
takip yazılarını görmezden gel. Raporda bunlardan bahsetme.

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

Çıktı Formatı (TEK SATIR):
Durum Açıklaması: [kısa Türkçe özet] | Risk: [Güvenli, Düşük, Orta, Kritik] | Aksiyon: [acil önlem]
"""


def _extract_timestamp(image_path: str) -> str:
    """Dosya adından zaman damgası çıkarır (örn. frame_00_14.jpg → 00:14)."""
    try:
        filename = os.path.basename(image_path)
        parts = filename.replace(".jpg", "").split("_")
        return f"{parts[-2]}:{parts[-1]}"
    except (IndexError, ValueError):
        return "Bilinmiyor"


def _build_user_prompt(trigger_reason: str, user_prompt: str = "") -> str:
    """Tetik sebebini ve (varsa) jürinin sorusunu VLM kullanıcı mesajına gömer."""
    reason = trigger_reason.strip() or "Dinamik olay tetiklendi"
    question = user_prompt.strip()
    question_block = (
        f"Operatörün / jürinin sorusu (cevabın bunu karşılamalı): {question}\n\n" if question else ""
    )
    return (
        f"{question_block}"
        f"Tetiklenme sebebi (Wake-Up sensörü): {reason}\n\n"
        "Bu kareler bir tetik penceresinden alınmıştır; ilk kare sakin görünebilir.\n"
        "Üzerindeki bounding box, ID, spd, conf yazılarını yok say.\n"
        "Tüm karelere bak: kaçış, yük/palet devrilmesi, çarpışma, düşme, yanma, ramak kala.\n"
        "Görünür kaza varsa küçümseme. Fiili düşme/çarpışma/çökme near_miss değildir.\n"
        "Net tehlike yoksa kaza uydurma.\n"
        "Türkçe, tek satır:\n"
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


async def _analyze_window(frames: list[str], user_prompt: str, trigger_reason: str) -> tuple[str, str]:
    """Tek pencereyi VLM'e sorar; (zaman damgası, tek satır çıktı) döner."""
    timestamp = _extract_timestamp(frames[-1])
    try:
        result = await chat_vlm(
            _build_user_prompt(trigger_reason, user_prompt),
            frames,
            system=VLM_SYSTEM_PROMPT,
            temperature=0.05,
            max_tokens=220,
        )
        return timestamp, " ".join(result.text.split())
    except ModelCallError as exc:
        print(f"[{timestamp}] VLM çağrısı başarısız, bu kesit atlanıyor: {exc}")
        return timestamp, SAFE_LINE


async def video_analyzer_tool(state: AgentState) -> dict[str, Any]:
    """
    Keyframe'leri VLM ile analiz eder; trigger_reason ve jüri sorusunu kullanır.

    Pencereler eşzamanlı sorulur (süre bütçesi), sonuç zaman sırasına göre
    değerlendirilir: ilk kritik pencere raporlanır.
    """
    print("\n--- [1] Video Analyzer (VLM) Çalışıyor ---")

    trigger_reason = state.get("trigger_reason", "") or ""
    user_prompt = state.get("user_prompt", "") or ""
    if trigger_reason:
        print(f"Tetik sebebi: {trigger_reason}")
    if user_prompt:
        print(f"Jüri sorusu: {user_prompt}")

    all_frames = state.get("keyframes", [])
    if not all_frames:
        return {
            "analysis_result": {
                "timestamp": "00:00",
                "event": "Klasörde incelenecek görüntü bulunamadı.",
            }
        }

    windows = [all_frames[i : i + WINDOW_SIZE] for i in range(0, len(all_frames), WINDOW_SIZE)]
    print(f"{len(all_frames)} kare, {len(windows)} pencere eşzamanlı taranıyor...")

    outputs = await asyncio.gather(
        *(_analyze_window(window, user_prompt, trigger_reason) for window in windows)
    )

    for timestamp, model_cikti in outputs:
        if _is_critical_finding(model_cikti):
            print(f"!!! KRİTİK / RAMAK KALA TESPİT EDİLDİ (Zaman: {timestamp}) !!!")
            print(f"Modelin Çıkarımı: {model_cikti}")
            return {"analysis_result": {"timestamp": timestamp, "event": model_cikti}}
        print(f"-> {timestamp} anı temiz. VLM: {model_cikti}")

    print("Tüm kareler tarandı, belirgin anormallik bulunamadı.")
    return {
        "analysis_result": {
            "timestamp": "Video Sonu",
            "event": "Durum Açıklaması: Güvenli ortam | Risk: Güvenli | Aksiyon: İzlemeye devam et",
        }
    }


async def second_look_tool(state: AgentState) -> dict[str, Any]:
    """LangGraph ikinci bakış: tetik var, ilk tarama sakin dedi."""
    print("\n--- [1b] Video Analyzer ikinci bakış ---")
    extra: AgentState = {
        **state,
        "trigger_reason": (
            (state.get("trigger_reason") or "")
            + " | İKİNCİ BAKIŞ: önceki tarama sakin göründü; "
            "çarpışma, düşme, çökme, yanma varsa küçümseme."
        ).strip(" |"),
    }
    result = await video_analyzer_tool(extra)
    result["second_look_done"] = True
    return result
