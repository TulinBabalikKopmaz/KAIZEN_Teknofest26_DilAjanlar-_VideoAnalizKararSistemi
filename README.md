# TEKNOFEST 2026 — Video etiketleme

Yapay Zeka Dil Ajanları, 3. senaryo (video analiz + karar destek).

Şartname sistemden şunu ister: videoyu analiz et, olayları **zaman damgasıyla** yaz, Türkçe özet ve aksiyon üret, JSON ver. Gold etiket de aynı formata bakmalı.

```json
{
  "summary": "Videoda forklift kazası ve yaralanma riski gözlenmiştir.",
  "events": [
    {"time": "00:15", "event": "Forklift devrildi"},
    {"time": "00:20", "event": "Yerde hareketsiz kişi"}
  ],
  "risk": "Yüksek",
  "actions": ["Sağlık ekibini çağır", "Alanı güvenlik altına al"]
}
```

## Ne etiketlenecek, ne etiketlenmeyecek

Her saniyeyi yazmayın. 70 video × 30 sn × 1 satır = gereksiz iş ve şartname bunu istemiyor.

Her video için:

1. `category`: `normal` / `near_miss` / `accident`
2. `summary`: 1-2 cümle Türkçe
3. `events`: sadece olay değişen anlar (`MM:SS`)
4. `risk`: Düşük / Orta / Yüksek
5. `actions`: operatöre emir cümleleri

Qwen taslak üretir, insan onaylar. Gold etiket insan onayı olmadan KPI ölçmek için kullanılmaz.

## Kurulum

```bash
git clone https://github.com/TulinBabalikKopmaz/KAIZEN_Teknofest26_DilAjanlar-_VideoAnalizKararSistemi.git
cd KAIZEN_Teknofest26_DilAjanlar-_VideoAnalizKararSistemi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Videoları koyun:

```
data/videos/normal/
data/videos/near_miss/
data/videos/accident/
```

## 1) Elle etiketleme (Qwen yokken de çalışır)

```bash
streamlit run app/review_app.py
```

Videoyu izleyin, özet / olay / risk / aksiyon yazın, durumu `gold` yapıp kaydedin.

Yerel makine `0.0.0.0:8501` dinler (`.streamlit/config.toml`). Başka evden erişim için bu Mac açık kalmalı; tünel URL'si şifre gibidir — yalnızca takım arkadaşına verin. Videolar bu Mac'te kalır.

```bash
# Streamlit zaten çalışıyorsa atlayın
streamlit run app/review_app.py

# Ücretsiz tünel (npx gerekir)
npx --yes localtunnel --port 8501
```

`enableCORS = false` tünel için gerekli: public HTTPS origin localhost ile eşleşmez, aksi halde Streamlit websocket'i keser.

## 2) Qwen ile taslak (önerilen)

Mac'te en kolay yol Ollama:

```bash
ollama pull qwen2.5vl
python scripts/auto_label_qwen.py --backend ollama --every-sec 2 --max-frames 12
streamlit run app/review_app.py
```

Yarışma sistemi için şartname **yerel vLLM** ister. Etiketleme ayrı iş; gold veriyi şimdi Ollama ile üretebilirsiniz. Final pipeline sonra vLLM + Qwen2.5-VL olur.

vLLM örneği:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 16384
python scripts/auto_label_qwen.py --backend openai --every-sec 2 --max-frames 16
```

## Çıktılar

- `data/labels/<video>.json` — takım içi zengin etiket
- `data/labels/<video>_spec.json` — şartname mock JSON'u (değerlendirme / demo)

## KPI (şartname zorunlu)

Gold etiketler hazır olunca sistem çıktısı bunlarla kıyaslanır:

- olay tespit doğruluğu (zaman ±2 sn tolerans + metin benzerliği)
- kritik olay yakalama oranı (kaza / near miss kaçırıldı mı)
- özet kalitesi (insan 1-5)
- aksiyon önerisi tutarlılığı
- işlem süresi
