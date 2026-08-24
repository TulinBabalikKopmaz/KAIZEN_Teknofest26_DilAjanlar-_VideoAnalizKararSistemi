# KAIZEN_Teknofest26_DilAjanlar-_VideoAnalizKararSistemi

KAIZEN Takımı — TEKNOFEST 2026 Yapay Zeka Dil Ajanları Yarışması Finalist  
Senaryo 3: Video Analiz ve Karar Destek Sistemi

## Model ayarı (ortak API)

Yarışmada VLM ve LLM ortak EVREN API üzerinden geliyor (alias: `vlm` + `llm-fast`).
Kodda hiçbir adres sabit değil; her şey `.env` üzerinden okunur.

```bash
cp .env.example .env      # PROVIDER=teknofest ve TEKNOFEST_BASE_URL / API_KEY doldurun
python scripts/smoke_api.py --provider teknofest    # endpoint doğrulama (VLM + LLM ping)
```

- `PROVIDER=teknofest` — yarışmanın ortak API'si
- `PROVIDER=ollama` — yerel geliştirme ve demo yedeği (`FALLBACK_PROVIDER=ollama` ile otomatik düşer)
- `PROVIDER=mock` — model olmadan pipeline denemesi

## Canlı demo (jürinin videosu + jürinin promptu)

```bash
streamlit run app/demo_app.py     # demoda kullanılacak ekran
```

Terminalden aynı işi yapan çekirdek:

```bash
python scripts/analyze_video.py --video demo.mp4 \
    --prompt "Bu videoda iş kazası var mı, kaçıncı saniyede?"
```

Çıktı `data/demo_runs/<ad>/` altına yazılır: `result.json`, `spec.json`, `report.txt`, `frames/`.
Süre sıkışırsa `--fast` (veya `.env` içinde `DEMO_FAST_MODE=1`): YOLO kanıtı, ikinci bakış ve
mevzuat referansı kapanır. 60 saniyeden uzun videolarda kareler hareket (wake-up) tepesinin
çevresinden seçilir.

Sunum öncesi prova:

```bash
python scripts/demo_rehearsal.py --videos data/videos --limit 3
# 3 prompt varyantı x 3 video, süre/uyarı tablosu + kontrol listesi
```

Jürinin dosyası ne formatta gelirse gelsin boru hattı ayakta kalmalı; model
gerektirmeyen giriş kontrolü:

```bash
python scripts/check_video_inputs.py --synthetic          # dikey / 4K / avi / mov / 3 dk varyantları
python scripts/check_video_inputs.py --videos demo.mov   # jürinin dosyasını önceden dene
```

Sunum akışı, konuşma notları, demo koreografisi ve jüri soruları:
[`docs/sunum_akisi.md`](docs/sunum_akisi.md).

## Gerçek zamanlı akış (RTSP / webcam)

Aynı çekirdek canlı akışta da çalışır. Okuyucu ayrı iş parçacığında canlı kalır,
hareket (wake-up) tetiklenmesinde kareler kuyruğa girer, VLM çağrıları asenkron
tüketilir — model yavaşsa kare düşer, akış donmaz.

```bash
python tools/rtsp_stream.py --source rtsp://kullanici:sifre@10.0.0.12:554/stream1
python tools/rtsp_stream.py --source 0                       # webcam
python tools/rtsp_stream.py --source demo.mp4 --loop         # kamera olmadan prova
```

Uyarılar `data/stream_alerts/` altına yazılır; `--workers` eşzamanlı analiz
sayısını, `--cooldown` aynı olayın tekrar raporlanmasını sınırlar.

## Video etiketleme

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

Her saniyeyi yazmayın. Her video için:

1. `category`: `normal` / `near_miss` / `accident`
2. `summary`: 1-2 cümle Türkçe
3. `events`: sadece olay değişen anlar (`MM:SS`)
4. `risk`: Düşük / Orta / Yüksek
5. `actions`: operatöre emir cümleleri

Qwen taslak üretir, insan onaylar. Gold etiket insan onayı olmadan KPI ölçmek için kullanılmaz.

### Kurulum

```bash
git clone https://github.com/TulinBabalikKopmaz/KAIZEN_Teknofest26_DilAjanlar-_VideoAnalizKararSistemi.git
cd KAIZEN_Teknofest26_DilAjanlar-_VideoAnalizKararSistemi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Videoları koyun (git'e yüklenmez):

```
data/videos/normal/
data/videos/near_miss/
data/videos/accident/
```

### Elle etiketleme

```bash
streamlit run app/review_app.py
```

Videoyu izleyin, özet / olay / risk / aksiyon yazın, durumu `gold` yapıp kaydedin.

### Qwen ile taslak

```bash
ollama pull qwen2.5vl
python scripts/auto_label_qwen.py --backend ollama --every-sec 2 --max-frames 4
streamlit run app/review_app.py
```

Etiketleme ayrı iştir: yarışma sistemi ortak API'yi (`--backend teknofest`) kullanır,
etiket taslağı için yerel Ollama yeterli.

### Colab’de KPI (Mac yorulmasın)

**Kod = GitHub, veri = Drive.** Zip ile uğraşmaya gerek yok.

Deney branch: **`mustafa`** (baseline için `main`).

1. Drive’da bir kez yükleyin:
   - `MyDrive/KAIZEN_KPI/data/videos/{accident,near_miss,normal}/*.mp4`
   - `MyDrive/KAIZEN_KPI/data/exports/gold_labels_hepsi.json`
2. Colab’de GPU açın (Runtime → T4)
3. Defteri açın: [`notebooks/TEKNOFEST_KPI_Colab.ipynb`](notebooks/TEKNOFEST_KPI_Colab.ipynb) — `BRANCH = 'mustafa'`
4. Hücreleri sırayla çalıştırın

Smoke (18 video, eski davranış):

```bash
python scripts/run_kpi_wide.py --n 18 --seed 42 --model qwen2.5vl:7b --no-second-look
```

Tüm gold:

```bash
python scripts/run_kpi_wide.py --n all --seed 42 --model qwen2.5vl:7b --no-second-look
```

Yarışmanın ortak API'si ile (GPU'ya gerek yok, ölçüm demo ile aynı modelde):

```bash
python scripts/run_kpi_wide.py --backend teknofest --n all --split holdout --tag holdout
```

Kural katmanının katkısını ölçmek için ablasyon (aynı tahminler, `refine_label` kapalı):

```bash
python scripts/run_kpi_wide.py --backend teknofest --n all --no-refine --tag norefine
```

Sunuma gidecek tek tablo:

```bash
python scripts/kpi_summary_table.py
# data/exports/kpi_final_ozet.csv + hedef karşılaştırması (risk %70, olay %50, kritik %90 ...)
```

Skor düştüğünde nedenini tahmin etmiyoruz, ölçüyoruz:

```bash
python scripts/diagnose_kpi.py --gold data/exports/kpi_wide_7b_gold.json --pred data/predictions_wide
# her gold olayı için: eslesti / metin_kacti / zaman_kacti / olay_yok + data/exports/kpi_teshis.csv
```

18 videoluk ölçümde kaçırılan gold olaylarının %74'ü zaman olarak doğru yakalanmış,
yalnızca ifade farkından eşleşmiyor (zaman kaçırma %5). Bu yüzden
`prompts/video_label_prompt.txt` içinde İSG terminoloji sözlüğü var: "işçi" değil
**çalışan**, "önünden geçti" değil **çok yakınından geçti**.

`video: 0` görürsen gold vardır ama mp4’ler yanlış klasördedir; yol yukarıdaki gibi olmalı.

### Çıktılar

- `data/labels/<video>.json` — takım içi zengin etiket
- `data/labels/<video>_spec.json` — şartname mock JSON'u
- `data/demo_runs/<ad>/result.json` — demo koşusu (cevap + spec + süreler + kanıt kareleri)
- `data/exports/kpi_final_ozet.csv` — sunumdaki KPI tablosu
