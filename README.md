# KAIZEN_Teknofest26_DilAjanlar-_VideoAnalizKararSistemi

KAIZEN Takımı — TEKNOFEST 2026 Yapay Zeka Dil Ajanları Yarışması Finalist  
Senaryo 3: Video Analiz ve Karar Destek Sistemi

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

Yarışma sistemi için şartname **yerel vLLM** ister. Etiketleme ayrı iştir.

### Colab’de KPI (Mac yorulmasın)

**Kod = GitHub, veri = Drive.** Zip ile uğraşmaya gerek yok.

1. Drive’da bir kez yükleyin:
   - `MyDrive/KAIZEN_KPI/data/videos/{accident,near_miss,normal}/*.mp4`
   - `MyDrive/KAIZEN_KPI/data/exports/gold_labels_hepsi.json`
2. Colab’de GPU açın (Runtime → T4)
3. Defteri açın: [`notebooks/TEKNOFEST_KPI_Colab.ipynb`](notebooks/TEKNOFEST_KPI_Colab.ipynb)
4. Hücreleri sırayla çalıştırın → kod `git clone main`, sonuçlar Drive `exports/` altına düşer
5. Kod güncellenince: push → Colab’de sadece clone/pull hücresini tekrar çalıştır

`video: 0` görürsen gold vardır ama mp4’ler yanlış klasördedir; yol yukarıdaki gibi olmalı.

### Çıktılar

- `data/labels/<video>.json` — takım içi zengin etiket
- `data/labels/<video>_spec.json` — şartname mock JSON'u
