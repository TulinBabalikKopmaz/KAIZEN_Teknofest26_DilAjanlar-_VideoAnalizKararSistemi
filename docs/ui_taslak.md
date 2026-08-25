# UI taslağı (demo ekranı)

Jüriye gösterilen yüzey `app/demo_app.py` (Streamlit). Çekirdek `utils/demo_pipeline.py`;
kopya `utils/display.py`. Etiketleme aracı `app/review_app.py` jüri sahnesi değil.

Çalıştırma:

```bash
streamlit run app/demo_app.py
```

Tema: `.streamlit/config.toml` (koyu zemin, altın vurgu `#C9A227`).

---

## Ne değişmez (kilit)

- Şartname JSON: `summary`, `events[{time,event}]`, `risk` ∈ {Düşük, Orta, Yüksek}, `actions`.
- İç kategori anahtarları: `normal` | `near_miss` | `accident`. Klasör adları ve KPI aynı kalır.
- Pipeline / kural katmanı / test split: dokunma. UI sadece gösterir.
- Sahne klibi ekip kararı; merdiven klibini varsayılan seçme.

## Ekranda ne yazılır

| İç kod | Şartname `risk` | Ekran (saha durumu) | Ekran (karar) |
|---|---|---|---|
| `normal` | Düşük | Rutin operasyon | Kontrol altında |
| `near_miss` | Orta | Ramak kala | Yüksek dikkat |
| `accident` | Yüksek | İş kazası | Kritik durum |

Ham `accident` veya "Risk: Düşük" basma. Düşük token'ı JSON'da kalır; insana
"kontrol altında" dersin. Kaynak: `utils.display.verdict`.

---

## Mevcut iskelet (senin işin buradan)

1. Üst: KAIZEN markası + başlık + tek satır iddia.
2. Sol: video yükle / klasörden seç. Sağ: operatör sorusu + **Analiz et**.
3. Sonuç: dört metrik (saha durumu, karar, süre, olay sayısı) + karar kartı
   (durum · karar + alt başlık + düz Türkçe cevap).
4. Sol kolon: video + kanıt kareleri. Sağ: zaman çizelgesi, özet, aksiyonlar.
5. Expander: jüri JSON'u, süreler, sensör kanıtı. İndirme düğmeleri.
6. Kenar: sağlayıcı, hızlı mod, kare sayısı, RAG, sahne yedeği.

CSS sınıfı: `.kz-verdict.{ok|watch|critical}`. Ton `verdict(...)["tone"]`.

---

## İstediğimiz premium his

- Endüstriyel kontrol odası, dashboard oyunu değil. Az emoji, az gradient.
- Karar kartı tek bakışta okunur. Metrikler İngilizce kod göstermez.
- Jüri JSON'u her zaman bir tık uzakta (şeffaflık), ana hikâye değil.
- 1 dk sahnede: video seç → analiz (veya yedek) → kart + zaman + aksiyon.
- Mobil şart değil; 1280–1920 genişlik ve projektör kontrastı önemli.
- Streamlit "Made with" gizlendi; menüyü de sade tut.

İyileştirme fikirleri (zorunlu değil): zaman çizelgesini yatay bar, aksiyonları
numaralı emir listesi, kritikte kartı biraz daha baskın yap, yükleme sırasında
"wake-up → VLM → kural" üç adımlı durum. React'e geçme — süre yok; Streamlit'i
cilala.

---

## Dokunma

- `utils/risk_rules.py`, `prompts/`, `scripts/run_kpi_wide.py`, gold / split.
- Test kümesine bakıp eşiği oynatma.
- `.env` commit etme.

Takıldığın kopya: `utils/display.py`. Mimari özeti: `AGENTS.md`.
