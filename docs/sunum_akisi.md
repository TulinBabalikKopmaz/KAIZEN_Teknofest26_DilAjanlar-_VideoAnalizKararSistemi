# Sunum ve Demo Akışı

Şartnamedeki **4 dk sunum + 1 dk demo**, sahneye çıkış süresidir (kendi slayt +
kendi gösterim). Jürinin sonradan vereceği video + prompt için resmi bir süre
söylenmedi. İkisini karıştırmayın: sunumda sahneyi bitirin; jüri kaydında
doğru JSON üretin.

---

## 0. İki ayrı iş

| | A — Sahne (4+1) | B — Jüri videosu + prompt |
|---|---|---|
| Ne | Kendi slayt, kendi örnek video | Onların kaydı, onların sorusu |
| Süre | 4 dk konuşma + 1 dk gösterim | Belirtilmedi; 2–4 dk analiz normal |
| Amaç | Anlatmak, ekranda karar göstermek | Sınıf, risk, zaman, aksiyon doğru olsun |
| Ayar | Hızlı mod açık olabilir; önceden koşulmuş / kayıtlı yedek | Hızlı mod **kapalı**, 8 kare, ikinci bakış + eleştirmen + RAG açık |
| Model | EVREN `vlm` + `llm-fast` (ısınmış); yedek ollama veya `demo_runs` | EVREN `vlm` + `llm-fast` |

Sunumdaki 1 dk’da jüri videosunun bitmesini beklemeyin. O slot’ta **sizin
seçtiğiniz** kısa klibi (veya önceden alınmış ekran kaydını) gösterin. Jüri
dosyası gelince Streamlit’te tam pipeline çalıştırın; beklerken mimariyi
anlatırsınız.

Cursor’da bu dosya + `AGENTS.md` yeterli bağlamdır. UI arkadaşı
`docs/ui_taslak.md`.

---

## 0.1 Nasıl konuşulur (jüri kulağı)

İç kod (`normal` / `near_miss` / `accident`) slaytta görünmesin. JSON’daki
`risk: Düşük` token’ını ekranda “risk düşük” diye okumayın.

| İç kod | Şartname | Söylenen |
|---|---|---|
| `normal` + Düşük | Düşük | Rutin operasyon, saha **kontrol altında** |
| `near_miss` + Orta | Orta | **Ramak kala** — kaza olmadı, **yüksek dikkat** |
| `accident` + Yüksek | Yüksek | **İş kazası** — **kritik durum**, derhal müdahale |

Belirsizlik dördüncü sınıf değil; sahnenin okunabilirlik bayrağı.

Kapanış cümlesi aynı: kayıtları arşiv değil karar verici hale getiriyoruz.

---

## 1. Sunum zaman çizelgesi (240 sn)

| Süre | Slayt | Ne söylenir |
|---|---|---|
| 0:00–0:25 | Problem | Kamera var, izleyen yok. Kaza sonrası kaydı bulmak saatler alıyor; ramak kala olayları hiç raporlanmıyor. |
| 0:25–1:00 | Çözüm | Videoyu izleyip **karar veren** üç ajanlı sistem: algı → risk → aksiyon. Çıktı serbest metin değil, şartname JSON'u. |
| 1:00–1:40 | Mimari | Wake-up, EVREN `vlm`, kural katmanı, Risk Assessor, mevzuat RAG, Action Recommender, LangGraph. |
| 1:40–2:30 | Doğruluk kanıtı | KPI tablosu: risk doğruluğu, kritik olay yakalama, yanlış alarm. Ablasyon: kural katmanı açık/kapalı farkı. |
| 2:30–3:05 | Veri ve metodoloji | Golden dataset dağılımı (%30 kaza, %30 normal, %20 ramak kala, %20 belirsiz), gold etiketleme süreci, teşhis aracı. |
| 3:05–3:40 | Gerçek zamanlı | İkinci sekme: canlı izleme (8502). Wake-up aday pencere; kart modelle gelir. |
| 3:40–4:00 | Kapanış | Tek cümle: "Kayıtları arşiv değil, karar verici hale getiriyoruz." Kendi demo klibine geçiş. |

### Konuşma notları

**Problem (0:00–0:25).** Sahada onlarca kamera var ama kayıtları kimse izlemiyor.
Kaza olduğunda İSG uzmanı saatlerce kayıt tarıyor. Ramak kala olaylar —
kazadan önceki en değerli sinyal — hiç kayda geçmiyor.

**Çözüm (0:25–1:00).** Biz videoyu etiketleyen bir sınıflandırıcı yapmadık;
olayı bulan, riski mevzuata göre seviyelendiren ve müdahale adımı üreten bir
karar sistemi yaptık. Çıktı sabit şemalı JSON: özet, zaman damgalı olaylar,
risk seviyesi, aksiyonlar. Bu yüzden çıktı doğrudan İSG kayıt sistemine yazılabilir.

**Mimari (1:00–1:40).** Üç noktayı vurgula:
1. **Wake-up katmanı** — her kareyi modele sormuyoruz. Hareket analizi ve YOLO
   ile aday pencereyi buluyoruz, VLM'i sadece oraya bakmaya zorluyoruz. 3 dakikalık
   kayıtta bile analiz süresi sabit kalıyor.
2. **Kural katmanı** — model çıktısını sensör kanıtıyla çapraz kontrol ediyoruz.
   Hareket yoksa "kaza" iddiası düşürülüyor; bu yanlış alarmı sıfıra indiren şey.
3. **Mevzuat RAG'i** — aksiyonlar modelin hayal gücü değil, İSG dokümanlarından
   getirilen bağlamla üretiliyor.

**Doğruluk (1:40–2:30).** Metrikleri şartname tanımıyla ver: olay yakalama ±2 sn
toleransla ölçülüyor. Kritik olay yakalama ve yanlış alarm oranını öne çıkar —
saha için asıl önemli olan bu iki sayı. Kural katmanı ablasyonunu göster
(`run_kpi_wide.py --no-refine`), böylece katkının ölçülmüş olduğu anlaşılır.

**Metodoloji (2:30–3:05).** Gold etiketleri ekip olarak elle doğruladık,
`app/review_app.py` ile. `scripts/diagnose_kpi.py` her kaybın nedenini
(zaman kaçtı / metin kaçtı / sahne yanlış) ayırıyor; iyileştirmeleri tahminle
değil ölçümle yaptık.

**Gerçek zamanlı (3:05–3:40).** Aynı çekirdek canlı kameraya bağlanır. Wake-up
yerelde aday pencereyi açar; EVREN o kısa klibe bakar, akış durmaz.
İsterseniz bu 35 sn’de ikinci sekmede `app/live_app.py` (port 8502) açık olsun:
oyuncak/hareket → ekranda **Aday pencere**. Kart (İş kazası · Kritik durum)
model dönünce gelir — 1 dk jüri demosuna bağlamayın. Yoksa slaytta aynı cümle yeter.

---

## 2. KPI slaytı

İç ölçüm (dev **46 video**, resmi EVREN `vlm` + `llm-fast`).
Sunumda "jüri skoru" demeyin. Test ayrı satır; ayar için kullanılmadı.

| Metrik | Değer | Sunumda nasıl anlatılır |
|---|---|---|
| Risk doğruluğu | %87 | Kural katmanı + VLM; ayar yalnız dev'de. |
| Kritik olay yakalama | %100 | Kaza/ramak videolarında olayı kaçırmıyoruz. |
| Yanlış alarm | %0 | Normal sahnede kritik alarm yok — saha güveni. |
| Aksiyon doluluğu | %100 | Her çıktıda uygulanabilir müdahale var. |
| Olay yakalama (±2 sn + metin) | %92 | Dil sıkıştırma + tepe tohumu; kalan kayıp sahne yanlış okuma. |
| Özet benzerliği | %83 | Aynı dürüst çerçeve. |

Tutulmuş **test** (31 video, aynı dondurulmuş pipeline, bir kez, üzerine kural yok):

| Metrik | Test |
|---|---|
| Risk doğruluğu | %74 |
| Olay yakalama | %73 |
| Kritik olay yakalama | %100 |
| Yanlış kaza (normal) | %0 |

Near miss testte zayıf (%14 risk): VLM tamamlanmış kaza yazınca indirmedik. Dev–test düşüşü beklenen; slaytta jüri skoru demeyin.

**Olay yakalama için dürüst çerçeve.** Jüri skoru değil; 46 videoluk EVREN
iç ölçüm. Kural katmanı **donduruldu** — kalan 4 miss VLM sahne okuması
(pres yükü, kontrolden çıkan forklift, kamyon indirme, proses dumanı).
Skorlayıcıya eşanlam veya test split ile şişirmedik.

> Jüri zayıf sayıyı görürse kendimiz açıklamış olmak avantaj.
> Sayıyı saklamak yerine nedenini ölçmüş olmak mühendislik olgunluğu gösterir.

---

## 3. Sahne demosu (1 dk, kendi videomuz)

Sahne klibi **ekip kararı**. Merdiven (`6AISVvob4C0_trim_7`) bir aday; kilitli değil.
Jüri odaya girmeden, seçilen klibi bir kez koşup `data/demo_runs/` yedeğini alın:

```powershell
python scripts/smoke_api.py
streamlit run app/demo_app.py
py -m streamlit run app/live_app.py --server.port 8502
python scripts/analyze_video.py --video <SECILEN_KLIP> --fast --run-name sahne_yedek
```

Canlı şov (4 dk içindeki 35 sn, 1 dk demoya bağlanmaz): odaya girmeden 8502’de
**İzlemeyi başlat**. 3:05’te sekmeyi göster, hareket/oyuncak → **Aday pencere**.
Kartı beklemeyin; “aynı çekirdek kameraya bağlı, operatörü kaza anında uyarır”
deyin. VLM o sırada dönerse bonus.

Son başarılı koşu `data/demo_runs/` altında dursun (kayıt yedek).

Canlı 60 sn (önceden bildiğiniz klip; hızlı mod açık):

| Saniye | Yapılan | Söylenen |
|---|---|---|
| 0–8 | Kendi videoyu seç / bırak | "Sahadaki kaydı sisteme veriyorum." |
| 8–14 | Prompt, Analiz Et (veya önceden bitmiş sonucu aç) | "Operatör sorusunu olduğu gibi giriyorum." |
| 14–40 | İlerleme veya hazır sonuç | "Wake-up aday pencereyi buldu, VLM o karelere bakıyor." |
| 40–50 | Karar kartı + zaman çizelgesi | Durumu ve kararı oku; olay zamanını göster. Proses alevi/dumanı varsa **Zor sahne** satırını oku: görünen ateş kaza değil. |
| 50–58 | Aksiyonlar + JSON | "Çıktı şartname JSON'u, aksiyonlar uygulanabilir." |
| 58–60 | Süre metriği | "Bu gösterim X saniye." |

Yedek: API yoksa kenardan `ollama`; o da yoksa `data/demo_runs/.../report.txt`.

### Jüri videosu (süre serbest)

Hızlı mod **kapalı**, kare 8. Prompt’u olduğu gibi yapıştırın. Bitene kadar
konuşun. İkinci bakış ve eleştirmen 50 sn’ye takılmasın diye varsayılan analiz
bütçesi 10 dk. Doğru sınıf/risk, sahne süresinden önemli.

```powershell
python scripts/analyze_video.py --video JURI.mp4 --prompt "jürinin sorusu"
python scripts/demo_rehearsal.py --videos data/videos --limit 3
python scripts/check_video_inputs.py --synthetic
```

---

## 4. Jüri soruları ve cevapları

**"Bu YOLO ile de yapılabilirdi, VLM'e ne gerek var?"**
YOLO nesne söyler, olay söylemez. "Çalışan yerde" ile "çalışan yüksekten düştü ve
hareketsiz" arasındaki farkı ancak dil modeli kurar. Zaten YOLO'yu atmadık;
wake-up katmanında kanıt üreticisi olarak kullanıyoruz.

**"Model uydurursa (halüsinasyon) ne olur?"**
Kural katmanı model çıktısını sensör kanıtıyla karşılaştırıyor. Hareket kanıtı
olmayan kaza iddiası düşürülüyor; yanlış alarm oranımızın sıfır olması bunun sonucu.

**"Risk doğruluğu fazla iyi değil mi?"**
İç ölçüm 46 videoluk **dev**; test split'e bakmadık. Yanlış alarm %0 ve kritik
yakalama %100 saha için asıl sayı. Near miss risk ~%58: VLM tamamlanmış kaza
yazınca gold ramak olsa bile indirmiyoruz.

**"Gerçek zamanlı çalışır mı?"**
Wake-up katmanı sayesinde model çağrısı sayısı video uzunluğundan bağımsız.
RTSP prototipi akıştan kare alıp asenkron analiz ediyor; ölçek tarafında darboğaz
GPU değil, tetiklenme sıklığı.

**"Kaç kamera taşır?"**
Analiz sadece tetiklenmede çalıştığı için maliyet olay sayısıyla artıyor.
Tek H200 ile onlarca kamerayı, tetiklenmeleri kuyruğa alarak taşıyabiliriz.

---

## 5. Kontrol listesi (sunum sabahı)

- [ ] `.env` EVREN anahtarı dolu, `python scripts/smoke_api.py --provider teknofest` yeşil
- [ ] Sahne klibi ekipçe seçildi; 1 dk prova + `data/demo_runs/` yedek
- [ ] API düşünce Streamlit yedeği otomatik açıyor (en az bir `demo_runs` kaydı olsun)
- [ ] Jüri yolu: hızlı mod kapalı, 8 kare; Streamlit açık ve ısıtıldı
- [ ] KPI slaytı `docs/sunum_akisi.md` / `data/exports/kpi_final_ozet.csv` ile aynı (jüri skoru demeyin)
- [ ] Dil: rutin operasyon / ramak kala / iş kazası; "risk düşük" demeyin
- [ ] Sunum dosyası ve sahne demosu aynı ekranda, geçiş provası yapıldı
- [ ] `python -m pytest tests -q` geçiyor
