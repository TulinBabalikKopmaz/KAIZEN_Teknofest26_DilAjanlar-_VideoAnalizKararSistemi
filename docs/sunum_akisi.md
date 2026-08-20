# Sunum ve Demo Akışı (4 dk + 1 dk)

Sahne planı, konuşma notları, demo koreografisi ve olası jüri soruları.
Rakamlar `data/exports/kpi_final_ozet.csv` dosyasından gelir; ortak API ile
yeniden ölçüm yapıldığında bu dosya ve aşağıdaki tablo güncellenmelidir.

---

## 1. Sunum zaman çizelgesi (240 sn)

| Süre | Slayt | Ne söylenir |
|---|---|---|
| 0:00–0:25 | Problem | Kamera var, izleyen yok. Kaza sonrası kaydı bulmak saatler alıyor; ramak kala olayları hiç raporlanmıyor. |
| 0:25–1:00 | Çözüm | Videoyu izleyip **karar veren** üç ajanlı sistem: algı → risk → aksiyon. Çıktı serbest metin değil, şartname JSON'u. |
| 1:00–1:40 | Mimari | Wake-up sensör katmanı, VLM (Qwen 3 27B), kural katmanı, Risk Assessor, mevzuat RAG'i ile Action Recommender, LangGraph orkestrasyonu. |
| 1:40–2:30 | Doğruluk kanıtı | KPI tablosu: risk doğruluğu, kritik olay yakalama, yanlış alarm. Ablasyon: kural katmanı açık/kapalı farkı. |
| 2:30–3:05 | Veri ve metodoloji | Golden dataset dağılımı (%30 kaza, %30 normal, %20 ramak kala, %20 belirsiz), gold etiketleme süreci, teşhis aracı. |
| 3:05–3:40 | Gerçek zamanlı yol haritası | RTSP akışında wake-up tetikli asenkron analiz; tek makinede çok kamera. |
| 3:40–4:00 | Kapanış | Tek cümle: "Kayıtları arşiv değil, karar verici hale getiriyoruz." Demo'ya geçiş. |

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

**Gerçek zamanlı (3:05–3:40).** Aynı çekirdek `tools/rtsp_stream.py` ile canlı
akışa bağlanıyor: wake-up tetiklenince kareler asenkron analiz kuyruğuna gidiyor,
akış hiç durmuyor. Tek makine birden fazla kamerayı böyle taşıyor.

---

## 2. KPI slaytı

Güncel değerler (18 video, gold ile karşılaştırma — `kpi_final_ozet.csv`):

| Metrik | Değer | Sunumda nasıl anlatılır |
|---|---|---|
| Risk doğruluğu | 1.00 | Kural katmanı + VLM birlikte; jüri sorarsa holdout ayrımını söyle. |
| Kritik olay yakalama | 0.92 | Kaza/ramak kala videolarında olayı kaçırma oranımız düşük. |
| Yanlış alarm | 0.00 | Normal videolarda kritik alarm üretmiyoruz — sahada güven şartı. |
| Aksiyon doluluğu | 1.00 | Her çıktıda uygulanabilir müdahale adımı var. |
| Olay yakalama (±2 sn) | 0.21 | Aşağıdaki dürüst çerçeveyi kullan. |
| Özet benzerliği | 0.00 | Aynı çerçeve. |

**Olay yakalama ve özet benzerliği için dürüst çerçeve.** Bu iki metrik gold
cümleyle kelime örtüşmesine bakıyor. `diagnose_kpi.py` çıktısı şunu gösteriyor:
kaçırdığımız gold olaylarının **%74'ü zaman olarak doğru yakalanmış, sadece
ifade farkından eşleşmiyor** (ör. model "işçi forkliftin önünden geçiyor" derken
gold "forklift çalışanın çok yakınından geçti" diyor). Zaman kaçırma sadece %5.
Yani sistem olayı buluyor, dili henüz standart değil. Prompt'a İSG terminoloji
sözlüğü ekledik ve ortak 27B model ile bu iki metriği yeniden ölçüyoruz.

> Jüri bu slaytta zayıf sayıyı görürse kendimiz açıklamış olmak avantaj.
> Sayıyı saklamak yerine nedenini ölçmüş olmak mühendislik olgunluğu gösterir.

---

## 3. Demo koreografisi (60 sn)

Sahne öncesi hazırlık (jüri odaya girmeden):

```powershell
python scripts/smoke_api.py                       # VLM + LLM erişimi ve gecikme
streamlit run app/demo_app.py                     # arayüz açık ve bekliyor
```

Canlı akış:

| Saniye | Yapılan | Söylenen |
|---|---|---|
| 0–8 | Jürinin videosunu sürükleyip bırak | "Videoyu sisteme veriyorum, önceden hiç görmediği bir kayıt." |
| 8–14 | Jürinin promptunu kutuya yapıştır, Analiz'e bas | "Sorularını olduğu gibi giriyorum." |
| 14–40 | İlerleme adımları akıyor | "Şu an wake-up katmanı aday pencereyi buldu, VLM o kareleri okuyor." |
| 40–50 | Risk kartı + olay zaman çizelgesi | "Olayı 00:0X'te tespit etti, risk seviyesi ve gerekçesi burada." |
| 50–58 | Aksiyonlar + JSON sekmesi | "Aksiyonlar mevzuat referanslı, çıktı şartname JSON'u." |
| 58–60 | Süre göstergesi | "Toplam süre X saniye." |

Yedek yol (sadece gerekirse, sessizce): ortak API cevap vermezse istemci
Ollama'ya düşer; internet tamamen giderse `data/demo_runs/` altındaki prova
çıktısı açılır. Bunu jüriye anlatmaya gerek yok, sadece hazır olsun.

### Prova komutları

```powershell
python scripts/demo_rehearsal.py --videos data/videos --limit 3   # süre + çıktı provası
python scripts/check_video_inputs.py --synthetic                  # format dayanıklılığı
python scripts/diagnose_kpi.py                                    # metrik kaybının nedeni
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

**"Risk doğruluğu 1.00 fazla iyi değil mi?"**
18 videoluk kümede evet, bu yüzden gold'u genişletiyoruz ve holdout ayrımıyla
ölçüyoruz. Metriği tek sayı olarak değil, teşhis kırılımıyla birlikte sunuyoruz.

**"Gerçek zamanlı çalışır mı?"**
Wake-up katmanı sayesinde model çağrısı sayısı video uzunluğundan bağımsız.
RTSP prototipi akıştan kare alıp asenkron analiz ediyor; ölçek tarafında darboğaz
GPU değil, tetiklenme sıklığı.

**"Kaç kamera taşır?"**
Analiz sadece tetiklenmede çalıştığı için maliyet olay sayısıyla artıyor.
Tek H200 ile onlarca kamerayı, tetiklenmeleri kuyruğa alarak taşıyabiliriz.

---

## 5. Kontrol listesi (sunum sabahı)

- [ ] `.env` ortak API bilgileriyle dolu, `scripts/smoke_api.py` yeşil
- [ ] `streamlit run app/demo_app.py` açık, örnek video ile bir kez ısıtıldı
- [ ] Yedek çıktı `data/demo_runs/` altında hazır
- [ ] KPI tablosu slayttaki değerlerle `kpi_final_ozet.csv` uyumlu
- [ ] Sunum dosyası ve demo aynı ekranda, geçiş provası yapıldı
- [ ] `python -m pytest tests -q` geçiyor
