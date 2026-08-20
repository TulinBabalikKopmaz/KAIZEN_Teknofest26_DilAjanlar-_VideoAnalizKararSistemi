# Etiketleme Kılavuzu

Gold etiket bizim cevap anahtarımız. Kişiye göre değişirse ölçüm gürültülü olur ve
"iyileştirdik mi kötüleştirdik mi" sorusunu cevaplayamayız. Bu kılavuz kategori
sınırlarını, belirsizlik işaretini ve kullanacağımız dili sabitler.

Arayüz: `streamlit run app/review_app.py` → "Gold etiketleme".

---

## 1. Kategori: ne oldu?

Kategori **olayın gerçek sonucuna** göre seçilir. Görüntünün ne kadar net olduğu
kategoriyi etkilemez, onu bir sonraki bölümde ayrıca işaretliyoruz.

| Kategori | Tanım | Ayırt edici soru |
|---|---|---|
| `accident` | Fiili temas veya zarar var: çarpma, düşme, devrilme, yükün üstüne düşmesi, yanma, yaralanma. | Bir şey **gerçekten oldu** mu? |
| `near_miss` | Temas olmadı ama bir adım kalmıştı: forklift çalışanın çok yakınından geçti, yük çalışanın yanına düştü, çalışan son anda kaçtı. | Şans veya son saniye tepkisi olmasa kaza olur muydu? |
| `normal` | Rutin iş akışı. Riskli görünen ama olağan işlemler (kaynak kıvılcımı, makine dumanı, hızlı ama kontrollü hareket) buraya girer. | Sahada İSG uzmanı müdahale eder miydi? Hayırsa normal. |

Sık karışan durumlar:

- Çalışan düştü ama kendi kendine kalkıp devam etti → `accident` (temas ve düşme var).
- Yük sallandı, kimse yakınında değil → `normal`. Yakınında çalışan varsa `near_miss`.
- İki araç çarpıştı, kimse yaralanmadı → `accident` (araç hasarı da kazadır).
- Çalışan koruyucu ekipman takmıyor ama olay yok → `normal`, notta belirt.

## 2. Belirsizlik: sahne okunabiliyor mu?

Bu alan kategoriden bağımsızdır ve **kategoriyi değiştirmez**. Sorduğu şey:
"Bu videoyu başka bir makul insan farklı yorumlayabilir mi?"

- `net` — ne olduğu tartışmasız.
- `belirsiz` — makul bir insan farklı yorumlayabilir.

Şartname test kümesinin yaklaşık **%20'sinin belirsiz** olmasını istiyor. Bunu
ayrı video toplayarak değil, mevcut videoları bu gözle işaretleyerek karşılıyoruz.

Gerekçe seçenekleri ve ne zaman kullanılır:

| Gerekçe | Durum |
|---|---|
| `gorus_engeli` | Olay kısmen bir nesnenin arkasında veya açı kötü. |
| `cerceve_disi` | Olayın bir kısmı kare dışında oluyor. |
| `niyet_belirsiz` | Rutin mi olay mı belli değil. Örnek: yerdeki çalışan bakım mı yapıyor, düştü mü? |
| `alarm_gorunumlu_normal` | Normal ama alarm verdirebilir: kaynak kıvılcımı, duman, buhar, hızlı forklift. |
| `kalite` | Bulanık, karanlık, düşük çözünürlük veya düşük fps. |
| `sonuc_gorunmuyor` | Kamera dönüyor veya kesme var, olayın sonucu görünmüyor. |
| `diger` | Yukarıdakilere girmiyor; notta açıklayın. |

Hızlandırma: `python scripts/suggest_ambiguity.py --apply --top 16` adayları taslak
olarak işaretler (`ambiguity_source=auto`). Arayüzde "Sadece belirsizlik onayı
bekleyenler" filtresiyle hızlıca onaylayın veya `net`'e çevirin. Otomatik öneri
karar değil, sıralama aracıdır.

## 3. Dil: aynı olayı aynı kelimelerle yaz

Ölçümde gold ile sistem çıktısı kelime örtüşmesine göre eşleşiyor. Daha önemlisi,
İSG raporu tutarlı terminoloji ister. Prompt'a da aynı sözlüğü verdik.

**Kullan:**

- Kişi: **çalışan** (işçi, adam, personel, sürücü değil)
- Araç: **forklift**, **kamyon**, **iş makinesi**, **motosiklet**
- Tehlikeli yaklaşma: **çok yakınından geçti**, **son anda kaçtı**
- Kaza: **yere düştü**, **yüksekten düştü**, **çarptı**, **devrildi**,
  **altında kaldı**, **yük düştü**, **kıyafeti tutuştu**, **yerde hareketsiz**

**Kullanma:**

- "tehlikesi var", "olabilir", "riski mevcut" gibi tahmin dili — ne olduğunu yaz.
- "Sistem normal durumda" gibi içi boş cümle — ne gördüğünü yaz.
- İngilizce kelime, emoji, uzun hikâye.

## 4. Olay satırları

- Her saniyeyi yazmayın; **olayın değiştiği anı** yazın. Video başına 1–3 olay yeter.
- Zaman `MM:SS` ve oynatıcıdaki süreye göre. Olayın **başladığı** an yazılır.
- Aynı olayı iki satıra bölmeyin. Sistem tarafında tekrarları birleştiren bir katman
  var (`utils.label_json.dedupe_events`) ama gold temiz olmalı.
- Normal videolarda da bir olay satırı yazın (ör. `00:00 — Çalışanlar fabrikada yürüyor.`);
  boş olay listesi ölçümü bozar.

## 5. Özet, risk, aksiyon

- **Özet:** 1–2 kısa cümle, ne olduğunu söyleyen. "Çalışan yüksekten yere düştü ve
  hareketsiz kaldı." gibi.
- **Risk:** normal → `Düşük`, ramak kala → `Orta`, kaza → `Yüksek`. İstisna olursa
  notta gerekçelendirin.
- **Aksiyon:** 1–3 madde, sahada uygulanabilir olsun ("Sağlık ekibini çağır",
  "Alanı güvenlik altına al"). Kaza videosunda ilk aksiyon müdahale olmalı.

## 6. Kalite kuralları

- **Test kümesi iki kişi tarafından bağımsız etiketlenir.** Uyuşmazlıkları konuşup
  çözün; hangi videoda anlaşamadığınızı not edin (sunumda etiketçi uyumu olarak geçer).
- Emin değilseniz **Taslak** kaydedin, `Gold` demeyin. Gold = cevap anahtarı.
- Etiketlediğiniz videoyu prompt ayarı için kullandıysanız o video **dev** kümesindedir;
  test kümesine dokunmuyoruz (`scripts/make_splits.py`).
- Etiket bittiğinde: `python scripts/export_gold_labels.py --name gold_labels_hepsi`
  ardından `python scripts/make_splits.py`.

## 7. Yeni video eklerken

1. Videoyu kategorisine göre `data/videos/<accident|near_miss|normal>/` altına koy.
2. Taslak etiketi modelle üret (varsa): `python scripts/auto_label_qwen.py --video ...`
3. `review_app` ile taslağı düzelt, belirsizlik alanını işaretle, gold onayı ver.
4. Export ve split komutlarını tekrar çalıştır.

Sıfırdan yazmak yerine taslağı düzeltmek etiket başına süreyi belirgin düşürür;
karar yine insanda kalır.
