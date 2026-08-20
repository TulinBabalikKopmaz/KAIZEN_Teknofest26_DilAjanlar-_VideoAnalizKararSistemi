# Video Toplama Briefi

Elimizde **77 video** var (27 kaza, 18 ramak kala, 31 normal, 76'sı gold etiketli).
Kaza tarafı doygun; oraya yeni video eklemenin ölçüme katkısı düşük. Aşağıdaki üç
boşluk kapanırsa 30 hedefli video, 100 rastgele videodan daha değerli olur.

**Belirsiz (ambiguous) kategorisi için video aramaya gerek yok.** Belirsizlik ayrı bir
sonuç sınıfı değil, sahnenin okunabilirliği; mevcut videoları o gözle işaretliyoruz
(bkz. `docs/etiketleme_kilavuzu.md`).

---

## Öncelik 1 — Zor "normal" videolar (hedef ~10)

Yanlış alarm oranımız şu an %0 ve bu en güçlü sayımız. Ama kolay normal videolarla
elde edildi. Sistemin **alarm verdirecek ama olağan** sahnelerde de sessiz kalması gerek.

Aranan sahneler:

- Kaynak / taşlama yapan çalışan — kıvılcım var, yangın yok
- Duman, buhar, toz bulutu olan olağan üretim hattı
- Yerde yatarak veya eğilerek bakım yapan çalışan (düşmüş gibi görünen)
- Kontrollü ama hızlı forklift manevrası, yakın ama güvenli geçiş
- Yük indirme/kaldırma işlemi sorunsuz tamamlanıyor
- Vardiya değişimi, kalabalık ama düzenli hareket

## Öncelik 2 — Ramak kala (hedef ~5-7)

Şartname bu kategoriyi ayrıca ölçüyor; 18'den 25'e çıkması ölçümü sağlamlaştırır.

Aranan sahneler:

- Yaya forklift yoluna giriyor, sürücü son anda duruyor
- Yük devriliyor, çalışan bir adım uzakta
- Çalışan merdivenden kayıyor ama tutunuyor
- İki araç kesişiyor, biri frene basıyor
- Asılı yük çalışanın üstünden geçiyor

## Öncelik 3 — Uzun kayıt (hedef 3-5)

Şu an sadece 2 videomuz 60 saniyeden uzun. Jürinin vereceği demo videosu uzun
olabilir ve uzun video yolumuz (wake-up penceresi) neredeyse hiç ölçülmedi.

Aranan: 2–10 dakikalık, olayın **bir yerde** olduğu sabit kamera kaydı. Olay anını
etiketlerken saniyesi önemli; kırpmayın, tam kaydı verin.

---

## Kalite şartları

- Sabit kamera (CCTV/güvenlik kamerası tercih edilir), el kamerası son seçenek
- En az 480p; 320p altını almayın (bulanık videolar zaten belirsiz sayılıyor)
- Üzerinde büyük logo, altyazı bandı, montaj efekti olmasın
- Aynı olayın farklı açılarını ayrı video saymayın
- Ses gerekmiyor
- Dosya adında Türkçe karakter ve emoji olmasın, uzantı `.mp4` olsun

## Nereye konur

```
data/videos/normal/      # zor normaller buraya
data/videos/near_miss/
data/videos/accident/
```

Ardından:

```powershell
python scripts/check_video_inputs.py --videos data/videos/normal   # okunabilirlik kontrolü
streamlit run app/review_app.py                                   # etiketleme + belirsizlik
python scripts/export_gold_labels.py --name gold_labels_hepsi
python scripts/make_splits.py
```

## Ne kadar yeter

30 hedefli video ile:

- normal 41 videoya çıkar → yanlış alarm oranı zor sahnelerde ölçülmüş olur
- ramak kala 25'e çıkar → kategori bazlı sayı anlamlı hale gelir
- uzun video yolu ilk kez gerçek veriyle sınanır

Bundan sonrası için getiri azalıyor; enerji etiket kalitesine ve iki kişilik
doğrulamaya gitmeli.
