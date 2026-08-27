# KAIZEN — TEKNOFEST 2026 Dil Ajanları (Senaryo 3)

Otonom endüstriyel İSG video analiz ve karar sistemi. Resmi API: EVREN
(`vlm` video, `llm-fast` metin). LangGraph: Video Analyzer → Risk Assessor →
Action Recommender.

Yeni gelen: `git pull origin main`. `.env` yok; `.env.example` kopyala, anahtarı
yazma (paylaşma).

## Kim neye bakar

| Rol | Belgeler | Çalıştır |
|---|---|---|
| UI | `docs/ui_taslak.md`, `app/jury_ui/`, `app/demo_app.py` | `py app/jury_server.py` → :8503 · Streamlit yedek :8501 · canlı :8502 |
| Sunum | `docs/sunum_akisi.md` | slayt + 1 dk demo prova; canlı izleme ayrı sekme |
| Pipeline | `utils/demo_pipeline.py`, `utils/risk_rules.py` | donduruldu — yarışmadan önce kural oynatma |

## Dil (ekran ≠ JSON)

İç kod `normal` / `near_miss` / `accident` KPI ve klasör içindir.
Jüri JSON `risk`: Düşük | Orta | Yüksek.
İnsan yüzü: Rutin operasyon · Kontrol altında; Ramak kala · Yüksek dikkat;
İş kazası · Kritik durum. "Risk: Düşük" demeyin.

## Dondurma

Ayar yalnız **dev** (46 video). **Test** (31) bir kez koşuldu; üzerine kural yok.
VLM tamamlanmış kaza yazdıysa gold ramak olsa bile indirme.
Sahne klibi ekip kararı; merdiveni kilitleme.

İç ölçüm (jüri skoru değil): dev risk %87, olay %92, kritik %100, yanlış alarm %0.
Test: risk %74, olay %73, kritik %100, normalde yanlış kaza %0. Near miss testte zayıf.

## Demo

Sahne: hızlı mod açılabilir + `data/demo_runs/` yedek.
Jüri videosu: hızlı mod kapalı, 8 kare, tam pipeline.
