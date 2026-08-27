repo: TulinBabalikKopmaz/KAIZEN_Teknofest26_DilAjanlar-_VideoAnalizKararSistemi
branch: main
path: app/, utils/, .streamlit/, docs/

## Last sync

date: 2026-08-27T12:27:54Z

### Updated in this project

- Senkronize edildi: `app/demo_app.py` büyüdü (Kaynak rozeti, Sunum kilidi: yalnız EVREN, mevzuat/hukuki destek expander'ı) ve yeni `app/live_app.py` (port 8502, canlı saha izleme) eklendi.
- Bu yeni özellikler mevcut Claude tasarımına entegre edildi: sidebar'a görünüm anahtarı (Jüri demosu / Canlı izleme) ve sunum kilidi toggle'ı, karar kartına Kaynak (EVREN/Ollama/yedek) rozeti, sonuç bölümüne mevzuat desteği expander'ı, ve yeni bir "Canlı izleme" ekranı (mock akış: bekleniyor → hareket algılandı → model okuyor → saha kararı) eklendi.
- Önceki tüm animasyon/tema/akış çalışması (açık-koyu tema, forklift'li analiz akışı, overlay, sayfa içi kayan sonuç) korundu — üstüne inşa edildi.

## Screen map

| Ekran | Kaynak dosyalar |
|---|---|
| Juri Demo Arayuzu v2.dc.html — Jüri demosu (view=demo) | app/demo_app.py (AnalysisFlowBoard, verdict_card/Kaynak rozeti, sidebar_settings/Sunum kilidi, show_law_support), app/demo_theme.css (renk/token fikirleri), utils/display.py (dil kilidi, model_source, law_support_card), utils/demo_pipeline.py (adım metinleri, DEFAULT_PROMPT) |
| Juri Demo Arayuzu v2.dc.html — Canlı izleme (view=live) | app/live_app.py (sidebar/kaynak seçimi, banner_html, metrics_html, render_feed, timeline_html/actions_html, render_briefing), AGENTS.md (port 8502 notu) |
