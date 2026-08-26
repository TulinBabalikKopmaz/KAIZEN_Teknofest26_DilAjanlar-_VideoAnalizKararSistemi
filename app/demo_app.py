"""Canlı demo ekranı: jürinin videosu + jürinin promptu → tek ekranda İSG kararı.

    streamlit run app/demo_app.py

Etiketleme aracı ayrı dosyada (app/review_app.py); bu ekran sadece demo içindir.
Görsel cilayı UI arkadaşı üstlenir — kopya ve karar kartı burada kilitlidir.
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import config  # noqa: E402
from utils.demo_pipeline import (  # noqa: E402
    DEFAULT_PROMPT,
    DemoResult,
    list_saved_runs,
    load_saved_run,
    run_demo_analysis_sync,
)
from utils.display import hard_case_note, spec_footnote, verdict  # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
INCOMING_DIR = ROOT / "data" / "incoming"
VIDEO_DIRS = (ROOT / "data" / "videos", INCOMING_DIR)

_UI_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap');

:root {
  --kz-bg: #0C1014;
  --kz-panel: #141A21;
  --kz-panel-2: #1A222C;
  --kz-line: #2A3542;
  --kz-text: #E8EEF4;
  --kz-muted: #8B9AAB;
  --kz-brass: #B8952E;
  --kz-ok: #3D7A62;
  --kz-watch: #B8952E;
  --kz-critical: #A84A3C;
  --kz-font: "IBM Plex Sans", "Segoe UI", sans-serif;
  --kz-display: "IBM Plex Serif", Georgia, serif;
}

html, body, [class*="css"] {
  font-family: var(--kz-font) !important;
}

#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
header { background: transparent; }

.stApp {
  background:
    radial-gradient(1200px 500px at 12% -10%, rgba(184, 149, 46, 0.07), transparent 55%),
    radial-gradient(900px 420px at 100% 0%, rgba(61, 122, 98, 0.05), transparent 50%),
    var(--kz-bg);
}

.block-container {
  padding-top: 1.6rem;
  padding-bottom: 3rem;
  max-width: 1240px;
}

[data-testid="stSidebar"] {
  background: #0A0E12;
  border-right: 1px solid var(--kz-line);
}
[data-testid="stSidebar"] * { color: var(--kz-text); }

.kz-top {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
  margin-bottom: 1.35rem;
  padding-bottom: 1.15rem;
  border-bottom: 1px solid var(--kz-line);
}
.kz-brand {
  letter-spacing: 0.28em;
  text-transform: uppercase;
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--kz-brass);
}
.kz-hero {
  font-family: var(--kz-display);
  font-size: 2rem;
  font-weight: 600;
  line-height: 1.15;
  color: var(--kz-text);
  margin: 0;
}
.kz-lede {
  max-width: 42rem;
  color: var(--kz-muted);
  font-size: 0.98rem;
  line-height: 1.55;
  margin: 0;
}

.kz-panel {
  background: var(--kz-panel);
  border: 1px solid var(--kz-line);
  border-radius: 4px;
  padding: 1rem 1.1rem 1.15rem;
  margin-bottom: 0.85rem;
}
.kz-panel-label {
  font-size: 0.68rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--kz-muted);
  margin-bottom: 0.65rem;
  font-weight: 600;
}

.kz-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.65rem;
  margin: 0.2rem 0 1rem;
}
.kz-metric {
  background: var(--kz-panel);
  border: 1px solid var(--kz-line);
  border-radius: 4px;
  padding: 0.85rem 0.95rem;
}
.kz-metric .l {
  font-size: 0.65rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--kz-muted);
  margin-bottom: 0.35rem;
}
.kz-metric .v {
  font-family: var(--kz-display);
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--kz-text);
  line-height: 1.25;
}

.kz-verdict {
  border: 1px solid var(--kz-line);
  border-left-width: 3px;
  padding: 1.15rem 1.3rem 1.25rem;
  border-radius: 4px;
  margin: 0 0 1.15rem;
  background: var(--kz-panel);
}
.kz-verdict.ok { border-left-color: var(--kz-ok); }
.kz-verdict.watch { border-left-color: var(--kz-watch); }
.kz-verdict.critical {
  border-left-color: var(--kz-critical);
  background: linear-gradient(90deg, rgba(168, 74, 60, 0.12), var(--kz-panel) 42%);
}
.kz-kicker {
  font-size: 0.65rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--kz-muted);
  font-weight: 600;
}
.kz-title {
  font-family: var(--kz-display);
  font-size: 1.55rem;
  font-weight: 600;
  margin: 0.35rem 0 0.25rem;
  line-height: 1.2;
  color: var(--kz-text);
}
.kz-sub { color: var(--kz-muted); margin: 0; font-size: 0.92rem; line-height: 1.45; }
.kz-answer {
  margin-top: 0.95rem;
  padding-top: 0.85rem;
  border-top: 1px solid var(--kz-line);
  font-size: 1.02rem;
  line-height: 1.5;
  color: var(--kz-text);
}
.kz-hard {
  margin-top: 0.85rem;
  padding-top: 0.75rem;
  border-top: 1px solid rgba(184, 149, 46, 0.28);
  font-size: 0.92rem;
  line-height: 1.45;
  color: var(--kz-text);
}
.kz-hard-kicker {
  font-size: 0.65rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--kz-brass);
  font-weight: 600;
  margin-bottom: 0.2rem;
}

.kz-section {
  font-size: 0.68rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--kz-muted);
  font-weight: 600;
  margin: 0.2rem 0 0.7rem;
}

.kz-timeline { list-style: none; padding: 0; margin: 0 0 1rem; }
.kz-timeline li {
  display: grid;
  grid-template-columns: 4.2rem 1fr;
  gap: 0.75rem;
  padding: 0.55rem 0;
  border-bottom: 1px solid rgba(42, 53, 66, 0.85);
}
.kz-timeline li:last-child { border-bottom: none; }
.kz-time {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  color: var(--kz-brass);
  font-size: 0.9rem;
}
.kz-event { color: var(--kz-text); line-height: 1.4; font-size: 0.95rem; }

.kz-actions { list-style: none; padding: 0; margin: 0; counter-reset: act; }
.kz-actions li {
  counter-increment: act;
  display: grid;
  grid-template-columns: 1.6rem 1fr;
  gap: 0.65rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(42, 53, 66, 0.7);
  color: var(--kz-text);
  line-height: 1.4;
}
.kz-actions li::before {
  content: counter(act, decimal-leading-zero);
  color: var(--kz-brass);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  font-size: 0.85rem;
}

.kz-empty {
  color: var(--kz-muted);
  font-size: 0.92rem;
  padding: 0.35rem 0 0.6rem;
}

div[data-testid="stMetric"] {
  background: var(--kz-panel);
  border: 1px solid var(--kz-line);
  border-radius: 4px;
  padding: 0.65rem 0.8rem;
}
div[data-testid="stMetricLabel"] { color: var(--kz-muted) !important; }
div[data-testid="stFileUploader"] {
  background: var(--kz-panel);
  border: 1px dashed var(--kz-line);
  border-radius: 4px;
  padding: 0.4rem 0.6rem;
}
.stButton > button[kind="primary"] {
  background: var(--kz-brass) !important;
  color: #11161c !important;
  border: none !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em;
  border-radius: 3px !important;
}
.stButton > button[kind="primary"]:hover {
  filter: brightness(1.06);
}
.stTextArea textarea {
  background: var(--kz-panel-2) !important;
  border-color: var(--kz-line) !important;
  color: var(--kz-text) !important;
}

@media (max-width: 900px) {
  .kz-metrics { grid-template-columns: 1fr 1fr; }
  .kz-hero { font-size: 1.55rem; }
}
</style>
"""

st.set_page_config(
    page_title="KAIZEN · Saha İSG",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_chrome() -> None:
    st.markdown(_UI_CSS, unsafe_allow_html=True)


def list_local_videos() -> list[Path]:
    found: list[Path] = []
    for directory in VIDEO_DIRS:
        if directory.exists():
            found.extend(p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def save_upload(uploaded: Any) -> Path:
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    dest = INCOMING_DIR / uploaded.name
    dest.write_bytes(uploaded.getbuffer())
    return dest


def sidebar_settings() -> dict[str, Any]:
    st.sidebar.markdown(
        '<div class="kz-brand" style="margin-bottom:0.75rem">KAIZEN</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Saha İSG karar sistemi")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Altyapı**")
    provider = st.sidebar.selectbox(
        "Model sağlayıcı",
        options=list(config.PROVIDERS),
        index=list(config.PROVIDERS).index(config.provider()),
        help="teknofest = yarışmanın ortak API'si, ollama = yerel yedek, mock = modelsiz deneme",
    )
    os.environ["PROVIDER"] = provider

    fast = st.sidebar.toggle(
        "Hızlı mod (süre bütçesi)",
        value=config.demo_fast_mode(),
        help="YOLO kanıtı, ikinci bakış ve mevzuat referansı kapanır",
    )
    max_frames = st.sidebar.slider("Kare sayısı", 4, 12, config.demo_max_frames())
    use_rag = st.sidebar.toggle("İSG mevzuat referansı (RAG)", value=not fast)

    st.sidebar.caption(config.describe())

    saved = list_saved_runs()
    if saved:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Sahne yedeği**")
        labels = [p.name for p in saved]
        pick = st.sidebar.selectbox("Kayıtlı koşu", ["(canlı analiz)"] + labels)
        if pick != "(canlı analiz)" and st.sidebar.button("Yedeği ekrana getir"):
            chosen = next(p for p in saved if p.name == pick)
            st.session_state["last_result"] = load_saved_run(chosen)
            st.session_state["showing_backup"] = True
            st.sidebar.success(f"Açıldı: {pick}")

    if provider == "ollama":
        st.sidebar.warning(
            "Yerel model 2–4 dk sürebilir. "
            "1 dk sahnede hızlı mod veya kayıtlı yedek kullanın."
        )
    return {"fast": fast, "max_frames": max_frames, "use_rag": use_rag}


def verdict_card(result: DemoResult) -> dict[str, str]:
    v = verdict(result.label.get("category"), result.spec.get("risk"))
    answer_html = html.escape(result.answer or "").replace("\n", "<br>")
    note = hard_case_note(result.label, result.spec, result.evidence)
    hard_html = ""
    if note:
        hard_html = (
            '<div class="kz-hard">'
            f'<div class="kz-hard-kicker">{html.escape(note["kicker"])}</div>'
            f"{html.escape(note['text'])}"
            "</div>"
        )
    st.markdown(
        f"""
<div class="kz-verdict {v['tone']}">
  <div class="kz-kicker">{html.escape(v['kicker'])}</div>
  <div class="kz-title">{html.escape(v['situation'])} · {html.escape(v['decision'])}</div>
  <p class="kz-sub">{html.escape(v['subtitle'])}</p>
  <div class="kz-answer">{answer_html}</div>
  {hard_html}
</div>
""",
        unsafe_allow_html=True,
    )
    return v


def metrics_strip(result: DemoResult, v: dict[str, str]) -> None:
    events = result.spec.get("events") or []
    cells = [
        ("Saha durumu", v["situation"]),
        ("Karar", v["decision"]),
        ("Analiz süresi", f"{result.total_s:.1f} sn"),
        ("İşaretlenen olay", str(len(events))),
    ]
    parts = ['<div class="kz-metrics">']
    for label, value in cells:
        parts.append(
            "<div class='kz-metric'>"
            f"<div class='l'>{html.escape(label)}</div>"
            f"<div class='v'>{html.escape(value)}</div>"
            "</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def timeline_html(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<p class="kz-empty">Ayrı bir olay satırı işaretlenmedi; rutin akış.</p>'
    items = []
    for item in events:
        t = html.escape(str(item.get("time") or "00:00"))
        e = html.escape(str(item.get("event") or ""))
        items.append(f'<li><span class="kz-time">{t}</span><span class="kz-event">{e}</span></li>')
    return f'<ul class="kz-timeline">{"".join(items)}</ul>'


def actions_html(actions: list[str]) -> str:
    clean = [a for a in actions if a]
    if not clean:
        return '<p class="kz-empty">Aksiyon önerisi üretilmedi.</p>'
    items = "".join(f"<li>{html.escape(a)}</li>" for a in clean)
    return f'<ol class="kz-actions">{items}</ol>'


def show_result(result: DemoResult) -> None:
    spec = result.spec
    events = spec.get("events") or []
    v = verdict(result.label.get("category"), spec.get("risk"))

    metrics_strip(result, v)
    verdict_card(result)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="kz-section">Kayıt ve kanıt</div>', unsafe_allow_html=True)
        video_path = Path(result.video)
        if video_path.exists():
            st.video(str(video_path))
        frames = result.frames
        if frames:
            grid = st.columns(min(4, len(frames)))
            for i, frame in enumerate(frames):
                path = frame.get("demo_path") or frame.get("path")
                if path and Path(path).exists():
                    grid[i % len(grid)].image(path, caption=frame.get("time", ""))

    with right:
        st.markdown('<div class="kz-section">Olay zaman çizelgesi</div>', unsafe_allow_html=True)
        st.markdown(timeline_html(events), unsafe_allow_html=True)

        st.markdown('<div class="kz-section">Özet</div>', unsafe_allow_html=True)
        st.write(spec.get("summary") or "—")

        st.markdown('<div class="kz-section">Saha aksiyonları</div>', unsafe_allow_html=True)
        st.markdown(actions_html(list(spec.get("actions") or [])), unsafe_allow_html=True)

        with st.expander("Jüri çıktısı (şartname JSON)"):
            st.caption(spec_footnote())
            st.json(spec)

        with st.expander("Aşama süreleri ve model çağrıları"):
            st.table(
                [{"aşama": key, "saniye": value} for key, value in result.timings.items()]
            )
            if result.model_calls:
                st.table(result.model_calls)

        with st.expander("Sensör kanıtı (hareket / yakınlık / yangın)"):
            st.json(result.evidence)

        d1, d2 = st.columns(2)
        d1.download_button(
            "Şartname JSON",
            data=json.dumps(spec, ensure_ascii=False, indent=2),
            file_name=f"{Path(result.video).stem}_spec.json",
            mime="application/json",
            use_container_width=True,
        )
        d2.download_button(
            "Tam sonuç JSON",
            data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            file_name=f"{Path(result.video).stem}_result.json",
            mime="application/json",
            use_container_width=True,
        )

    for warning in result.warnings:
        st.warning(warning)


def main() -> None:
    inject_chrome()
    settings = sidebar_settings()

    st.markdown(
        """
<div class="kz-top">
  <div class="kz-brand">KAIZEN · TEKNOFEST 2026</div>
  <h1 class="kz-hero">Saha İSG Karar Sistemi</h1>
  <p class="kz-lede">
    Kamera kaydını arşiv değil karar haline getirir.
    Sahne için kısa klip veya kayıtlı yedek; jüri videosunda yükle, soruyu yapıştır, analiz et.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.session_state.get("showing_backup"):
        st.warning(
            "Ekranda kayıtlı sahne yedeği var (canlı API sonucu değil). "
            "Jüri videosunda Analiz et ile taze koşu alın."
        )

    source_col, prompt_col = st.columns([1, 1.35], gap="large")
    with source_col:
        st.markdown('<div class="kz-panel-label">Girdi</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Jürinin videosu",
            type=[e.strip(".") for e in VIDEO_EXTS],
            label_visibility="collapsed",
        )
        st.caption("Video yükle (mp4, mov, …)")
        local_videos = list_local_videos()
        picked = None
        if local_videos:
            names = ["(yüklenen dosyayı kullan)"] + [
                str(p.relative_to(ROOT)) for p in local_videos[:40]
            ]
            choice = st.selectbox("veya klasörden seç", names)
            if choice != names[0]:
                picked = ROOT / choice

    with prompt_col:
        st.markdown('<div class="kz-panel-label">Operatör</div>', unsafe_allow_html=True)
        prompt = st.text_area(
            "Operatör sorusu",
            value=DEFAULT_PROMPT,
            height=130,
            label_visibility="collapsed",
        )
        run = st.button("Analiz et", type="primary", use_container_width=True)

    video_path: Path | None = None
    if uploaded is not None:
        video_path = save_upload(uploaded)
    elif picked is not None:
        video_path = picked

    if run:
        if video_path is None:
            st.error("Önce bir video yükleyin veya klasörden seçin.")
            return
        status = st.status("Analiz çalışıyor…", expanded=True)
        try:
            result = run_demo_analysis_sync(
                video_path,
                prompt,
                max_frames=settings["max_frames"],
                fast=settings["fast"],
                use_rag=settings["use_rag"],
                progress=status.write,
            )
        except Exception as exc:
            status.update(label="Analiz başarısız", state="error")
            st.error(f"Analiz tamamlanamadı: {exc}")
            backup = list_saved_runs(limit=1)
            if backup:
                st.session_state["last_result"] = load_saved_run(backup[0])
                st.session_state["showing_backup"] = True
                st.warning(
                    f"EVREN yanıt vermedi. Kayıtlı sahne yedeği açıldı: `{backup[0].name}`."
                )
            else:
                st.info(
                    "Kayıtlı yedek yok. Sahneden önce seçilen klibi "
                    "`python scripts/analyze_video.py --video <klip> --fast --run-name sahne_yedek` "
                    "ile bir kez koşun veya kenar çubuğundan ollama deneyin."
                )
                return
        else:
            status.update(label=f"Tamamlandı ({result.total_s:.1f} sn)", state="complete")
            st.session_state["last_result"] = result
            st.session_state["showing_backup"] = False

    result = st.session_state.get("last_result")
    if isinstance(result, DemoResult):
        show_result(result)
    else:
        st.markdown(
            '<p class="kz-empty">Kayıt ve soruyu verip Analiz et’e basın. Karar kartı burada açılır.</p>',
            unsafe_allow_html=True,
        )


main()
