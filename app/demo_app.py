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
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent;}
.block-container {padding-top: 1.4rem; max-width: 1280px;}
.kz-brand {
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: #C9A227;
    margin-bottom: 0.15rem;
}
.kz-verdict {
    border-left: 4px solid;
    padding: 1.05rem 1.25rem 1.15rem;
    border-radius: 6px;
    margin: 0.35rem 0 1.1rem;
}
.kz-verdict.ok { border-color: #3D8B6E; background: #12211C; }
.kz-verdict.watch { border-color: #C9A227; background: #211C12; }
.kz-verdict.critical { border-color: #C45C4A; background: #231512; }
.kz-kicker {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    opacity: 0.65;
}
.kz-title { font-size: 1.55rem; font-weight: 650; margin: 0.2rem 0 0.15rem; line-height: 1.2; }
.kz-sub { opacity: 0.88; margin: 0; font-size: 0.95rem; }
.kz-answer { margin-top: 0.85rem; font-size: 1.02rem; line-height: 1.45; }
.kz-hard {
    margin-top: 0.8rem;
    padding-top: 0.7rem;
    border-top: 1px solid rgba(201, 162, 39, 0.38);
    font-size: 0.92rem;
    line-height: 1.4;
    color: #E6D5A3;
}
.kz-hard-kicker {
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.15rem;
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
    st.sidebar.markdown("**KAIZEN**")
    st.sidebar.caption("Saha İSG karar sistemi")
    st.sidebar.header("Altyapı")
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
        st.sidebar.subheader("Sahne yedeği")
        labels = [p.name for p in saved]
        pick = st.sidebar.selectbox("Kayıtlı koşu", ["(canlı analiz)"] + labels)
        if pick != "(canlı analiz)" and st.sidebar.button("Yedeği ekrana getir"):
            chosen = next(p for p in saved if p.name == pick)
            st.session_state["last_result"] = load_saved_run(chosen)
            st.session_state["showing_backup"] = True
            st.sidebar.success(f"Açıldı: {pick}")

    if provider == "ollama":
        st.sidebar.warning(
            "Yerel model 2–4 dk sürebilir; jüri kaydı için sorun değil. "
            "Sahnedeki 1 dk gösterimde **Hızlı mod** açın veya kayıtlı yedeği gösterin."
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
  <div class="kz-kicker">{v['kicker']}</div>
  <div class="kz-title">{v['situation']} · {v['decision']}</div>
  <p class="kz-sub">{v['subtitle']}</p>
  <div class="kz-answer">{answer_html}</div>
  {hard_html}
</div>
""",
        unsafe_allow_html=True,
    )
    return v


def show_result(result: DemoResult) -> None:
    spec = result.spec
    events = spec.get("events") or []
    v = verdict(result.label.get("category"), spec.get("risk"))

    cols = st.columns(4)
    cols[0].metric("Saha durumu", v["situation"])
    cols[1].metric("Karar", v["decision"])
    cols[2].metric("Analiz süresi", f"{result.total_s:.1f} sn")
    cols[3].metric("İşaretlenen olay", len(events))

    verdict_card(result)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Kayıt ve kanıt kareleri")
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
        st.subheader("Olay zaman çizelgesi")
        if events:
            for item in events:
                st.markdown(f"**{item.get('time', '00:00')}** — {item.get('event', '')}")
        else:
            st.info("Ayrı bir olay satırı işaretlenmedi; rutin akış.")

        st.subheader("Özet")
        st.write(spec.get("summary") or "-")

        st.subheader("Saha aksiyonları")
        for i, action in enumerate(spec.get("actions") or [], start=1):
            st.markdown(f"{i}. {action}")

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

        st.download_button(
            "Şartname JSON indir",
            data=json.dumps(spec, ensure_ascii=False, indent=2),
            file_name=f"{Path(result.video).stem}_spec.json",
            mime="application/json",
        )
        st.download_button(
            "Tam sonuç JSON indir",
            data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            file_name=f"{Path(result.video).stem}_result.json",
            mime="application/json",
        )

    for warning in result.warnings:
        st.warning(warning)


def main() -> None:
    inject_chrome()
    settings = sidebar_settings()

    st.markdown('<div class="kz-brand">KAIZEN · TEKNOFEST 2026</div>', unsafe_allow_html=True)
    st.title("Saha İSG Karar Sistemi")
    st.caption(
        "Kamera kaydını arşiv değil karar haline getirir. "
        "Sahne (1 dk): ekibin seçtiği kısa klip veya kayıtlı yedek. "
        "Jüri videosu: yükle, soruyu yapıştır, hızlı mod kapalı."
    )
    if st.session_state.get("showing_backup"):
        st.warning(
            "Ekranda **kayıtlı sahne yedeği** var (canlı API sonucu değil). "
            "Jüri videosunda Analiz Et ile taze koşu alın."
        )

    source_col, prompt_col = st.columns([1, 1.4])
    with source_col:
        uploaded = st.file_uploader("Jürinin videosu", type=[e.strip(".") for e in VIDEO_EXTS])
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
        prompt = st.text_area("Operatör sorusu", value=DEFAULT_PROMPT, height=120)
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
        status = st.status("Analiz çalışıyor...", expanded=True)
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
                    "Kayıtlı yedek yok. Sahneden önce ekibin seçtiği klibi "
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
        st.info("Kayıt ve soruyu verip Analiz et'e basın. Karar kartı burada açılır.")


main()
