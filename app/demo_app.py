"""Canlı demo ekranı: jürinin videosu + jürinin promptu → tek ekranda İSG kararı.

    streamlit run app/demo_app.py

Etiketleme aracı ayrı dosyada (app/review_app.py); bu ekran sadece demo içindir.
"""

from __future__ import annotations

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
    run_demo_analysis_sync,
)

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
INCOMING_DIR = ROOT / "data" / "incoming"
VIDEO_DIRS = (ROOT / "data" / "videos", INCOMING_DIR)

RISK_STYLE = {
    "Yüksek": ("KRİTİK RİSK", "error"),
    "Orta": ("ORTA RİSK", "warning"),
    "Düşük": ("DÜŞÜK RİSK", "success"),
}

st.set_page_config(page_title="İSG Canlı Demo", layout="wide")


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
    st.sidebar.header("Sistem")
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
    if provider == "ollama":
        st.sidebar.warning(
            "Yerel 7B 2–4 dk sürebilir; bu jüri kaydı için sorun değil. "
            "Sahnedeki 1 dk gösterimde **Hızlı mod** açın veya önceden koşulmuş sonucu gösterin."
        )
    return {"fast": fast, "max_frames": max_frames, "use_rag": use_rag}


def risk_banner(risk: str, answer: str) -> None:
    title, kind = RISK_STYLE.get(risk, ("DEĞERLENDİRİLDİ", "info"))
    body = f"**{title}**\n\n{answer}"
    getattr(st, kind)(body)


def show_result(result: DemoResult) -> None:
    spec = result.spec
    events = spec.get("events") or []

    cols = st.columns(4)
    cols[0].metric("Risk", spec.get("risk", "-"))
    cols[1].metric("Kategori", result.label.get("category", "-"))
    cols[2].metric("Toplam süre", f"{result.total_s:.1f} sn")
    cols[3].metric("Olay sayısı", len(events))

    risk_banner(spec.get("risk", ""), result.answer)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Video ve kanıt kareleri")
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
            st.info("Modelin işaretlediği ayrı bir olay yok.")

        st.subheader("Özet")
        st.write(spec.get("summary") or "-")

        st.subheader("Saha aksiyonları")
        for i, action in enumerate(spec.get("actions") or [], start=1):
            st.markdown(f"{i}. {action}")

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
    settings = sidebar_settings()

    st.title("Otonom İSG Video Analizi — Canlı Demo")
    st.caption(
        "Videoyu yükleyin, jürinin verdiği promptu yazın ve tek tuşla zaman damgalı "
        "İSG kararını alın."
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
        prompt = st.text_area("Prompt (jürinin sorusu)", value=DEFAULT_PROMPT, height=120)
        run = st.button("Analiz Et", type="primary", use_container_width=True)

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
            st.info(
                "Ortak API yanıt vermiyorsa kenar çubuğundan sağlayıcıyı 'ollama' "
                "yapıp tekrar deneyin."
            )
            return
        status.update(label=f"Tamamlandı ({result.total_s:.1f} sn)", state="complete")
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if isinstance(result, DemoResult):
        show_result(result)
    else:
        st.info("Henüz analiz yapılmadı. Video ve prompt verip 'Analiz Et'e basın.")


main()
