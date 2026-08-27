"""Canlı demo ekranı: jürinin videosu + jürinin promptu → tek ekranda İSG kararı.

    streamlit run app/demo_app.py

Etiketleme aracı ayrı dosyada (app/review_app.py); bu ekran sadece demo içindir.
Görsel cilayı UI arkadaşı üstlenir — kopya ve karar kartı burada kilitlidir.
"""

from __future__ import annotations

import html
import json
import os
import re
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

_THEME_PATH = Path(__file__).resolve().parent / "demo_theme.css"
_UI_CSS = f"<style>{_THEME_PATH.read_text(encoding='utf-8')}</style>"

st.set_page_config(
    page_title="KAIZEN · Saha İSG",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_chrome() -> None:
    st.markdown(_UI_CSS, unsafe_allow_html=True)


# Jüri yüzü: sabit 5 adım (pipeline [n/5] mesajlarına kilitli)
_FLOW_STEPS: tuple[dict[str, str], ...] = (
    {"title": "Kare çıkarma ve sensör kanıtı", "idle": "Hareket, yakınlık ve yangın sinyalleri"},
    {"title": "Görsel model analizi", "idle": "Kısa klip üzerinden sahne okuması"},
    {"title": "İkinci bakış", "idle": "Şüpheli kanıtta odaklanmış yeniden okuma"},
    {"title": "Kural ve birleştirme", "idle": "Zaman hizalama ve kanıt birleştirme"},
    {"title": "Cevap ve saha aksiyonları", "idle": "Operatör cevabı ve aksiyon listesi"},
)


class AnalysisFlowBoard:
    """Analiz sırasında jüriye adım adım ilerleme paneli gösterir."""

    def __init__(self, slot: Any) -> None:
        self._slot = slot
        self.current = 0  # 1..5 aktif; 0 henüz başlamadı
        self.details: dict[int, str] = {}
        self.state = "running"  # running | done | fail
        self.meta = "Çalışıyor"
        self.render()

    def render(self) -> None:
        current_idx = max(self.current, 1) if self.state != "done" else 5
        if self.state == "done":
            title = "Tamamlandı"
            detail = self.meta
        elif self.state == "fail":
            title = "Analiz kesildi"
            detail = self.details.get(max(self.current, 1), self.meta)
        else:
            step = _FLOW_STEPS[min(current_idx, 5) - 1]
            title = step["title"]
            detail = self.details.get(current_idx) or step["idle"]
        dots: list[str] = []
        for index in range(1, 6):
            if self.state == "fail" and index == max(self.current, 1):
                klass = "bad"
            elif self.state == "done" or index < self.current:
                klass = "on"
            elif index == self.current:
                klass = "now"
            else:
                klass = ""
            dots.append(f'<span class="{klass}"></span>')
        panel = {"running": "is-live", "done": "is-done", "fail": "is-fail"}[self.state]
        busy = "true" if self.state == "running" else "false"
        shimmer = '<span class="kz-shimmer"></span>' if self.state == "running" else ""
        self._slot.markdown(
            f"""
<div class="kz-flow {panel}" role="status" aria-live="polite" aria-busy="{busy}">
  {shimmer}
  <p class="kz-flow-kicker">Analiz akışı</p>
  <p class="kz-flow-title">{html.escape(title)}</p>
  <p class="kz-flow-detail">{html.escape(detail)}</p>
  <div class="kz-dots">{"".join(dots)}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    def _human_detail(self, message: str) -> str | None:
        text = message.strip()
        if text.startswith("Kaydedildi:"):
            return "Koşu arşive alındı"
        if text.startswith("Toplam süre:"):
            # "Toplam süre: 8.3 sn | İş kazası · Kritik durum (şartname: Yüksek) | Cevap: ..."
            head = text.split("|", 2)
            if len(head) >= 2:
                return f"{head[0].strip()} · {head[1].split('(')[0].strip()}"
            return head[0].strip()
        risk = re.search(r"risk=([^\s]+)\s+kategori=([^\s(]+)", text)
        if risk:
            v = verdict(risk.group(2), risk.group(1))
            timing = re.search(r"\(([\d.]+s)\)", text)
            suffix = f" · {timing.group(1)}" if timing else ""
            return f"{v['situation']} · {v['decision']}{suffix}"
        if "EVREN klibi" in text or "klibi" in text.lower():
            return "Kısa analiz klibi hazır"
        if re.match(r"\d+ kare:", text):
            count = text.split(" ", 1)[0]
            return f"{count} kanıt karesi seçildi"
        if "wake-up" in text.lower() or "uzun video" in text.lower():
            return "Uzun kayıtta odak penceresi belirlendi"
        if "metin eleştirmeni" in text.lower():
            return "Metin tutarlılığı kontrol edildi"
        if "gerekmedi" in text.lower():
            return "Kanıt sakin — atlandı"
        if text.lower().startswith("ikinci bakış"):
            return "Odak karelerde yeniden okuma"
        if text.lower().startswith("vlm") or "vlm analizi" in text.lower():
            return "Sahne okunuyor"
        if "kare çıkarma" in text.lower() or "sensör" in text.lower():
            return "Kareler ve kanıt toplanıyor"
        if "kural" in text.lower():
            return "Kurallar uygulanıyor"
        if "cevap" in text.lower() or "aksiyon" in text.lower():
            return "Cevap ve aksiyonlar yazılıyor"
        # Ham teknik satırlar jüriye gitmesin
        if re.match(r"\[\d/5\]", text):
            return None
        if text.startswith("/") or "demo_runs" in text:
            return None
        if len(text) > 120:
            return text[:110].rstrip() + "…"
        return text

    def on_progress(self, message: str) -> None:
        raw = message.strip()
        match = re.match(r"\[(\d)/5\]\s*(.*)$", raw)
        if match:
            step = int(match.group(1))
            self.current = step
            rest = match.group(2).strip()
            detail = self._human_detail(rest) if rest else None
            if detail:
                self.details[step] = detail
            elif step not in self.details:
                self.details[step] = "Devam ediyor…"
            self.meta = f"Adım {step} / 5"
            self.render()
            return
        detail = self._human_detail(raw)
        if detail and self.current:
            self.details[self.current] = detail
            if raw.startswith("Toplam süre:"):
                self.meta = detail
            self.render()

    def complete(self, total_s: float) -> None:
        self.state = "done"
        self.current = 6
        self.meta = f"Tamamlandı · {total_s:.1f} sn"
        for index in range(1, 6):
            self.details.setdefault(index, _FLOW_STEPS[index - 1]["idle"])
        self.render()

    def fail(self, reason: str = "Analiz tamamlanamadı") -> None:
        self.state = "fail"
        self.meta = "Kesildi"
        if self.current:
            self.details[self.current] = reason[:140]
        self.render()


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

    verdict_card(result)
    metrics_strip(result, v)

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
        elif not video_path.exists():
            st.caption("Bu koşuda kayıt veya kare yok.")

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

        preview = st.session_state.get("girdi_video")
        preview_path = Path(preview) if preview else None
        if preview_path is not None and preview_path.exists():
            st.video(str(preview_path))

        upload_n = int(st.session_state.get("girdi_upload_n", 0))
        uploaded = st.file_uploader(
            "Jürinin videosu",
            type=[e.strip(".") for e in VIDEO_EXTS],
            label_visibility="collapsed",
            key=f"girdi_upload_{upload_n}",
        )
        if uploaded is not None:
            dest = save_upload(uploaded)
            st.session_state["girdi_video"] = str(dest)
            st.session_state["girdi_upload_n"] = upload_n + 1
            st.rerun()

        local_videos = list_local_videos()
        picked = None
        if local_videos:
            names = ["(yüklenen dosyayı kullan)"] + [
                str(p.relative_to(ROOT)) for p in local_videos[:40]
            ]
            choice = st.selectbox("veya klasörden seç", names)
            if choice != names[0]:
                picked = ROOT / choice
                if st.session_state.get("girdi_video") != str(picked):
                    st.session_state["girdi_video"] = str(picked)
                    st.rerun()

        video_path: Path | None = None
        stored = st.session_state.get("girdi_video")
        if stored and Path(stored).exists():
            video_path = Path(stored)
        elif preview_path is None:
            st.caption("mp4, mov, avi, mkv")

    with prompt_col:
        st.markdown('<div class="kz-panel-label">Operatör</div>', unsafe_allow_html=True)
        prompt = st.text_area(
            "Operatör sorusu",
            value=DEFAULT_PROMPT,
            height=130,
            label_visibility="collapsed",
        )
        run = st.button("Analiz et", type="primary", use_container_width=True)

    if run:
        if video_path is None:
            st.error("Önce bir video yükleyin veya klasörden seçin.")
            return
        flow = AnalysisFlowBoard(st.empty())
        try:
            result = run_demo_analysis_sync(
                video_path,
                prompt,
                max_frames=settings["max_frames"],
                fast=settings["fast"],
                use_rag=settings["use_rag"],
                progress=flow.on_progress,
            )
        except Exception as exc:
            flow.fail("Bağlantı veya model yanıtı alınamadı")
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
            flow.complete(result.total_s)
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
