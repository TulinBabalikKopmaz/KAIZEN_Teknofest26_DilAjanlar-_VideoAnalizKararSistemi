"""Canlı operatör konsolu: webcam / RTSP → wake-up → EVREN.

Jüri videosu ve şartname koşusu `app/demo_app.py`. Bu ekran sunumdaki kısa
canlı izleme şovu içindir; pipeline kuralını değiştirmez.

    py -m streamlit run app/live_app.py --server.port 8502
"""

from __future__ import annotations

import html
import importlib
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for _path in (ROOT, APP_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ui_chrome import inject_chrome, theme_toggle  # noqa: E402
from utils import config  # noqa: E402
from utils import display as kz_display  # noqa: E402
from utils.live_watch import StreamConfig, get_hub, looks_like_jpeg  # noqa: E402

if not hasattr(kz_display, "watch_banner"):
    kz_display = importlib.reload(kz_display)
if not hasattr(kz_display, "law_support_card"):
    kz_display = importlib.reload(kz_display)

watch_banner = kz_display.watch_banner
model_source = kz_display.model_source
law_support_card = kz_display.law_support_card

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

st.set_page_config(
    page_title="KAIZEN · Canlı izleme",
    layout="wide",
    initial_sidebar_state="expanded",
)


def list_local_videos() -> list[Path]:
    found: list[Path] = []
    for directory in (ROOT / "data" / "videos", ROOT / "data" / "incoming"):
        if directory.exists():
            found.extend(p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def sidebar() -> StreamConfig:
    st.sidebar.markdown('<p class="kz-brand">KAIZEN</p>', unsafe_allow_html=True)
    st.sidebar.caption("Canlı saha izleme. Kayıt gerçek zamanlı oynar.")
    theme_toggle()
    st.sidebar.markdown("---")
    source_kind = st.sidebar.radio(
        "Kaynak",
        ("dosya", "webcam"),
        index=0,
        help="Kayıt dosyası kamera gibi akar. Webcam ayrı deneme.",
    )
    source = "0"
    loop_file = False
    if source_kind == "webcam":
        source = str(st.sidebar.number_input("Kamera indeksi", min_value=0, max_value=8, value=0))
    else:
        videos = list_local_videos()
        names = [p.name for p in videos]
        uploaded = st.sidebar.file_uploader("Video bırak", type=[ext.lstrip(".") for ext in VIDEO_EXTS])
        if uploaded is not None:
            incoming = ROOT / "data" / "incoming"
            incoming.mkdir(parents=True, exist_ok=True)
            dest = incoming / uploaded.name
            dest.write_bytes(uploaded.getvalue())
            source = str(dest)
            loop_file = True
            st.sidebar.caption(f"Yüklendi: {uploaded.name}")
        elif names:
            pick = st.sidebar.selectbox("Kayıt (döngü)", names)
            chosen = next(p for p in videos if p.name == pick)
            source = str(chosen)
            loop_file = True
        else:
            st.sidebar.warning("data/videos altında dosya yok; yukarıdan yükleyin.")
            source = "0"
    motion = st.sidebar.slider("Hareket eşiği", 4.0, 40.0, 12.0, 1.0)
    cooldown = st.sidebar.slider("Tetik sonrası sessizlik (sn)", 4.0, 20.0, 8.0, 1.0)
    ep = config.vlm_endpoint()
    st.sidebar.markdown("---")
    st.sidebar.write(f"VLM · {'EVREN' if ep.provider == 'teknofest' else ep.provider}")
    st.sidebar.caption("Ekran gerçek akış (~18 fps). Model 12 karelik kısa klip okur.")
    return StreamConfig(
        source=source,
        motion_threshold=float(motion),
        cooldown_s=float(cooldown),
        loop_file=loop_file,
        max_workers=1,
    )


def banner_html(snap: dict[str, Any]) -> str:
    banner = snap.get("banner") or watch_banner(str(snap.get("phase") or "idle"))
    tone = html.escape(str(banner.get("tone") or "ok"))
    source = model_source(str(snap.get("provider") or config.provider()), backup=False)
    phase = str(snap.get("phase") or "idle")
    dot = "is-alert" if tone == "critical" else ("is-wait" if phase in {"candidate", "analyzing"} else "")
    err = str(snap.get("error") or "")
    err_html = f'<p class="kz-sub">{html.escape(err)}</p>' if err else ""
    return f"""
<div class="kz-verdict {tone}">
  <div class="kz-glow"></div>
  <div class="kz-source {html.escape(source['tone'])}">
    Kaynak · {html.escape(source['label'])}
    <span>{html.escape(source.get('detail') or '')}</span>
  </div>
  <div class="kz-kicker"><span class="kz-live-dot {dot}"></span>{html.escape(str(banner.get('kicker') or ''))}</div>
  <div class="kz-title">{html.escape(str(banner.get('title') or ''))}</div>
  <p class="kz-sub">{html.escape(str(banner.get('subtitle') or ''))}</p>
  {err_html}
</div>
"""


def metrics_html(snap: dict[str, Any]) -> str:
    banner = snap.get("banner") or {}
    phase = str(snap.get("phase") or "")
    if phase == "decided":
        event_time = str(snap.get("event_time") or snap.get("trigger_time") or "—")
        cells = [
            ("Algılanan an", event_time),
            ("Saha durumu", str(banner.get("situation") or banner.get("title") or "—")),
            ("Karar", str(banner.get("decision") or "—")),
        ]
        if snap.get("latency_s"):
            cells.append(("Model", f"{float(snap['latency_s']):.0f} sn"))
    else:
        cells = [
            ("Durum", str(banner.get("kicker") or phase or "—")),
            ("Hareket", f"{float(snap.get('motion_score') or 0):.0f}"),
            ("Tetik", str(snap.get("triggers") or 0)),
            ("Algılanan an", str(snap.get("trigger_time") or "—")),
        ]
    parts = ['<div class="kz-metrics">']
    for label, value in cells:
        parts.append(
            "<div class='kz-metric'>"
            f"<div class='l'>{html.escape(label)}</div>"
            f"<div class='v'>{html.escape(str(value) or '—')}</div>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_feed(hub: Any) -> None:
    """Dosyaya yarım yazılmış JPEG'i okumaz; bellekteki tam kareyi basar."""
    data = hub.latest_jpeg()
    if looks_like_jpeg(data):
        try:
            st.image(data, width="stretch")
            return
        except Exception:
            pass
    st.markdown(
        '<p class="kz-empty">Kayıt henüz kare göndermedi. İzlemeyi başlatın.</p>',
        unsafe_allow_html=True,
    )


def actions_html(actions: list[str]) -> str:
    clean = [item for item in actions if item]
    if not clean:
        return '<p class="kz-empty">Aksiyon önerisi üretilmedi.</p>'
    items = "".join(f"<li>{html.escape(item)}</li>" for item in clean[:6])
    return f'<ol class="kz-actions">{items}</ol>'


def show_law_support(snap: dict[str, Any]) -> None:
    law_note = str(snap.get("law_note") or "")
    law_detail = str(snap.get("law_detail") or "")
    card = law_support_card(law_detail) if law_detail else None
    if not law_note and not card:
        return
    label = str((card or {}).get("kicker") or law_note)
    articles = (card or {}).get("articles") or []
    body_parts: list[str] = []
    for item in articles:
        title = html.escape(str(item.get("title") or ""))
        text = html.escape(str(item.get("text") or ""))
        body_parts.append(
            f'<div class="kz-law-item"><div class="t">{title}</div><p>{text}</p></div>'
        )
    inner = "".join(body_parts) or (
        f'<p class="kz-law">{html.escape(law_detail or law_note)}</p>'
    )
    st.markdown('<div class="kz-law-wrap">', unsafe_allow_html=True)
    with st.expander(label, expanded=False):
        st.markdown(f'<div class="kz-law-body">{inner}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_briefing(snap: dict[str, Any]) -> None:
    """Tetikte iskelet, kararda dolu. Kartın altında tam genişlik."""
    spec = snap.get("spec") or {}
    phase = str(snap.get("phase") or "")
    decided = phase == "decided" or bool(
        spec.get("summary") or spec.get("actions") or spec.get("events")
    )
    summary = str(spec.get("summary") or "").strip()
    actions = list(spec.get("actions") or [])
    st.markdown('<div class="kz-brief">', unsafe_allow_html=True)
    st.markdown('<div class="kz-section">Olay özeti</div>', unsafe_allow_html=True)
    if decided and summary:
        st.write(summary)
    elif phase in {"candidate", "analyzing"}:
        st.markdown(
            '<p class="kz-live-note">Özet model dönünce yazılır. Akış durmadı.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="kz-empty">Tetik yok. Olay özeti burada durur.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="kz-section">Saha aksiyonları</div>', unsafe_allow_html=True)
    if decided and any(actions):
        st.markdown(actions_html(actions), unsafe_allow_html=True)
    elif phase in {"candidate", "analyzing"}:
        st.markdown(
            '<p class="kz-live-note">Aksiyon önerisi karar ile birlikte gelir.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="kz-empty">Tetik yok. Aksiyon listesi burada durur.</p>',
            unsafe_allow_html=True,
        )

    if decided:
        show_law_support(snap)
    elif phase in {"candidate", "analyzing"}:
        st.markdown('<div class="kz-section">Mevzuat</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="kz-live-note">Madde özeti karar gelince expander olarak açılır.</p>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_chrome()
    hub = get_hub()
    cfg = sidebar()

    st.markdown(
        """
<div class="kz-top">
  <p class="kz-brand">KAIZEN</p>
  <h1 class="kz-hero">Canlı saha izleme</h1>
  <p class="kz-lede">Canlı akışta her kare modele gitmez. Hareket algılanınca kısa klip arka planda okunur, operatör kartı güncellenir.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    start, stop = st.columns(2)
    if start.button("İzlemeyi başlat", type="primary", width="stretch"):
        hub.start(cfg)
    if stop.button("Durdur", width="stretch"):
        hub.stop()
    if hub.running():
        st.caption("Akış açık — soldaki görüntü canlı; model kısa klibi arkada okur.")
    else:
        st.caption("Akış kapalı. Kenardan dosya seçin veya video bırakın, sonra başlatın.")

    col_a, col_b = st.columns([1.15, 1], gap="large")
    with col_a:
        with st.container(border=True):
            st.markdown('<div class="kz-section">Kayıt</div>', unsafe_allow_html=True)

            @st.fragment(run_every=0.12)
            def preview_panel() -> None:
                render_feed(hub)

            preview_panel()
    with col_b:
        @st.fragment(run_every=0.5)
        def status_panel() -> None:
            snap = hub.snapshot()
            st.markdown(banner_html(snap), unsafe_allow_html=True)
            st.markdown(metrics_html(snap), unsafe_allow_html=True)

        status_panel()

    with st.container(border=True):
        @st.fragment(run_every=0.7)
        def briefing_panel() -> None:
            render_briefing(hub.snapshot())

        briefing_panel()


main()
