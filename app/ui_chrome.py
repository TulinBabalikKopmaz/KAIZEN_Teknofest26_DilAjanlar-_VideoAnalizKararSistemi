"""Shared Streamlit chrome for the jury demo and live console.

Visual language comes from the Claude Design mock (IBM Plex, cream/dark
tokens, 8px panels). Pipeline and copy stay in utils.display / demo_app.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_THEME_PATH = Path(__file__).resolve().parent / "demo_theme.css"

# Light tokens from the design mock. Dark lives in demo_theme.css :root.
_LIGHT_ROOT = """
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stSidebar"] {
  --kz-bg: #F4EFE6;
  --kz-bg-glow: #FFF7E8;
  --kz-panel-bg: rgba(255,251,245,0.95);
  --kz-panel-hover: #E8E0D2;
  --kz-surface: #FFFBF5;
  --kz-surface-2: #FDF9F0;
  --kz-surface-3: #E8E0D2;
  --kz-stripe-1: #F4EFE6;
  --kz-stripe-2: #EDE4D2;
  --kz-stripe-3: #EDE4D2;
  --kz-stripe-4: #E4D9C2;
  --kz-text: #1C1917;
  --kz-muted: #57534E;
  --kz-muted-soft: #78716C;
  --kz-code-text: #44403C;
  --kz-gold: #A16207;
  --kz-gold-deep: #7C4A05;
  --kz-on-gold: #FFFBF5;
  --kz-ok: #3F6B4A;
  --kz-critical: #BE123C;
  --kz-critical-pulse: #FF2D2D;
  --kz-shadow-soft: rgba(28,25,23,0.12);
  --kz-shadow-med: rgba(28,25,23,0.14);
  --kz-shadow-strong: rgba(28,25,23,0.16);
  --kz-overlay: rgba(28,25,23,0.35);
  --kz-line-rgb: 28,25,23;
  --kz-gold-rgb: 161,98,7;
  --kz-ok-rgb: 63,107,74;
  --kz-critical-rgb: 190,18,60;
  --kz-card-ok: var(--kz-surface);
  --kz-card-watch: var(--kz-surface);
  --kz-card-critical: var(--kz-surface);
  --kz-noise-opacity: 0.035;
  --kz-noise-blend: multiply;
  color-scheme: light;
}
"""


def current_theme() -> str:
    radio = st.session_state.get("kz_theme_radio")
    if radio == "Karanlık":
        return "dark"
    if radio == "Aydınlık":
        return "light"
    return str(st.session_state.get("kz_theme") or "light")


def theme_toggle() -> str:
    """Sidebar sun/moon. Design mock defaulted to light."""
    labels = ("Aydınlık", "Karanlık")
    current = current_theme()
    index = 0 if current == "light" else 1
    picked = st.sidebar.radio("Tema", labels, index=index, horizontal=True, key="kz_theme_radio")
    theme = "light" if picked == "Aydınlık" else "dark"
    st.session_state["kz_theme"] = theme
    return theme


def inject_chrome(theme: str | None = None) -> None:
    theme = theme or current_theme()
    css = _THEME_PATH.read_text(encoding="utf-8")
    extra = _LIGHT_ROOT if theme == "light" else ""
    st.markdown(
        f"<style>{css}</style><style>{extra}</style>",
        unsafe_allow_html=True,
    )
