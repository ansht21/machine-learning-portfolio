"""Metric Flow UI Component library for the E-Commerce Reconciliation Dashboard.

Provides identical shape, grid, panel, topbar, and KPI structures as the Supply Chain Cockpit.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from html import escape
from typing import Any

KPI_TONE_CLASS = {
    "positive": "positive",
    "warning": "warning",
    "negative": "negative",
}


def _escape(value: Any) -> str:
    return escape(str(value), quote=True)


def render_topbar(
    *,
    title: str,
    subtitle: str,
    period: str = "Live data",
    initials: str = "AT",
) -> None:
    import streamlit as st

    st.markdown(
        f"""
        <div class="metric-flow-topbar">
          <div class="metric-flow-brand">
            <div class="metric-flow-mark">⚡</div>
            <div class="metric-flow-heading">
              <div class="metric-flow-title">{_escape(title)}</div>
              <div class="metric-flow-subtitle">{_escape(subtitle)}</div>
            </div>
          </div>
          <div class="metric-flow-context">
            <div class="metric-flow-period">{_escape(period)}</div>
            <div class="metric-flow-avatar" aria-label="Profile">{_escape(initials)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand(
    *,
    product: str,
    role: str,
    initials: str = "ER",
) -> None:
    import streamlit as st

    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
          <div class="sidebar-brand-mark">{_escape(initials)}</div>
          <div>
            <div class="sidebar-brand-name">{_escape(product)}</div>
            <div class="sidebar-brand-role">{_escape(role)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_profile(name: str = "Ansh Thakur", role: str = "Data Scientist & AI Practitioner") -> None:
    import streamlit as st

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.65rem;padding:0.4rem 0;">
            <div style="width:2rem;height:2rem;border-radius:50%;background:#6D28D9;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:0.75rem;">AT</div>
            <div>
                <div style="font-size:0.85rem;font-weight:600;color:#F3F0FF;">{_escape(name)}</div>
                <div style="font-size:0.72rem;color:#9D97BF;">{_escape(role)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi_html(card: dict[str, Any]) -> str:
    delta = card.get("delta")
    tone = card.get("tone")
    delta_html = ""
    if delta:
        tone_class = KPI_TONE_CLASS.get(str(tone), "")
        delta_html = f'<div class="kpi-delta {tone_class}">{_escape(delta)}</div>'
    sub = card.get("sub", "")
    sub_html = f'<div class="kpi-sub">{_escape(sub)}</div>' if sub else ""
    return f"""
      <article class="metric-flow-kpi">
        {delta_html}
        <div class="kpi-label">{_escape(card.get("label", "Metric"))}</div>
        <div class="kpi-value">{_escape(card.get("value", "—"))}</div>
        {sub_html}
      </article>
    """


def kpi_row(cards: Sequence[dict[str, Any]], *, columns: int = 4) -> None:
    import streamlit as st

    if not cards:
        return
    count = min(columns, len(cards))
    grid_style = f"grid-template-columns: repeat({count}, minmax(0, 1fr));"
    markup = "".join(_kpi_html(card) for card in cards)
    st.markdown(
        f'<div class="kpi-grid" style="{grid_style}">{markup}</div>',
        unsafe_allow_html=True,
    )


@contextmanager
def panel(title: str, subtitle: str = "") -> Iterator[Any]:
    import streamlit as st

    sub_html = ""
    if subtitle:
        sub_html = f'<div class="metric-flow-panel-sub">{_escape(subtitle)}</div>'
    container = st.container(border=True)
    with container:
        st.markdown(
            f'<div class="metric-flow-panel-title">{_escape(title)}</div>'
            f"{sub_html}",
            unsafe_allow_html=True,
        )
        yield container
