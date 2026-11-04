"""Cyber Violet & Emerald Luxury Executive Theme.

Maintains the exact executive shape alignment, 4 equal KPI cards, and two-column
panel structure of Metric Flow from the supply-chain dashboard, but uses an advanced,
modern Deep Cyber-Violet & Emerald colorway for E-Commerce Reconciliation & AI.
"""

from __future__ import annotations

from typing import Any

# Canvas and surfaces (Deep Night Obsidian + Violet glass panels)
CANVAS_BG = "#080711"
PANEL_BG = "#0e0c1e"
CARD_BG = "#15122b"
CARD_HI = "#1d193b"
BORDER = "#2a2453"
GRID = "#1a1638"

# Text
TEXT_LIGHT = "#F3F0FF"
TEXT_MUTED = "#9D97BF"

# Brand and semantic colours: Vibrant Cyber Violet, Radiant Emerald, Amber, Crimson
ACCENT = "#8B5CF6"         # Electric Violet
ACCENT_SOFT = "#A78BFA"    # Soft Lavender
ACCENT_DEEP = "#6D28D9"    # Royal Deep Violet
POSITIVE = "#10B981"       # Emerald Realized Cash
WARNING = "#F59E0B"        # Amber Float/In-Transit
NEGATIVE = "#F43F5E"       # Rose/Crimson Leakage
CYAN = "#06B6D4"           # Cyan Accents
NEUTRAL = "#6B7280"

CSS = f"""
<style>
:root {{
    --canvas: {CANVAS_BG};
    --panel: {PANEL_BG};
    --card: {CARD_BG};
    --card-hi: {CARD_HI};
    --border: {BORDER};
    --grid: {GRID};
    --text: {TEXT_LIGHT};
    --muted: {TEXT_MUTED};
    --accent: {ACCENT};
    --accent-soft: {ACCENT_SOFT};
    --accent-deep: {ACCENT_DEEP};
    --positive: {POSITIVE};
    --warning: {WARNING};
    --negative: {NEGATIVE};
}}

html, body, [class*="css"] {{
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}}

.stApp {{
    background: radial-gradient(circle at 10% 10%, #130f2c 0%, {CANVAS_BG} 70%) !important;
    color: var(--text) !important;
}}

.block-container {{
    width: min(100%, 1680px) !important;
    max-width: 1680px !important;
    padding: 1.15rem 1.4rem 2.25rem 1.4rem !important;
}}

header[data-testid="stHeader"], [data-testid="stToolbar"] {{
    background: transparent !important;
}}

[data-testid="stDecoration"] {{
    display: none !important;
}}

/* Sidebar Design Alignment */
[data-testid="stSidebar"] {{
    background: {PANEL_BG} !important;
    border-right: 1px solid var(--border) !important;
}}

.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.35rem 0 1.25rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}}

.sidebar-brand-mark {{
    width: 2.35rem;
    height: 2.35rem;
    border-radius: 0.5rem;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-deep) 100%);
    color: #FFFFFF;
    font-size: 0.95rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.35);
}}

.sidebar-brand-name {{
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.15;
}}

.sidebar-brand-role {{
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.15rem;
}}

.sidebar-section {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: var(--muted);
    margin: 1.2rem 0 0.5rem 0;
}}

/* Topbar Design Alignment */
.metric-flow-topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.2rem;
    background: {CARD_BG};
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    margin-bottom: 1.35rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}}

.metric-flow-brand {{
    display: flex;
    align-items: center;
    gap: 0.85rem;
}}

.metric-flow-mark {{
    width: 2.1rem;
    height: 2.1rem;
    border-radius: 0.45rem;
    background: linear-gradient(135deg, var(--accent) 0%, #10B981 100%);
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.metric-flow-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
}}

.metric-flow-subtitle {{
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.15rem;
}}

.metric-flow-context {{
    display: flex;
    align-items: center;
    gap: 0.85rem;
}}

.metric-flow-period {{
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--accent-soft);
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 0.3rem 0.7rem;
    border-radius: 0.45rem;
}}

.metric-flow-avatar {{
    width: 2.1rem;
    height: 2.1rem;
    border-radius: 50%;
    background: var(--card-hi);
    border: 1px solid var(--border);
    color: var(--text);
    font-size: 0.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}}

/* 4-Column Exact KPI Row Alignment */
.kpi-grid {{
    display: grid;
    gap: 0.95rem;
    margin-bottom: 1.35rem;
}}

.metric-flow-kpi {{
    background: {CARD_BG};
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    padding: 1rem 1.15rem;
    position: relative;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    transition: transform 0.15s ease, border-color 0.15s ease;
}}

.metric-flow-kpi:hover {{
    transform: translateY(-2px);
    border-color: rgba(139, 92, 246, 0.5);
}}

.kpi-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 0.45rem;
}}

.kpi-value {{
    font-size: 1.85rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1.1;
    letter-spacing: -0.02em;
}}

.kpi-sub {{
    font-size: 0.74rem;
    color: var(--muted);
    margin-top: 0.45rem;
}}

.kpi-delta {{
    position: absolute;
    top: 0.95rem;
    right: 1.1rem;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.2rem 0.5rem;
    border-radius: 0.35rem;
}}

.kpi-delta.positive {{
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}}

.kpi-delta.warning {{
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}}

.kpi-delta.negative {{
    background: rgba(244, 63, 94, 0.15);
    color: #FB7185;
    border: 1px solid rgba(244, 63, 94, 0.3);
}}

/* Containers / Panels */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {CARD_BG} !important;
    border: 1px solid var(--border) !important;
    border-radius: 0.75rem !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
}}

.metric-flow-panel-title {{
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.15rem;
}}

.metric-flow-panel-sub {{
    font-size: 0.74rem;
    color: var(--muted);
    margin-bottom: 0.65rem;
}}
</style>
"""


def inject_theme() -> None:
    """Injects the cyber violet executive stylesheet."""
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def apply_plotly_theme(fig: Any, *, height: int = 285, title: str = "") -> Any:
    """Applies the executive slate styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=10 if not title else 30, b=15),
        font=dict(family="Inter, sans-serif", color=TEXT_LIGHT, size=11),
        xaxis=dict(
            gridcolor=GRID,
            zerolinecolor=GRID,
            linecolor=BORDER,
            tickfont=dict(color=TEXT_MUTED, size=10),
        ),
        yaxis=dict(
            gridcolor=GRID,
            zerolinecolor=GRID,
            linecolor=BORDER,
            tickfont=dict(color=TEXT_MUTED, size=10),
        ),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=12, color=TEXT_LIGHT)))
    return fig
