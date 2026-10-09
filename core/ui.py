"""Shared UI helpers: CSS theme, KPI cards, badges, chart styling."""
from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

from .config import COLORS, RISK_COLORS

C = COLORS

CSS = f"""
<style>
:root {{
  --success:{C['success']}; --warning:{C['warning']}; --danger:{C['danger']}; --info:{C['info']};
  --navy:{C['navy']}; --muted:{C['muted']}; --border:{C['border']};
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ color: var(--navy); letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: {C['navy']}; }}
[data-testid="stSidebar"] * {{ color: #E2E8F0; }}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{ background: rgba(37,99,235,0.35); }}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {{ color: {C['navy']}; }}
[data-testid="stSidebar"] button {{ background: transparent; border: 1px solid #475569; }}
[data-testid="stSidebar"] code {{ background: #1E293B; color: #93C5FD; }}
.ft-header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:16px;
  border-bottom:1px solid var(--border); padding-bottom:12px; margin-bottom:20px; flex-wrap:wrap; }}
.ft-header h1 {{ font-size: 1.9rem; margin: 0; padding: 0; }}
.ft-header p {{ color: var(--muted); margin: 4px 0 0 0; }}
.ft-scope {{ font-size: .8rem; font-weight:600; color: var(--info); background:#EFF4FF;
  border:1px solid #C7D7FE; padding:4px 10px; border-radius:999px; white-space:nowrap; }}
.ft-kpi {{ background:#FFFFFF; border:1px solid var(--border); border-left:5px solid var(--info);
  border-radius:10px; padding:14px 16px; height:100%; }}
.ft-kpi .label {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:600; }}
.ft-kpi .value {{ font-size:1.65rem; font-weight:700; color:var(--navy); line-height:1.25; margin-top:2px; }}
.ft-kpi .sub {{ font-size:.8rem; color:var(--muted); margin-top:2px; }}
.ft-badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:700;
  color:#FFFFFF; letter-spacing:.02em; }}
.ft-verdict {{ border-radius:12px; padding:18px 22px; color:#FFFFFF; margin-bottom:12px; }}
.ft-verdict .t {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; opacity:.9; font-weight:600; }}
.ft-verdict .v {{ font-size:1.5rem; font-weight:700; margin-top:4px; }}
.ft-verdict .s {{ font-size:.95rem; opacity:.95; margin-top:4px; }}
.ft-note {{ background:#EFF4FF; border:1px solid #C7D7FE; color:{C['navy']}; border-radius:10px; padding:12px 14px; font-size:.9rem; }}
.ft-confidential {{ background:#FEF2F2; border:1px solid #FECACA; color:{C['danger']}; border-radius:10px;
  padding:10px 14px; font-size:.85rem; font-weight:600; }}
.ft-login-brand {{ background:{C['navy']}; color:#F1F5F9; border-radius:16px; padding:36px; height:100%; }}
.ft-login-brand h1 {{ color:#FFFFFF; font-size:2.4rem; margin:0; }}
.ft-login-brand p {{ color:#CBD5E1; }}
.ft-login-brand .eyebrow {{ color:#F59E0B; font-weight:700; letter-spacing:.12em; font-size:.78rem; text-transform:uppercase; }}
.ft-chip {{ display:inline-block; background:#F1F5F9; border:1px solid var(--border); border-radius:6px;
  padding:1px 8px; margin:2px 4px 2px 0; font-size:.8rem; color:{C['slate']}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", scope: str = "") -> None:
    scope_html = f'<span class="ft-scope">🔒 {escape(scope)}</span>' if scope else ""
    st.markdown(
        f'<div class="ft-header"><div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>{scope_html}</div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value, sub: str = "", color: str = "info") -> None:
    col = C.get(color, color)
    st.markdown(
        f'<div class="ft-kpi" style="border-left-color:{col}"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(str(value))}</div><div class="sub">{escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            kpi(*item)


def badge(text: str, color: str) -> str:
    return f'<span class="ft-badge" style="background:{C.get(color, color)}">{escape(text)}</span>'


def risk_badge(band: str) -> str:
    return badge(f"{band} risk", RISK_COLORS.get(band, C["muted"]))


def verdict_box(band: str, verdict: str, sub: str) -> None:
    col = RISK_COLORS.get(band, C["muted"])
    st.markdown(
        f'<div class="ft-verdict" style="background:{col}"><div class="t">AML verdict · {escape(band)} risk</div>'
        f'<div class="v">{escape(verdict)}</div><div class="s">{escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="ft-note">{escape(text)}</div>', unsafe_allow_html=True)


def confidential(text: str) -> None:
    st.markdown(f'<div class="ft-confidential">🔒 {escape(text)}</div>', unsafe_allow_html=True)


def chips(items) -> str:
    return "".join(f'<span class="ft-chip">{escape(str(i))}</span>' for i in items)


def style_fig(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, sans-serif", color=C["slate"], size=13),
        title=dict(text=fig.layout.title.text or "", font=dict(size=15, color=C["navy"]), x=0, xanchor="left"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"])
    fig.update_yaxes(gridcolor="#EEF2F6", linecolor=C["border"])
    return fig


def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:,.1f}K"
    return f"{x:,.0f}"
