"""Small design system: one look for every page (colours, cards, callouts, KPI tiles).

All text passed in is HTML-escaped. Markup is emitted on single lines (no blank lines / indentation) so Markdown
never mistakes it for a code block.
"""
from __future__ import annotations

from html import escape

import streamlit as st

NAVY, BLUE, TEAL, SOFT, BORDER = "#0f2a4a", "#1f5fbf", "#0e9aa7", "#f3f6fb", "#dfe6f1"

CSS = f"""<style>
.block-container {{padding-top: 1.6rem; max-width: 1280px;}}
h1, h2, h3 {{color: {NAVY}; letter-spacing: -0.01em;}}
.wx-hero {{background: linear-gradient(120deg, {NAVY} 0%, #17467c 55%, {TEAL} 130%); color: #fff; padding: 26px 30px; border-radius: 16px; margin-bottom: 18px;}}
.wx-hero h1 {{color: #fff; margin: 0 0 6px 0; font-size: 2rem; padding: 0;}}
.wx-hero p {{color: #dce9f7; margin: 0; font-size: 1.02rem; line-height: 1.5;}}
.wx-hero .wx-badge {{display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3); color: #fff; padding: 2px 10px; border-radius: 999px; font-size: .78rem; margin-bottom: 10px;}}
.wx-pagehead {{border-left: 5px solid {BLUE}; padding: 4px 0 4px 14px; margin: 0 0 14px 0;}}
.wx-pagehead h1 {{margin: 0; font-size: 1.7rem; padding: 0;}}
.wx-pagehead p {{margin: 4px 0 0 0; color: #4a5b6d;}}
.wx-kpis {{display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 8px 0 14px 0;}}
.wx-kpi {{background: #fff; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(15,42,74,.06);}}
.wx-kpi-label {{font-size: .8rem; color: #5a6b7d; text-transform: uppercase; letter-spacing: .04em;}}
.wx-kpi-value {{font-size: 1.75rem; font-weight: 700; color: {NAVY}; line-height: 1.25;}}
.wx-kpi-sub {{font-size: .82rem; color: #6b7c8e;}}
.wx-card {{background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(15,42,74,.06); height: 100%;}}
.wx-card h4 {{margin: 0 0 6px 0; color: {NAVY}; font-size: 1.05rem;}}
.wx-card p {{margin: 0; color: #3d4e60; font-size: .93rem; line-height: 1.5;}}
.wx-num {{display: inline-block; background: {BLUE}; color: #fff; width: 28px; height: 28px; line-height: 28px; text-align: center; border-radius: 50%; font-weight: 700; margin-bottom: 8px;}}
.wx-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 8px 0 16px 0;}}
.wx-callout {{border-radius: 12px; padding: 12px 16px; margin: 8px 0 14px 0; border: 1px solid {BORDER}; background: {SOFT}; color: #243546; line-height: 1.5;}}
.wx-callout b.wx-t {{display: block; margin-bottom: 2px; color: {NAVY};}}
.wx-callout.tip {{border-left: 5px solid {TEAL};}}
.wx-callout.info {{border-left: 5px solid {BLUE};}}
.wx-callout.warn {{border-left: 5px solid #e0961f; background: #fff7ea;}}
.wx-callout.good {{border-left: 5px solid #2e9e5b; background: #eefaf2;}}
.wx-explain {{display: grid; grid-template-columns: 1fr 1fr; gap: 14px; background: {SOFT}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px; margin: 4px 0 12px 0;}}
.wx-explain div b {{color: {NAVY}; display: block; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 2px;}}
.wx-explain div {{font-size: .92rem; color: #2f4156; line-height: 1.45;}}
.wx-chips {{display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 12px 0;}}
.wx-chip {{background: {SOFT}; border: 1px solid {BORDER}; color: #2b3d52; padding: 3px 11px; border-radius: 999px; font-size: .82rem;}}
.wx-chip.good {{background: #eefaf2; border-color: #bfe5cd; color: #1d6b3f;}}
.wx-chip.warn {{background: #fff7ea; border-color: #f0d6a6; color: #8a5a07;}}
.wx-brand {{font-weight: 700; color: {NAVY}; font-size: 1.05rem;}}
.wx-foot {{color: #6b7c8e; font-size: .78rem; line-height: 1.4;}}
@media (max-width: 800px) {{ .wx-explain {{grid-template-columns: 1fr;}} }}
</style>"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _e(x) -> str:
    return escape(str(x))


def hero(title: str, subtitle: str, badge: str | None = None) -> None:
    b = f'<span class="wx-badge">{_e(badge)}</span><br>' if badge else ""
    st.markdown(f'<div class="wx-hero">{b}<h1>{_e(title)}</h1><p>{_e(subtitle)}</p></div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="wx-pagehead"><h1>{_e(title)}</h1><p>{_e(subtitle)}</p></div>', unsafe_allow_html=True)


def kpi_html(items: list[tuple]) -> str:
    """items: (label, value, sub[, tooltip])."""
    tiles = []
    for it in items:
        label, value, sub = it[0], it[1], it[2] if len(it) > 2 else ""
        tip = f' title="{_e(it[3])}"' if len(it) > 3 and it[3] else ""
        tiles.append(f'<div class="wx-kpi"{tip}><div class="wx-kpi-label">{_e(label)}</div><div class="wx-kpi-value">{_e(value)}</div><div class="wx-kpi-sub">{_e(sub)}</div></div>')
    return f'<div class="wx-kpis">{"".join(tiles)}</div>'


def kpis(items: list[tuple]) -> None:
    st.markdown(kpi_html(items), unsafe_allow_html=True)


def callout(kind: str, title: str, body: str) -> None:
    """kind: info | tip | warn | good. `body` is plain text."""
    t = f'<b class="wx-t">{_e(title)}</b>' if title else ""
    st.markdown(f'<div class="wx-callout {kind}">{t}{_e(body)}</div>', unsafe_allow_html=True)


def explainer(what: str, how: str) -> None:
    st.markdown(f'<div class="wx-explain"><div><b>What you are seeing</b>{_e(what)}</div><div><b>How to read it</b>{_e(how)}</div></div>', unsafe_allow_html=True)


def chips(items: list[tuple]) -> None:
    """items: (text, kind) with kind '' | 'good' | 'warn'."""
    html = "".join(f'<span class="wx-chip {k}">{_e(t)}</span>' for t, k in items)
    st.markdown(f'<div class="wx-chips">{html}</div>', unsafe_allow_html=True)


def cards(items: list[tuple], numbered: bool = False, cols: int | None = None) -> None:
    """items: (title, text). cols fixes the number of columns (otherwise the grid adapts to the width)."""
    out = []
    for i, (t, p) in enumerate(items, 1):
        n = f'<div class="wx-num">{i}</div>' if numbered else ""
        out.append(f'<div class="wx-card">{n}<h4>{_e(t)}</h4><p>{_e(p)}</p></div>')
    style = f' style="grid-template-columns:repeat({cols},1fr)"' if cols else ""
    st.markdown(f'<div class="wx-grid"{style}>{"".join(out)}</div>', unsafe_allow_html=True)


def callout_list(kind: str, title: str, items: list[str]) -> None:
    lis = "".join(f"<li>{_e(i)}</li>" for i in items)
    st.markdown(f'<div class="wx-callout {kind}"><b class="wx-t">{_e(title)}</b><ul style="margin:4px 0 0 18px;padding:0">{lis}</ul></div>', unsafe_allow_html=True)
