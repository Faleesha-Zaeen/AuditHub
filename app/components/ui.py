"""
AuditHub - UI Component Library
================================

Shared render helpers so every page speaks the same visual language.

Pages used to hand-roll their own headers, emoji titles and inline HTML, which
is why the app looked assembled rather than designed. Everything visual now
goes through these functions, and the styling lives in
``app/assets/style.css`` -- change a colour there and it changes everywhere.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import streamlit as st

from src.utils.constants import APP_NAME, APP_VERSION

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_CSS_PATH = _ASSETS / "style.css"
_FONTS_PATH = _ASSETS / "fonts.css"
_POLISH_PATH = _ASSETS / "polish.css"

# Status token -> badge modifier class. Keys cover the vocabularies used by the
# drift, quality and version-comparison engines so pages never map them again.
_STATUS_CLASS: Dict[str, str] = {
    "STABLE": "ok", "OK": "ok", "PASS": "ok", "PASSED": "ok", "COMPLETED": "ok",
    "WARNING": "warn", "WARN": "warn", "PARTIAL": "warn",
    "CRITICAL": "danger", "ERROR": "danger", "FAILED": "danger", "FAIL": "danger",
    "REMOVED_COLUMN": "danger", "TYPE_CHANGED": "warn",
    "INFO": "info", "NEW_COLUMN": "info",
    "NOT_COMPARABLE": "neutral", "SKIPPED": "neutral", "PENDING": "neutral",
}

# Chart palette, matched to the CSS custom properties.
PLOT_COLORS = {
    "accent": "#6366f1",
    "ok": "#10b981",
    "warn": "#f59e0b",
    "danger": "#f43f5e",
    "info": "#38bdf8",
    "muted": "#6b7382",
    "reference": "#5a6bd8",
    "current": "#f59e0b",
}

STATUS_COLORS = {
    "STABLE": PLOT_COLORS["ok"],
    "WARNING": PLOT_COLORS["warn"],
    "CRITICAL": PLOT_COLORS["danger"],
    "NEW_COLUMN": PLOT_COLORS["info"],
    "REMOVED_COLUMN": PLOT_COLORS["danger"],
    "TYPE_CHANGED": "#a855f7",
    "NOT_COMPARABLE": PLOT_COLORS["muted"],
}


# ---------------------------------------------------------------------------
# Page scaffolding
# ---------------------------------------------------------------------------


def page_setup(title: str, icon: str = "▪", layout: str = "wide") -> None:
    """Configure the page and inject the stylesheet.

    Call once at the top of every page, before any other Streamlit call.

    Under ``st.navigation`` the router has already configured the page, and a
    second call raises. It is swallowed so the same page file works both as a
    routed page and when run on its own (which is how the tests drive them).
    """
    try:
        st.set_page_config(
            page_title=f"{APP_NAME} · {title}",
            page_icon=icon,
            layout=layout,
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    inject_css()
    render_brand()


_STYLE_FILES = (_FONTS_PATH, _CSS_PATH, _POLISH_PATH)


@lru_cache(maxsize=4)
def _read_stylesheets(_fingerprint: tuple) -> str:
    """Read and concatenate the stylesheets.

    The embedded fonts come first: Streamlit strips ``@import`` from injected
    CSS, so the faces have to arrive as inline ``@font-face`` data or every
    heading silently falls back to the stock face.

    ``_fingerprint`` is unused except as the cache key -- see
    :func:`_stylesheet`.
    """
    parts = []
    for path in _STYLE_FILES:
        try:
            parts.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(parts)


def _stylesheet() -> str:
    """Return the combined stylesheet, re-reading it when a file changes.

    Caching purely on the function would pin the CSS for the life of the
    server process, so editing a stylesheet would appear to do nothing until
    the app was restarted. Keying on modification times keeps the read cheap
    while still picking up edits.
    """
    fingerprint = tuple(
        (path.stat().st_mtime_ns if path.exists() else 0) for path in _STYLE_FILES
    )
    return _read_stylesheets(fingerprint)


def inject_css() -> None:
    """Load the design system stylesheet into the page."""
    css = _stylesheet()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_brand() -> None:
    """Render the product mark at the top of the sidebar."""
    st.sidebar.markdown(
        f"""
        <div class="ah-brand">
            <div class="ah-brand-name">
                <span class="ah-brand-mark"></span>{APP_NAME}
            </div>
            <div class="ah-brand-tagline">Data Quality &amp; MLOps · v{APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    """Render a page title block.

    Replaces the emoji-prefixed ``st.markdown("# ...")`` headers: a small
    uppercase eyebrow for section context, then the title, then one line of
    explanation.
    """
    parts = ['<div class="ah-header">']
    if eyebrow:
        parts.append(f'<span class="ah-eyebrow">{html.escape(eyebrow)}</span>')
    parts.append(f'<h1 class="ah-title">{html.escape(title)}</h1>')
    if subtitle:
        parts.append(f'<p class="ah-subtitle">{html.escape(subtitle)}</p>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(title: str, description: str = "") -> None:
    """Render a section heading within a page."""
    st.markdown(f"<h2>{html.escape(title)}</h2>", unsafe_allow_html=True)
    if description:
        st.markdown(
            f'<p class="ah-subtitle" style="margin-top:-6px;">{html.escape(description)}</p>',
            unsafe_allow_html=True,
        )


def divider() -> None:
    """Render a soft horizontal rule."""
    st.markdown('<hr class="ah-rule">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------


def status_class(status: str) -> str:
    """Map a status token to its badge modifier."""
    return _STATUS_CLASS.get(str(status).upper(), "neutral")


def badge(text: str, status: Optional[str] = None) -> str:
    """Return badge HTML. Use inside another markdown block."""
    modifier = status_class(status if status is not None else text)
    return f'<span class="ah-badge ah-badge-{modifier}">{html.escape(str(text))}</span>'


def render_badge(text: str, status: Optional[str] = None) -> None:
    """Render a status badge on its own."""
    st.markdown(badge(text, status), unsafe_allow_html=True)


def grade_pill(grade: str) -> str:
    """Return HTML for a large letter-grade pill."""
    letter = str(grade).upper()[:1] or "F"
    modifier = letter.lower() if letter in "ABCDF" else "f"
    return f'<div class="ah-grade ah-grade-{modifier}">{html.escape(letter)}</div>'


# ---------------------------------------------------------------------------
# Data display
# ---------------------------------------------------------------------------


# A value made only of digits, separators and units is rendered as a figure;
# anything else is prose and needs the smaller, proportional treatment.
_NUMERIC_VALUE = re.compile(r"^[\d.,+\-%/×x\s]+$")


def stat(label: str, value: Any, sub: str = "") -> str:
    """Return HTML for a single stat tile.

    Long text values (a method name, a task type) are rendered smaller and in
    the sans face. Left in the large monospace figure style they overflow the
    tile and break mid-word -- "Classificat / ion".
    """
    text = str(value)
    # Short tokens (a grade letter, a dash placeholder) stay in the large
    # figure style so a row of tiles keeps a consistent visual weight.
    is_figure = len(text) <= 3 or (bool(_NUMERIC_VALUE.match(text)) and len(text) <= 12)

    if is_figure:
        value_class = "ah-stat-value"
    else:
        # A single long word cannot wrap, so it would be broken mid-word
        # ("Classificat / ion"). Those get a step smaller again.
        longest_word = max((len(w) for w in text.split()), default=0)
        value_class = "ah-stat-value ah-stat-value-text"
        if longest_word > 12:
            value_class += " ah-stat-value-compact"

    parts = [
        '<div class="ah-stat">',
        f'<span class="ah-stat-label">{html.escape(str(label))}</span>',
        f'<div class="{value_class}">{html.escape(text)}</div>',
    ]
    if sub:
        parts.append(f'<div class="ah-stat-sub">{sub}</div>')
    parts.append("</div>")
    return "".join(parts)


def stat_row(items: Sequence[Dict[str, Any]]) -> None:
    """Render a row of stat tiles.

    Each item takes ``label``, ``value`` and optionally ``sub`` (which may
    contain a badge, so it is not escaped).
    """
    if not items:
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            st.markdown(
                stat(item.get("label", ""), item.get("value", ""), item.get("sub", "")),
                unsafe_allow_html=True,
            )


def card(title: str = "", body: str = "") -> None:
    """Render a bordered card containing pre-rendered HTML."""
    inner = f'<div class="ah-card-title">{html.escape(title)}</div>' if title else ""
    st.markdown(f'<div class="ah-card">{inner}{body}</div>', unsafe_allow_html=True)


def _arrow_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame Streamlit can serialise.

    Streamlit renders tables through Arrow, which raises on an object column
    holding more than one Python type -- a "value" column spanning ints,
    strings and timestamps, say. Rather than let that take down the whole page,
    such columns are rendered as text.
    """
    problem_columns = []
    for column in frame.columns:
        series = frame[column]
        if series.dtype != object:
            continue
        kinds = {type(v) for v in series.dropna().head(200)}
        if len(kinds) > 1:
            problem_columns.append(column)

    if not problem_columns:
        return frame

    safe = frame.copy()
    for column in problem_columns:
        safe[column] = safe[column].map(lambda v: "" if pd.isna(v) else str(v))
    return safe


def dataframe(frame: pd.DataFrame, height: Optional[int] = None) -> None:
    """Render a DataFrame with the app's standard settings.

    ``height`` is omitted rather than passed as ``None``: Streamlit validates
    the argument strictly and rejects an explicit ``None``.
    """
    frame = _arrow_safe(frame)
    if height is None:
        st.dataframe(frame, width="stretch", hide_index=True)
    else:
        st.dataframe(frame, width="stretch", hide_index=True, height=int(height))


def empty_state(message: str, hint: str = "") -> None:
    """Render a calm placeholder instead of a bare warning.

    Shown when a page has nothing to display yet -- a blank screen reads as a
    bug, whereas this reads as a next step.
    """
    body = (
        f'<div style="text-align:center;padding:44px 20px;">'
        f'<div style="font-size:0.95rem;color:var(--text-secondary);margin-bottom:6px;">'
        f'{html.escape(message)}</div>'
    )
    if hint:
        body += (
            f'<div style="font-size:0.84rem;color:var(--text-muted);">{hint}</div>'
        )
    body += "</div>"
    st.markdown(f'<div class="ah-card">{body}</div>', unsafe_allow_html=True)


def flow(steps: Sequence[str], active: Optional[str] = None) -> None:
    """Render a left-to-right chain of stage chips."""
    chips = []
    for index, step in enumerate(steps):
        if index:
            chips.append('<span class="ah-flow-arrow">&rarr;</span>')
        css = "ah-chip ah-chip-accent" if step == active else "ah-chip"
        chips.append(f'<span class="{css}">{html.escape(str(step))}</span>')
    st.markdown(f'<div class="ah-flow">{"".join(chips)}</div>', unsafe_allow_html=True)


def timeline(events: Iterable[Dict[str, str]]) -> None:
    """Render a vertical timeline.

    Each event takes ``text`` and an optional ``stage`` label.
    """
    items = []
    for event in events:
        stage = event.get("stage", "")
        stage_html = (
            f'<div class="ah-timeline-stage">{html.escape(stage)}</div>' if stage else ""
        )
        items.append(
            f'<div class="ah-timeline-item">'
            f'<div class="ah-timeline-text">{html.escape(event.get("text", ""))}</div>'
            f"{stage_html}</div>"
        )
    st.markdown(f'<div class="ah-timeline">{"".join(items)}</div>', unsafe_allow_html=True)


def contribution_bars(
    contributions: Sequence[Dict[str, Any]],
    positive_label: str = "increases",
) -> None:
    """Render signed contribution bars diverging from a centre axis.

    Positive contributions extend right in the danger colour, negative ones
    left in the success colour, so the direction of an effect is readable
    without checking the sign.
    """
    if not contributions:
        return
    peak = max((abs(float(c.get("contribution", 0.0))) for c in contributions), default=1.0) or 1.0

    rows = []
    for item in contributions:
        value = float(item.get("contribution", 0.0))
        width = min(abs(value) / peak * 50.0, 50.0)
        css = "ah-contrib-pos" if value >= 0 else "ah-contrib-neg"
        style = f"width:{width:.1f}%;"
        name = html.escape(str(item.get("feature", "")))
        shown = html.escape(str(item.get("value", "")))[:18]
        rows.append(
            f'<div class="ah-contrib">'
            f'<div class="ah-contrib-name" title="{name}">{name}</div>'
            f'<div class="ah-contrib-track">'
            f'<div class="ah-contrib-axis"></div>'
            f'<div class="ah-contrib-bar {css}" style="{style}"></div>'
            f"</div>"
            f'<div class="ah-contrib-value">{value:+.4f}</div>'
            f'<div class="ah-contrib-value" style="width:120px;">= {shown}</div>'
            f"</div>"
        )

    legend = (
        f'<div style="display:flex;gap:18px;margin-top:10px;font-size:0.74rem;'
        f'color:var(--text-muted);">'
        f'<span><span style="display:inline-block;width:9px;height:9px;border-radius:2px;'
        f'background:var(--danger);margin-right:5px;"></span>{html.escape(positive_label)} the prediction</span>'
        f'<span><span style="display:inline-block;width:9px;height:9px;border-radius:2px;'
        f'background:var(--ok);margin-right:5px;"></span>argues against it</span>'
        f"</div>"
    )
    st.markdown("".join(rows) + legend, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def style_figure(fig, height: int = 360, show_legend: bool = True):
    """Apply the app's chart styling to a Plotly figure.

    Charts otherwise arrive with Plotly's default white background and blue
    palette, which is the single most obvious sign of an unstyled dashboard.
    """
    # Only style the title when the figure actually has one. Applying a title
    # dict to an untitled figure makes Plotly render the string "undefined".
    has_title = bool(getattr(fig.layout.title, "text", None))
    if has_title:
        fig.update_layout(
            title=dict(font=dict(size=13.5, color="#e6e8ee"), x=0, xanchor="left")
        )

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#a6adbb"),
        margin=dict(l=10, r=16, t=44 if has_title else 20, b=10),
        showlegend=show_legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="#161b25", bordercolor="#2f3846",
            font=dict(family="Inter, sans-serif", size=12, color="#e6e8ee"),
        ),
    )
    fig.update_xaxes(gridcolor="#232a37", zerolinecolor="#2f3846", linecolor="#232a37")
    fig.update_yaxes(gridcolor="#232a37", zerolinecolor="#2f3846", linecolor="#232a37")
    return fig


def plot(fig, height: int = 360, show_legend: bool = True) -> None:
    """Style and render a Plotly figure."""
    st.plotly_chart(style_figure(fig, height, show_legend), width="stretch")


__all__ = [
    "PLOT_COLORS",
    "STATUS_COLORS",
    "badge",
    "card",
    "contribution_bars",
    "dataframe",
    "divider",
    "empty_state",
    "flow",
    "grade_pill",
    "inject_css",
    "page_header",
    "page_setup",
    "plot",
    "render_badge",
    "render_brand",
    "section",
    "stat",
    "stat_row",
    "status_class",
    "style_figure",
    "timeline",
]
