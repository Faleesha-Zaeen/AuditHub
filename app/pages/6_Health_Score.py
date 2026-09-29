"""
AuditHub Page - Health Score
==============================

A weighted composite score across five quality dimensions, with every
deduction itemised.
"""

# Ensure the project root is importable when Streamlit runs this file as a
# standalone script (it puts the page's own directory on sys.path, not the root).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import ui
from app.components.session import (
    data_source_caption,
    get_original_df,
    has_repairs,
    require_dataset,
    target_selector,
)
from src.health.calculator import HealthScoreCalculator

ui.page_setup("Health Score", icon="◉")

ui.page_header(
    "Score the dataset",
    "One number from five weighted dimensions. Every point deducted is "
    "explained, so the score is auditable rather than opaque.",
    eyebrow="Step 6 · Health",
)

df = require_dataset()
data_source_caption()

control_left, control_right = st.columns([3, 1])
with control_left:
    target = target_selector(df, label="Target column (optional)")
with control_right:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run = st.button("Calculate score", type="primary", width="stretch")

if run:
    with st.spinner("Scoring dimensions…"):
        try:
            calculator = HealthScoreCalculator()
            st.session_state["health_report"] = calculator.calculate(
                df, target_column=target
            ).to_dict()

            # With repairs applied, score the original too so the improvement
            # is visible rather than merely asserted.
            if has_repairs():
                st.session_state["health_report_before"] = calculator.calculate(
                    get_original_df(), target_column=target
                ).to_dict()
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")

report = st.session_state.get("health_report")
if not report:
    ui.divider()
    ui.empty_state(
        "No health score yet.",
        "Press <strong>Calculate score</strong> to evaluate the dataset.",
    )
    st.stop()

ui.divider()

score = float(report.get("overall_score", 0.0))
grade = report.get("grade", "F")
dimensions = report.get("dimensions", {})
before = st.session_state.get("health_report_before")

# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

grade_column, score_column, delta_column = st.columns([1, 2, 2])

with grade_column:
    st.markdown(ui.grade_pill(grade), unsafe_allow_html=True)

with score_column:
    st.markdown(
        f'<div class="ah-stat-label">Overall score</div>'
        f'<div style="font-family:var(--font-mono);font-size:2.6rem;font-weight:600;'
        f'color:var(--text);line-height:1.1;">{score:.1f}'
        f'<span style="font-size:1rem;color:var(--text-muted);"> / 100</span></div>',
        unsafe_allow_html=True,
    )

with delta_column:
    if before:
        previous = float(before.get("overall_score", 0.0))
        change = score - previous
        arrow = "▲" if change >= 0 else "▼"
        colour = "var(--ok)" if change >= 0 else "var(--danger)"
        st.markdown(
            f'<div class="ah-stat-label">Since repair</div>'
            f'<div style="font-family:var(--font-mono);font-size:1.5rem;color:{colour};">'
            f'{arrow} {change:+.1f}</div>'
            f'<div style="font-size:0.82rem;color:var(--text-muted);margin-top:4px;">'
            f'was {before.get("grade")} ({previous:.1f})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ah-stat-label">Since repair</div>'
            '<div style="font-size:0.85rem;color:var(--text-muted);margin-top:8px;">'
            'Apply repairs, then re-score to see the improvement.</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

ui.section("Dimensions")

if dimensions:
    names = [n.title() for n in dimensions]
    values = [float(d.get("score", 0.0)) for d in dimensions.values()]

    radar = go.Figure()
    if before:
        radar.add_trace(go.Scatterpolar(
            r=[float(before.get("dimensions", {}).get(n, {}).get("score", 0.0))
               for n in dimensions],
            theta=names, fill="toself", name="before repair",
            line=dict(color=ui.PLOT_COLORS["muted"]),
            fillcolor="rgba(107,115,130,0.12)",
        ))
    radar.add_trace(go.Scatterpolar(
        r=values, theta=names, fill="toself", name="current",
        line=dict(color=ui.PLOT_COLORS["accent"]),
        fillcolor="rgba(99,102,241,0.18)",
    ))
    radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#232a37",
                            tickfont=dict(size=10)),
            angularaxis=dict(gridcolor="#232a37"),
        ),
    )

    radar_column, table_column = st.columns([1, 1])
    with radar_column:
        ui.plot(radar, height=360, show_legend=bool(before))
    with table_column:
        for name, dimension in dimensions.items():
            value = float(dimension.get("score", 0.0))
            colour = (
                "var(--ok)" if value >= 90
                else "var(--info)" if value >= 75
                else "var(--warn)" if value >= 50
                else "var(--danger)"
            )
            st.markdown(
                f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;
                                font-size:0.84rem;margin-bottom:4px;">
                        <span style="color:var(--text-secondary);">{name.title()}</span>
                        <span style="font-family:var(--font-mono);color:{colour};">
                            {value:.1f}</span>
                    </div>
                    <div style="background:var(--surface-2);border-radius:3px;height:6px;">
                        <div style="width:{max(0.0, min(value, 100.0))}%;background:{colour};
                                    height:6px;border-radius:3px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Deductions
# ---------------------------------------------------------------------------

explanations = report.get("explanations", [])
ui.divider()
ui.section("Why the score is what it is", f"{len(explanations)} deduction(s) applied.")

if not explanations:
    st.success("No deductions — a perfect score across every dimension.")
else:
    for name, dimension in dimensions.items():
        deductions = dimension.get("deductions", [])
        if not deductions:
            continue
        with st.expander(f"{name.title()} — {len(deductions)} deduction(s)", expanded=False):
            for deduction in deductions:
                st.markdown(
                    f'<div style="font-size:0.85rem;color:var(--text-secondary);'
                    f'padding:6px 0;border-bottom:1px solid var(--border);">{deduction}</div>',
                    unsafe_allow_html=True,
                )
