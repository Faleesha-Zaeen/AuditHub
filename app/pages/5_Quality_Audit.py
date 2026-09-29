"""
AuditHub Page - Quality Audit
===============================

Rule-based findings across completeness, uniqueness, accuracy, consistency and
reliability, including target leakage.
"""

# Ensure the project root is importable when Streamlit runs this file as a
# standalone script (it puts the page's own directory on sys.path, not the root).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components import ui
from app.components.session import (
    data_source_caption,
    get_active_summary,
    require_dataset,
    target_selector,
)
from src.quality.auditor import DatasetAuditor

ui.page_setup("Quality Audit", icon="◈")

ui.page_header(
    "Audit the data",
    "Eight rule-based checks graded by severity, including outliers, class "
    "imbalance, multicollinearity and target leakage.",
    eyebrow="Step 5 · Quality",
)

df = require_dataset()
data_source_caption()
summary = get_active_summary()

control_left, control_right = st.columns([3, 1])
with control_left:
    target = target_selector(df, label="Target column (for leakage and imbalance checks)")
with control_right:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run = st.button("Run audit", type="primary", width="stretch")

if run:
    with st.spinner("Analysing distributions and anomalies…"):
        try:
            report = DatasetAuditor().audit(df, target_column=target)
            st.session_state["quality_report"] = report.to_dict()
        except Exception as exc:
            st.error(f"Audit failed: {exc}")

report = st.session_state.get("quality_report")
if not report:
    ui.divider()
    ui.empty_state("No audit run yet.", "Press <strong>Run audit</strong> to begin.")
    st.stop()

ui.divider()

counts = report.get("summary", {})
findings = report.get("findings", [])

ui.stat_row([
    {"label": "Errors", "value": counts.get("ERROR", 0),
     "sub": ui.badge("critical", "CRITICAL") if counts.get("ERROR") else "none"},
    {"label": "Warnings", "value": counts.get("WARNING", 0),
     "sub": ui.badge("review", "WARNING") if counts.get("WARNING") else "none"},
    {"label": "Notes", "value": counts.get("INFO", 0)},
    {"label": "Findings", "value": len(findings)},
])

if not findings:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.success("No anomalies detected — this dataset is in good shape.")
else:
    ui.section("Findings", "Most severe first.")

    dimensions = sorted({f.get("dimension", "Other") for f in findings})
    severities = ["ERROR", "WARNING", "INFO"]

    filter_left, filter_right = st.columns(2)
    with filter_left:
        chosen_severity = st.multiselect("Severity", severities, default=severities)
    with filter_right:
        chosen_dimension = st.multiselect("Dimension", dimensions, default=dimensions)

    order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    visible = [
        f for f in findings
        if f.get("severity") in chosen_severity and f.get("dimension") in chosen_dimension
    ]
    visible.sort(key=lambda f: order.get(f.get("severity", "INFO"), 3))

    if not visible:
        ui.empty_state("No findings match the current filters.")

    for finding in visible:
        severity = finding.get("severity", "INFO")
        column = finding.get("column") or "dataset-wide"
        st.markdown(
            f"""
            <div class="ah-card" style="padding:14px 18px;margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                    {ui.badge(severity, severity)}
                    <span style="font-size:0.8rem;color:var(--text-muted);
                                 text-transform:uppercase;letter-spacing:0.07em;">
                        {finding.get('dimension', '')}</span>
                    <code>{column}</code>
                </div>
                <div style="font-size:0.88rem;color:var(--text-secondary);">
                    {finding.get('message', '')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

numeric_columns = [
    c.name for c in summary.column_stats
    if c.is_numeric and c.inferred_type != "constant" and c.name in df.columns
]

if len(numeric_columns) > 1:
    ui.divider()
    ui.section("Feature correlation", "Strong pairs are candidates for redundancy or leakage.")
    try:
        correlation = df[numeric_columns].corr()
        figure = px.imshow(
            correlation, text_auto=".2f", aspect="auto",
            color_continuous_scale=["#f43f5e", "#161b25", "#38bdf8"],
            zmin=-1, zmax=1,
        )
        ui.plot(figure, height=min(620, 90 + 40 * len(numeric_columns)), show_legend=False)
    except Exception as exc:
        st.warning(f"Could not compute the correlation matrix: {exc}")
