"""
AuditHub Page - Profiling
===========================

Generates a full ydata-profiling report for the active dataset.
"""

# Ensure the project root is importable when Streamlit runs this file as a
# standalone script (it puts the page's own directory on sys.path, not the root).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pathlib import Path

import streamlit as st

from app.components import ui
from app.components.session import data_source_caption, require_dataset
from src.profiling.profiler import DatasetProfiler

ui.page_setup("Profiling", icon="◧")

ui.page_header(
    "Profile every variable",
    "Distributions, correlations, interactions and missing-value patterns, "
    "rendered as a standalone report you can share.",
    eyebrow="Step 4 · Profiling",
)

df = require_dataset()
data_source_caption()
name = st.session_state.get("dataset_name", "dataset")

control_left, control_right = st.columns([3, 1])
with control_left:
    minimal = st.checkbox(
        "Fast mode", value=True,
        help=(
            "Skips the most expensive correlation and interaction plots. "
            "Full profiling on a wide dataset can take several minutes."
        ),
    )
with control_right:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run = st.button("Generate profile", type="primary", width="stretch")

if run:
    with st.spinner("Profiling variables… this can take a moment on wide data."):
        try:
            profiler = DatasetProfiler()
            profiler.minimal = minimal
            st.session_state["profiling_summary"] = profiler.profile(df, dataset_name=name)
        except Exception as exc:
            st.error(f"Profiling failed: {exc}")

summary = st.session_state.get("profiling_summary")
if not summary:
    ui.divider()
    ui.empty_state(
        "No profile generated yet.",
        "Press <strong>Generate profile</strong> to build the report.",
    )
    st.stop()

ui.divider()
ui.section("Overview")

ui.stat_row([
    {"label": "Records", "value": f"{summary.get('n_records', 0):,}"},
    {"label": "Features", "value": summary.get("n_features", 0)},
    {"label": "Duplicate rows", "value": f"{summary.get('n_duplicates', 0):,}"},
    {"label": "In memory", "value": f"{summary.get('memory_size', 0) / 1024:.1f} KB"},
])

report_path = Path(summary.get("report_path", ""))
if not report_path.exists():
    st.error("The generated report file could not be found on disk.")
    st.stop()

html_data = report_path.read_text(encoding="utf-8")

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.download_button(
    "Download full report (HTML)",
    data=html_data,
    file_name=report_path.name,
    mime="text/html",
)

ui.section("Report preview")
st.components.v1.html(html_data, height=820, scrolling=True)
