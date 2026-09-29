"""
AuditHub Page - Validation
============================

Runs schema checks, an auto-built Great Expectations suite and custom quality
checks against the active dataset.
"""

# Ensure the project root is importable when Streamlit runs this file as a
# standalone script (it puts the page's own directory on sys.path, not the root).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from app.components import ui
from app.components.session import data_source_caption, require_dataset, target_selector
from src.validation.validator import DatasetValidator

ui.page_setup("Validation", icon="✓")

ui.page_header(
    "Validate the data contract",
    "Expectations are built automatically from the inferred schema, so there is "
    "nothing to hand-write. Structural checks and custom quality rules run alongside.",
    eyebrow="Step 3 · Validation",
)

df = require_dataset()
data_source_caption()
name = st.session_state.get("dataset_name", "dataset")

control_left, control_right = st.columns([3, 1])
with control_left:
    target = target_selector(df, label="Target column (optional)")
with control_right:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run = st.button("Run validation", type="primary", width="stretch")

if run:
    with st.spinner("Running Great Expectations and custom checks…"):
        try:
            report = DatasetValidator().validate(df, target_column=target, dataset_name=name)
            st.session_state["validation_report"] = report.to_dict()
        except Exception as exc:
            st.error(f"Validation failed: {exc}")

report = st.session_state.get("validation_report")
if not report:
    ui.divider()
    ui.empty_state(
        "No validation run yet.",
        "Press <strong>Run validation</strong> to build and execute the suite.",
    )
    st.stop()

ui.divider()
ui.section("Results")

score = float(report.get("score", 0.0))
passed = report.get("passed_checks", 0)
total = report.get("total_checks", 0)
success = bool(report.get("success"))

ui.stat_row([
    {"label": "Outcome", "value": "Pass" if success else "Fail",
     "sub": ui.badge("passed" if success else "failed", "OK" if success else "CRITICAL")},
    {"label": "Score", "value": f"{score:.1f}", "sub": "out of 100"},
    {"label": "Checks passed", "value": f"{passed} / {total}"},
    {"label": "Failed checks", "value": max(int(total) - int(passed), 0)},
])

recommendations = report.get("recommendations", [])
if recommendations:
    ui.section("Recommendations", "What to fix, in the order it matters.")
    for recommendation in recommendations:
        st.markdown(
            f'<div class="ah-card" style="padding:12px 16px;margin-bottom:8px;">'
            f'<span style="color:var(--text-secondary);font-size:0.88rem;">'
            f"{recommendation}</span></div>",
            unsafe_allow_html=True,
        )
else:
    st.success("No recommendations — the dataset satisfies every check.")

ui.divider()
expectations_tab, schema_tab = st.tabs(["Expectations", "Schema"])

with expectations_tab:
    ge_results = report.get("ge_results", [])
    if ge_results:
        ui.dataframe(pd.DataFrame(ge_results), height=420)
    else:
        ui.empty_state("No expectation results were produced.")

with schema_tab:
    schema_results = report.get("schema_results", {})
    errors = schema_results.get("errors", [])
    warnings = schema_results.get("warnings", [])

    if errors:
        st.error("**Errors**\n\n" + "\n".join(f"- {e}" for e in errors))
    if warnings:
        st.warning("**Warnings**\n\n" + "\n".join(f"- {w}" for w in warnings))
    if not errors and not warnings:
        st.success("The dataset structure matches the expected schema.")
