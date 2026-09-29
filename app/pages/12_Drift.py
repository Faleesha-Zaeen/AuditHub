"""
AuditHub Page - Data Drift Detection
======================================

Compares the active dataset against a reference (training) dataset and shows
which columns have moved, by how much, and why.
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
import plotly.graph_objects as go
import streamlit as st

from app.components import ui
from app.components.session import data_source_caption, get_active_df, require_dataset
from src.drift.detector import DriftDetector, DriftStatus
from src.ingestion.dataset_loader import DatasetLoader
from src.utils.constants import RAW_DATA_DIR

ui.page_setup("Drift", icon="◨")

ui.page_header(
    "Does the new data still look like training data?",
    "Every shared column is scored with the Population Stability Index and a "
    "significance test, then graded Stable, Warning or Critical.",
    eyebrow="Drift detection",
)

current_df = require_dataset()
data_source_caption()

_STATUS_COLOR = ui.STATUS_COLORS


# ---------------------------------------------------------------------------
# Choose the reference dataset
# ---------------------------------------------------------------------------

ui.section("1 · Reference dataset")

source = st.radio(
    "Where should the reference (baseline) data come from?",
    ["Upload a file", "Previously ingested file"],
    horizontal=True,
)

reference_df = None
reference_name = "reference"

if source == "Upload a file":
    accepted = sorted(ext.lstrip(".") for ext in DatasetLoader.supported_extensions())
    uploaded = st.file_uploader("Reference dataset", type=accepted, key="drift_reference")
    if uploaded is not None:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ref_path = RAW_DATA_DIR / uploaded.name
        with open(ref_path, "wb") as fh:
            fh.write(uploaded.getbuffer())
        try:
            reference_df, _ = DatasetLoader().load(ref_path)
            reference_name = uploaded.name
        except Exception as exc:
            st.error(f"Could not load the reference dataset: {exc}")
else:
    existing = sorted(
        (p for p in RAW_DATA_DIR.glob("*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if RAW_DATA_DIR.exists() else []
    if not existing:
        st.info("No previously ingested files found. Upload one instead.")
    else:
        choice = st.selectbox("Previously ingested file", [p.name for p in existing])
        if choice:
            try:
                reference_df, _ = DatasetLoader().load(RAW_DATA_DIR / choice)
                reference_name = choice
            except Exception as exc:
                st.error(f"Could not load '{choice}': {exc}")

if reference_df is None:
    st.stop()

st.success(
    f"Reference: **{reference_name}** ({len(reference_df):,} rows x {len(reference_df.columns)} columns)"
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

ui.divider()
ui.section("2 · Thresholds")
with st.expander("Adjust drift thresholds", expanded=False):
    st.caption(
        "Drift is scored with the Population Stability Index (PSI). The 0.10 / 0.25 "
        "split is the long-standing convention: below 0.10 a shift is not actionable, "
        "above 0.25 the population has genuinely moved."
    )
    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        warning_threshold = st.number_input(
            "Warning PSI", min_value=0.0, max_value=2.0, value=0.10, step=0.01
        )
    with tcol2:
        critical_threshold = st.number_input(
            "Critical PSI", min_value=0.0, max_value=5.0, value=0.25, step=0.01
        )
    with tcol3:
        alpha = st.number_input(
            "Significance level", min_value=0.001, max_value=0.5, value=0.05, step=0.005,
            format="%.3f",
        )

shared = [c for c in reference_df.columns if c in current_df.columns]
selected = st.multiselect(
    "Columns to compare (leave empty for all shared columns)",
    options=[str(c) for c in shared],
    default=[],
)

if st.button("Detect drift", type="primary"):
    with st.spinner("Comparing distributions..."):
        try:
            detector = DriftDetector(thresholds={
                "psi_warning_threshold": warning_threshold,
                "psi_critical_threshold": critical_threshold,
                "significance_level": alpha,
            })
            report = detector.detect(
                reference_df,
                current_df,
                columns=selected or None,
                reference_name=reference_name,
                current_name=st.session_state.get("dataset_name", "current"),
            )
            st.session_state["drift_report"] = report.to_dict()
            st.session_state["_drift_results"] = report
        except Exception as exc:
            st.error(f"Drift detection failed: {exc}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

report = st.session_state.get("_drift_results")
if report is None:
    ui.empty_state("No drift check run yet.",
                   "Choose a reference dataset above and press <strong>Detect drift</strong>.")
    st.stop()

ui.divider()
ui.section("3 · Results")

ui.stat_row([
    {"label": "Verdict", "value": report.status.replace("_", " ").title(),
     "sub": ui.badge(report.status, report.status)},
    {"label": "Mean PSI", "value": f"{report.overall_score:.4f}",
     "sub": f"highest column {report.max_score:.4f}"},
    {"label": "Drifted columns",
     "value": f"{len(report.drifted_columns)} / {len(report.compared_columns)}"},
    {"label": "Schema changes",
     "value": len(report.added_columns) + len(report.removed_columns)
              + len(report.type_changed_columns)},
])

st.caption(report.summary())

if report.removed_columns:
    st.error(
        "**Removed columns** (present in reference, missing now): "
        + ", ".join(f"`{c}`" for c in report.removed_columns)
    )
if report.added_columns:
    st.info(
        "**New columns** (not available at training time): "
        + ", ".join(f"`{c}`" for c in report.added_columns)
    )
if report.type_changed_columns:
    st.warning(
        "**Type changes** (schema break): "
        + ", ".join(f"`{c}`" for c in report.type_changed_columns)
    )

# PSI ranking chart
scored = report.compared_columns
if scored:
    psi_df = pd.DataFrame([
        {"column": c.column, "psi": c.psi, "status": c.status} for c in scored
    ]).sort_values("psi", ascending=True)

    fig = px.bar(
        psi_df,
        x="psi",
        y="column",
        orientation="h",
        color="status",
        color_discrete_map=_STATUS_COLOR,
        title="Population Stability Index by column",
    )
    fig.add_vline(x=warning_threshold, line_dash="dash", line_color="#f59e0b",
                  annotation_text="warning")
    fig.add_vline(x=critical_threshold, line_dash="dash", line_color="#ef4444",
                  annotation_text="critical")
    ui.plot(fig, height=max(300, 26 * len(psi_df)))

# Per-column table
ui.section("Column verdicts")
table = report.to_frame()
ui.dataframe(table)

# ---------------------------------------------------------------------------
# Distribution comparison
# ---------------------------------------------------------------------------

ui.divider()
ui.section("Reference vs current distributions")

chartable = [c for c in report.columns if c.distribution]
if not chartable:
    st.info("No column had enough data to chart.")
else:
    default_col = (report.drifted_columns or chartable)[0].column
    names = [c.column for c in chartable]
    picked = st.selectbox(
        "Column", names, index=names.index(default_col) if default_col in names else 0
    )
    result = next(c for c in chartable if c.column == picked)

    badge = _STATUS_COLOR.get(result.status, "#6b7280")
    st.markdown(
        f"<span style='background:{badge};color:white;padding:3px 10px;"
        f"border-radius:10px;font-size:0.8rem;'>{result.status}</span>"
        f"&nbsp;&nbsp;PSI <strong>{result.psi:.4f}</strong>"
        + (f"&nbsp;&nbsp;{result.test_name} p={result.p_value:.3g}"
           if result.p_value is not None else ""),
        unsafe_allow_html=True,
    )
    st.caption(result.explanation)

    dist = result.distribution_frame()
    fig2 = px.bar(
        dist,
        x="bin",
        y="percent",
        color="dataset",
        barmode="group",
        color_discrete_map={"reference": ui.PLOT_COLORS["reference"],
                            "current": ui.PLOT_COLORS["current"]},
        labels={"percent": "% of rows", "bin": ""},
        title=f"Distribution of '{picked}'",
    )
    ui.plot(fig2, height=380)

    if result.column_kind == "numeric" and result.reference_stats:
        stats_df = pd.DataFrame([
            {"statistic": k, "reference": result.reference_stats.get(k),
             "current": result.current_stats.get(k)}
            for k in ("count", "mean", "std", "min", "p25", "median", "p75", "max", "missing_pct")
            if k in result.reference_stats
        ])
        ui.dataframe(stats_df)

ui.divider()
st.caption(
    "The drift report is included automatically in the consolidated report "
    "generated on the **Reports** page."
)
