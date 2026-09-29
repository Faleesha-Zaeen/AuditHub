"""
AuditHub Page - Reports
=========================

Compiles every module's findings into one consolidated HTML or JSON report.
"""

# Ensure the project root is importable when Streamlit runs this file as a
# standalone script (it puts the page's own directory on sys.path, not the root).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.components import ui
from app.components.session import data_source_caption, require_dataset
from src.reporting.generator import ReportGenerator

ui.page_setup("Reports", icon="▤")

ui.page_header(
    "Export the evidence",
    "One document covering validation, audit, health, repairs, drift, lineage and "
    "explainability — whichever of them you have run.",
    eyebrow="Step 12 · Reporting",
)

df = require_dataset()
data_source_caption()

validation_report = st.session_state.get("validation_report")
health_report = st.session_state.get("health_report")
quality_report = st.session_state.get("quality_report")
repair_log = (
    st.session_state["repairer"].get_log() if "repairer" in st.session_state else []
)
robustness_results = st.session_state.get("robustness_results")
training_results = st.session_state.get("training_results")
training_metrics = training_results.get("metrics") if training_results else None
drift_report = st.session_state.get("drift_report")
version_comparison = st.session_state.get("version_comparison")
lineage = st.session_state.get("lineage")
explainability = st.session_state.get("explainability")

SECTIONS = [
    ("Dataset summary", True, "Upload"),
    ("Validation", bool(validation_report), "Validation"),
    ("Quality audit", bool(quality_report), "Quality Audit"),
    ("Health score", bool(health_report), "Health Score"),
    ("Repair log", bool(repair_log), "Repair"),
    ("Data drift", bool(drift_report), "Drift"),
    ("Version comparison", bool(version_comparison), "Version Compare"),
    ("Data lineage", bool(lineage), "Version Compare"),
    ("Robustness", bool(robustness_results), "Robustness"),
    ("Training metrics", bool(training_metrics), "Training"),
    ("Explainability", bool(explainability), "Explainability"),
]

ready = sum(1 for _, available, _ in SECTIONS if available)

ui.stat_row([
    {"label": "Sections ready", "value": f"{ready} / {len(SECTIONS)}"},
    {"label": "Dataset", "value": st.session_state.get("dataset_name", "—")},
    {"label": "Repair steps", "value": len(repair_log)},
])

ui.divider()
ui.section("What will be included")

for row_start in range(0, len(SECTIONS), 3):
    columns = st.columns(3)
    for column, (title, available, page) in zip(columns, SECTIONS[row_start:row_start + 3]):
        with column:
            badge = ui.badge("included", "OK") if available else ui.badge("not run", "SKIPPED")
            hint = (
                "" if available
                else f'<div style="font-size:0.76rem;color:var(--text-muted);'
                     f'margin-top:6px;">run it on <strong>{page}</strong></div>'
            )
            st.markdown(
                f'<div class="ah-card" style="padding:13px 16px;margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;gap:8px;">'
                f'<span style="font-size:0.87rem;color:var(--text);">{title}</span>'
                f"{badge}</div>{hint}</div>",
                unsafe_allow_html=True,
            )

ui.divider()
ui.section("Generate")

format_column, action_column = st.columns([1, 3])
with format_column:
    format_option = st.selectbox("Format", ["HTML", "JSON"])
with action_column:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    generate = st.button("Generate report", type="primary")

if generate:
    with st.spinner("Compiling…"):
        try:
            filepath = ReportGenerator().export_report(
                dataset_name=st.session_state.get("dataset_name", "dataset"),
                validation_report=validation_report
                or {"success": False, "score": 0.0, "ge_results": []},
                health_report=health_report
                or {"overall_score": 0.0, "grade": "F", "dimensions": {}, "explanations": []},
                audit_report=quality_report or {"findings": [], "summary": {}},
                repair_log=repair_log,
                robustness_results=robustness_results,
                training_metrics=training_metrics,
                format_type=format_option.lower(),
                drift_report=drift_report,
                version_comparison=version_comparison,
                lineage=lineage,
                explainability=explainability,
            )
            st.session_state["_report_path"] = str(filepath)
            st.session_state["_report_format"] = format_option
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

report_path = st.session_state.get("_report_path")
if report_path:
    path = _Path(report_path)
    if not path.exists():
        st.error("The generated report could not be found on disk.")
        st.stop()

    report_format = st.session_state.get("_report_format", "HTML")
    data = path.read_text(encoding="utf-8")

    st.success(f"Saved to `reports/{path.name}`")
    st.download_button(
        f"Download {report_format} report",
        data=data,
        file_name=path.name,
        mime="text/html" if report_format == "HTML" else "application/json",
    )

    if report_format == "HTML":
        ui.section("Preview")
        st.components.v1.html(data, height=900, scrolling=True)
