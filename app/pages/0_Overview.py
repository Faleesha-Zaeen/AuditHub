"""
AuditHub - Streamlit Application Entry Point
==============================================

Landing page: platform state at a glance and where to go next.
"""

# Ensure the project root is importable when Streamlit runs this file as a
# standalone script (it puts the page's own directory on sys.path, not the root).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from app.components import ui
from app.components.session import has_dataset, has_repairs
from src.utils.dataset_registry import DatasetRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

ui.page_setup("Overview", icon="◆")

ui.page_header(
    "Data quality, end to end",
    "Ingest a dataset, audit it across six quality dimensions, repair it, "
    "check it for drift, train a model and explain what it learned.",
    eyebrow="Overview",
)

# ---------------------------------------------------------------------------
# Platform state
# ---------------------------------------------------------------------------

registry = DatasetRegistry()
try:
    registered = registry.count()
    groups = registry.list_version_groups()
except Exception as exc:  # a broken registry must not blank the landing page
    logger.warning("Could not read the registry: %s", exc)
    registered, groups = 0, []

active_dataset = st.session_state.get("dataset_name")
validation = st.session_state.get("validation_report") or {}
health = st.session_state.get("health_report") or {}

if active_dataset:
    session_sub = ui.badge("repaired" if has_repairs() else "loaded")
else:
    session_sub = '<span style="color:var(--text-muted);">nothing loaded yet</span>'

score = validation.get("score")
health_grade = health.get("grade")

ui.stat_row([
    {"label": "Registered datasets", "value": registered,
     "sub": f"{len(groups)} logical dataset(s)"},
    {"label": "Active session", "value": active_dataset or "—", "sub": session_sub},
    {"label": "Validation score",
     "value": f"{score:.0f}" if isinstance(score, (int, float)) else "—",
     "sub": "out of 100" if isinstance(score, (int, float)) else "not run yet"},
    {"label": "Health grade", "value": health_grade or "—",
     "sub": ui.badge(health_grade, health_grade) if health_grade else "not scored yet"},
])

ui.divider()

# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

ui.section(
    "The pipeline",
    "Each stage feeds the next. Repairs made in the middle are what the later "
    "stages analyse.",
)

ui.flow([
    "Ingest", "Validate", "Drift", "Profile", "Audit", "Health",
    "Repair", "Re-audit", "Mutate", "Robustness", "Train", "Explain", "Report",
])

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

WORKFLOWS = [
    ("Assess a new dataset",
     "Upload, then work down the sidebar: Summary, Validation, Quality Audit and "
     "Health Score each answer a different question about the same data.",
     "Upload"),
    ("Fix what is broken",
     "Repair applies one-click auto-repair or per-column fixes, then every other "
     "page recomputes against the repaired data.",
     "Repair"),
    ("Check for drift",
     "Compare incoming data against the data a model was trained on, per column, "
     "with a Stable / Warning / Critical verdict.",
     "Drift"),
    ("Track versions and lineage",
     "See what changed between two versions of a dataset, and what the pipeline "
     "did to any individual column.",
     "Version Compare"),
    ("Train and explain",
     "Train and compare models, then see which features drove a specific "
     "prediction and in which direction.",
     "Training"),
    ("Export the evidence",
     "One consolidated HTML or JSON report covering validation, audit, health, "
     "repairs, drift, lineage and explainability.",
     "Reports"),
]

for row_start in range(0, len(WORKFLOWS), 3):
    columns = st.columns(3)
    for column, (title, description, page) in zip(columns, WORKFLOWS[row_start:row_start + 3]):
        with column:
            st.markdown(
                f"""
                <div class="ah-card" style="height:176px;">
                    <div style="font-size:0.95rem;font-weight:600;color:var(--text);
                                margin-bottom:8px;">{title}</div>
                    <div style="font-size:0.83rem;color:var(--text-muted);
                                line-height:1.55;">{description}</div>
                    <div style="margin-top:12px;">
                        <span class="ah-badge ah-badge-neutral">{page}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

ui.divider()

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ui.section("Recent ingestions")

try:
    records = registry.list_all()
except Exception as exc:
    records = []
    logger.warning("Could not list datasets: %s", exc)

if not records:
    ui.empty_state(
        "No datasets registered yet.",
        "Head to <strong>Upload</strong> in the sidebar to ingest your first file.",
    )
else:
    ui.dataframe(
        pd.DataFrame([
            {
                "Dataset": r.filename,
                "Version": r.extra_metadata.get("version", 1),
                "Rows": f"{r.num_rows:,}",
                "Columns": r.num_columns,
                "Format": (r.source_format or "").upper().lstrip("."),
                "Registered": (r.upload_timestamp or "")[:19].replace("T", " "),
            }
            for r in records[:10]
        ])
    )

if not has_dataset():
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.info("No dataset is loaded in this session. Open **Upload** to begin.")
