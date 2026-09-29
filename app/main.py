"""
AuditHub - Application Router
===============================

Entry point. Defines the navigation and hands off to the selected page.

Streamlit's automatic ``pages/`` navigation truncates a long page list behind a
"View N more" control and renders bare filenames. Declaring the navigation
explicitly gives grouped sections, real titles and icons, and keeps every page
one click away.
"""

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.utils.constants import APP_NAME, APP_VERSION

st.set_page_config(
    page_title=f"{APP_NAME} · Data Quality & MLOps",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PAGES = _Path(__file__).resolve().parent / "pages"


def _page(filename: str, title: str, icon: str, default: bool = False):
    """Declare a navigation entry for a page file."""
    return st.Page(_PAGES / filename, title=title, icon=icon, default=default)


# Grouped so the sidebar reads as a workflow rather than an alphabetical dump.
navigation = st.navigation(
    {
        "Overview": [
            _page("0_Overview.py", "Overview", ":material/dashboard:", default=True),
        ],
        "Data": [
            _page("1_Upload.py", "Upload", ":material/upload_file:"),
            _page("2_Dataset_Summary.py", "Summary", ":material/table_chart:"),
            _page("3_Validation.py", "Validation", ":material/verified:"),
            _page("4_Profiling.py", "Profiling", ":material/analytics:"),
        ],
        "Quality": [
            _page("5_Quality_Audit.py", "Quality Audit", ":material/rule:"),
            _page("6_Health_Score.py", "Health Score", ":material/monitor_heart:"),
            _page("7_Repair.py", "Repair", ":material/build:"),
        ],
        "Monitoring": [
            _page("12_Drift.py", "Drift", ":material/ssid_chart:"),
            _page("13_Version_Compare.py", "Versions & Lineage", ":material/history:"),
        ],
        "Modelling": [
            _page("8_Mutation_Lab.py", "Mutation Lab", ":material/science:"),
            _page("9_Robustness.py", "Robustness", ":material/shield:"),
            _page("10_Training.py", "Training", ":material/model_training:"),
            _page("14_Explainability.py", "Explainability", ":material/psychology:"),
        ],
        "Output": [
            _page("11_Reports.py", "Reports", ":material/description:"),
        ],
    },
    # Without this Streamlit hides everything past the tenth entry behind a
    # "View 5 more" control, which buries a third of the product.
    expanded=True,
)

navigation.run()
