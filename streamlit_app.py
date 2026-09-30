"""
Streamlit Community Cloud entry point
======================================

The cloud deploy form validates the "Main file path" field against a Python
script, and a root-level file is the convention it expects. This shim forwards
to the real application in ``app/main.py`` so the deployed URL serves the full
multi-page dashboard without duplicating any logic.

Run locally with either:
    streamlit run streamlit_app.py
    streamlit run app/main.py
"""

import runpy
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "app" / "main.py"

runpy.run_path(str(_TARGET), run_name="__main__")
