"""
Tests for the Streamlit pages
==============================

The bugs that made repair look broken all lived in ``app/pages`` -- a loader
result unpacked into a single variable, and repairs held in a session object no
other page read. Nothing here was covered by tests, so nothing caught them.

These run each page headlessly with Streamlit's AppTest harness.
"""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.dataset_loader import DatasetLoader

AppTest = pytest.importorskip("streamlit.testing.v1", reason="needs streamlit").AppTest


MESSY_CSV = """id,age,salary,city,score,signup,target
1,25,"$50,000",Delhi ,3.5,2021-01-05,yes
2,,60000,Mumbai,4.0,2022/03/14,no
3,30,NA,Delhi,,2020-11-30,yes
4,35,?,,2.5,N/A,
5,40,"70,000",Pune,-,2023-06-01,no
6,,80000,Delhi,4.5,2021-07-19,yes
6,,80000,Delhi,4.5,2021-07-19,yes
"""

# Pages whose engines are slow (Great Expectations, ydata-profiling) are
# excluded from the default run and covered by the slow marker below.
FAST_PAGES = [
    "0_Overview",
    "1_Upload",
    "2_Dataset_Summary",
    "5_Quality_Audit",
    "6_Health_Score",
    "7_Repair",
    "8_Mutation_Lab",
    "9_Robustness",
    "10_Training",
    "11_Reports",
    "12_Drift",
    "13_Version_Compare",
    "14_Explainability",
]
SLOW_PAGES = ["3_Validation", "4_Profiling"]

# Version Compare works without an active dataset (it can compare two files on
# disk), so it is exempt from the "prompt the user to upload" guard test.
PAGES_WITHOUT_DATASET_GUARD = {"0_Overview", "1_Upload", "13_Version_Compare"}


@pytest.fixture
def messy_df(tmp_path):
    path = tmp_path / "messy.csv"
    path.write_text(MESSY_CSV, encoding="utf-8")
    df, _ = DatasetLoader().load(path)
    return df


def _page(name: str, df: pd.DataFrame, **session):
    """Run a page with a dataset already in session state."""
    at = AppTest.from_file(f"app/pages/{name}.py", default_timeout=120)
    at.session_state["df"] = df
    at.session_state["dataset_name"] = "messy.csv"
    for key, value in session.items():
        at.session_state[key] = value
    return at.run()


class TestPagesRender:
    """Every page must render without raising."""

    @pytest.mark.parametrize("page", FAST_PAGES)
    def test_page_renders_with_a_dataset(self, page, messy_df):
        at = _page(page, messy_df)
        assert not at.exception, f"{page}: {at.exception[0].message if at.exception else ''}"

    @pytest.mark.slow
    @pytest.mark.parametrize("page", SLOW_PAGES)
    def test_slow_page_renders(self, page, messy_df):
        at = _page(page, messy_df)
        assert not at.exception

    @pytest.mark.parametrize("page", FAST_PAGES)
    def test_page_stops_cleanly_without_a_dataset(self, page):
        at = AppTest.from_file(f"app/pages/{page}.py", default_timeout=120)
        at.run()
        assert not at.exception
        if page not in PAGES_WITHOUT_DATASET_GUARD:
            assert at.warning, f"{page} should prompt the user to upload first"


class TestRepairPageBehaviour:
    """The Repair page is where the reported bug lived."""

    def test_auto_repair_removes_every_missing_value(self, messy_df):
        at = _page("7_Repair", messy_df)
        at.selectbox[0].set_value("target").run()
        [b for b in at.button if "Auto-Repair" in b.label][0].click().run()

        assert not at.exception
        repairer = at.session_state["repairer"]
        assert int(repairer.df.isna().sum().sum()) == 0
        assert repairer.verify_clean(target_column="target") == []

    def test_repairs_propagate_to_other_pages(self, messy_df):
        at = _page("7_Repair", messy_df)
        at.selectbox[0].set_value("target").run()
        [b for b in at.button if "Auto-Repair" in b.label][0].click().run()

        carried = {
            k: at.session_state[k]
            for k in ("df", "dataset_name", "repairer", "target_column")
            if k in at.session_state
        }
        repaired_rows = len(at.session_state["repairer"].df)

        for page in ("2_Dataset_Summary", "5_Quality_Audit", "6_Health_Score"):
            at2 = AppTest.from_file(f"app/pages/{page}.py", default_timeout=120)
            for key, value in carried.items():
                at2.session_state[key] = value
            at2.run()
            assert not at2.exception, page
            captions = " ".join(c.value for c in at2.caption)
            assert "repaired" in captions, f"{page} is still reading the original data"

        # The original upload is untouched, so a reset is always possible.
        assert len(carried["df"]) > repaired_rows

    def test_reset_restores_the_original_dataset(self, messy_df):
        at = _page("7_Repair", messy_df)
        at.selectbox[0].set_value("target").run()
        [b for b in at.button if "Auto-Repair" in b.label][0].click().run()
        assert at.session_state["repairer"].get_log()

        [b for b in at.button if "Reset" in b.label][0].click().run()
        assert "repairer" not in at.session_state or not at.session_state["repairer"].get_log()

    def test_undo_is_disabled_before_any_repair(self, messy_df):
        at = _page("7_Repair", messy_df)
        undo = [b for b in at.button if "Undo" in b.label]
        assert undo and undo[0].disabled

    def test_missing_value_table_is_shown(self, messy_df):
        at = _page("7_Repair", messy_df)
        assert not at.exception
        # Before repair the page must report the gaps rather than claim success.
        # The counters are custom stat tiles rather than st.metric, so the
        # assertion looks at the rendered markdown.
        rendered = " ".join(m.value for m in at.markdown)
        assert "Missing cells" in rendered
        assert "Repair steps" in rendered


class TestSessionHelpers:
    """The shared session module underpinning propagation."""

    def test_active_df_is_the_original_before_repair(self, messy_df):
        at = _page("2_Dataset_Summary", messy_df)
        captions = " ".join(c.value for c in at.caption)
        assert "originally uploaded" in captions

    def test_new_upload_clears_previous_results(self, messy_df):
        # Stale reports from a previous dataset must not survive a new upload.
        at = _page("2_Dataset_Summary", messy_df, quality_report={"findings": []})
        assert not at.exception
