"""
Tests for the repair contract
==============================

These lock in the guarantees the Repair page promises, and each one maps to a
defect that shipped: mean on a text column raised, a constant fill turned a
numeric column into ``object``, mode filled a gap with another missing marker,
and a failed repair still pushed a frame onto the undo stack.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.ingestion.cleaning import MISSING_SENTINELS
from src.ingestion.dataset_loader import DatasetLoader
from src.repair.repairer import AutoRepairResult, DatasetRepairer


MESSY_CSV = """id,age,salary,city,score,signup,target
1,25,"$50,000",Delhi ,3.5,2021-01-05,yes
2,,60000,Mumbai,4.0,2022/03/14,no
3,30,NA,Delhi,,2020-11-30,yes
4,35,?,,2.5,N/A,
5,40,"70,000",Pune,-,2023-06-01,no
6,,80000,Delhi,4.5,2021-07-19,yes
6,,80000,Delhi,4.5,2021-07-19,yes
"""


@pytest.fixture
def messy_df(tmp_path):
    """A dataset carrying every defect the repairer is expected to handle."""
    path = tmp_path / "messy.csv"
    path.write_text(MESSY_CSV, encoding="utf-8")
    df, _ = DatasetLoader().load(path)
    return df


# ---------------------------------------------------------------------------
# Type-aware imputation
# ---------------------------------------------------------------------------


class TestImputationIsTypeAware:
    """Imputation must not raise on real-world columns."""

    def test_mean_on_numeric_text_column_works(self, messy_df):
        # "salary" arrives as text because of a "?" -- previously TypeError:
        # can only concatenate str (not "int") to str.
        r = DatasetRepairer(messy_df)
        out = r.impute_missing("salary", strategy="mean")
        assert out["salary"].isna().sum() == 0

    def test_mean_on_text_column_falls_back_to_mode(self, messy_df):
        r = DatasetRepairer(messy_df)
        out = r.impute_missing("city", strategy="mean")
        assert out["city"].isna().sum() == 0
        entry = r.get_log()[-1]["details"]
        assert entry["strategy"] == "mode"
        assert entry["requested_strategy"] == "mean"
        assert "note" in entry

    def test_integer_column_stays_integral(self, messy_df):
        # The mean of the age column is fractional; writing 32.5 into a column
        # of whole numbers is exactly what "not repaired properly" looked like.
        r = DatasetRepairer(messy_df)
        out = r.impute_missing("age", strategy="mean")
        assert pd.api.types.is_integer_dtype(out["age"].dtype)
        assert out["age"].isna().sum() == 0
        assert all(float(v) % 1 == 0 for v in out["age"])

    def test_constant_is_cast_to_column_dtype(self, messy_df):
        # Streamlit hands back the string "0"; filling with it used to turn the
        # whole column into object.
        r = DatasetRepairer(messy_df)
        out = r.impute_missing("age", strategy="constant", fill_value="0")
        assert pd.api.types.is_integer_dtype(out["age"].dtype)

    def test_blank_constant_is_rejected(self, messy_df):
        r = DatasetRepairer(messy_df)
        with pytest.raises(ValueError, match="needs a fill value"):
            r.impute_missing("age", strategy="constant", fill_value="")

    def test_non_numeric_constant_for_numeric_column_is_rejected(self, messy_df):
        r = DatasetRepairer(messy_df)
        with pytest.raises(ValueError, match="non-numeric value"):
            r.impute_missing("age", strategy="constant", fill_value="abc")

    def test_unknown_strategy_raises(self, messy_df):
        r = DatasetRepairer(messy_df)
        with pytest.raises(ValueError, match="Unknown imputation strategy"):
            r.impute_missing("age", strategy="bogus")

    def test_column_without_nulls_is_a_no_op(self, messy_df):
        r = DatasetRepairer(messy_df)
        r.impute_missing("id", strategy="median")
        assert r.get_log() == []


# ---------------------------------------------------------------------------
# History integrity
# ---------------------------------------------------------------------------


class TestHistoryIntegrity:
    """A repair that did not happen must not occupy the undo stack."""

    def test_failed_repair_leaves_no_history(self, messy_df):
        r = DatasetRepairer(messy_df)
        with pytest.raises(ValueError):
            r.impute_missing("age", strategy="bogus")
        assert r.get_log() == []
        assert r._history == []

    def test_undo_restores_previous_state(self, messy_df):
        r = DatasetRepairer(messy_df)
        before = int(r.df["age"].isna().sum())
        r.impute_missing("age", strategy="median")
        assert int(r.df["age"].isna().sum()) == 0
        r.revert()
        assert int(r.df["age"].isna().sum()) == before

    def test_log_records_the_value_written(self, messy_df):
        expected_nulls = int(messy_df["age"].isna().sum())
        r = DatasetRepairer(messy_df)
        r.impute_missing("age", strategy="median")
        details = r.get_log()[-1]["details"]
        assert details["fill_value"] is not None
        assert details["null_count"] == expected_nulls
        # Must be JSON-serialisable for the manifest and the API.
        json.dumps(details)


# ---------------------------------------------------------------------------
# Auto-repair
# ---------------------------------------------------------------------------


class TestAutoRepair:
    """Whole-dataset repair in one pass."""

    def test_removes_every_missing_value(self, messy_df):
        r = DatasetRepairer(messy_df)
        r.auto_repair(target_column="target")
        assert int(r.df.isna().sum().sum()) == 0

    def test_target_labels_are_never_imputed(self, messy_df):
        # Row 4 has no label. Inventing one would fabricate ground truth, so
        # the row must be dropped instead.
        r = DatasetRepairer(messy_df)
        result = r.auto_repair(target_column="target")
        assert result.target_rows_dropped == 1
        assert "target" not in result.columns_imputed
        assert r.df["target"].isna().sum() == 0

    def test_duplicates_are_removed(self, messy_df):
        r = DatasetRepairer(messy_df)
        result = r.auto_repair(target_column="target")
        assert result.duplicates_removed == 1
        assert int(r.df.duplicated().sum()) == 0

    def test_result_is_serialisable(self, messy_df):
        r = DatasetRepairer(messy_df)
        result = r.auto_repair(target_column="target")
        assert isinstance(result, AutoRepairResult)
        json.dumps(result.to_dict())
        assert isinstance(result.summary(), str)

    def test_runs_without_a_target(self, messy_df):
        r = DatasetRepairer(messy_df)
        r.auto_repair()
        assert int(r.df.isna().sum().sum()) == 0

    def test_all_null_column_is_dropped_not_filled(self):
        df = pd.DataFrame({"keep": [1, 2, 3], "empty": [None, None, None]})
        r = DatasetRepairer(df)
        result = r.auto_repair()
        assert "empty" not in r.df.columns
        assert any("empty" in s for s in result.skipped)


# ---------------------------------------------------------------------------
# Export guarantees
# ---------------------------------------------------------------------------


class TestExportGuarantees:
    """The contract the downloaded CSV must satisfy."""

    def test_verify_clean_flags_an_unrepaired_frame(self, messy_df):
        r = DatasetRepairer(messy_df)
        assert r.verify_clean(target_column="target")

    def test_verify_clean_passes_after_auto_repair(self, messy_df):
        r = DatasetRepairer(messy_df)
        r.auto_repair(target_column="target")
        assert r.verify_clean(target_column="target") == []

    def test_no_residual_sentinels_remain(self, messy_df):
        r = DatasetRepairer(messy_df)
        r.auto_repair(target_column="target")
        for col in r.df.columns:
            if r.df[col].dtype == object:
                values = r.df[col].astype(str).str.strip().str.lower()
                assert not values.isin(MISSING_SENTINELS).any()

    def test_export_round_trips_clean(self, messy_df, tmp_path):
        """Re-loading the exported file must need no further repair."""
        r = DatasetRepairer(messy_df)
        r.auto_repair(target_column="target")
        csv_path, manifest_path = r.export(
            dataset_name="messy", output_dir=tmp_path, target_column="target"
        )

        assert csv_path.exists() and manifest_path.exists()

        reloaded, _ = DatasetLoader().load(csv_path)
        assert reloaded.shape == r.df.shape
        assert list(reloaded.columns) == list(r.df.columns)
        assert int(reloaded.isna().sum().sum()) == 0

        second = DatasetRepairer(reloaded)
        again = second.auto_repair(target_column="target")
        assert again.columns_imputed == {}
        assert second.verify_clean(target_column="target") == []

    def test_export_writes_no_index_column(self, messy_df, tmp_path):
        r = DatasetRepairer(messy_df)
        r.auto_repair(target_column="target")
        csv_path, _ = r.export(dataset_name="m", output_dir=tmp_path)
        header = csv_path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert not header.startswith(",")
        assert "Unnamed" not in header

    def test_manifest_is_valid_json_and_reports_verification(self, messy_df, tmp_path):
        r = DatasetRepairer(messy_df)
        r.auto_repair(target_column="target")
        _, manifest_path = r.export(
            dataset_name="m", output_dir=tmp_path, target_column="target"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["verification"]["passed"] is True
        assert manifest["shape"]["rows"] == len(r.df)
        assert len(manifest["columns"]) == len(r.df.columns)
        assert manifest["repair_log"]

    def test_csv_bytes_match_the_written_file(self, messy_df, tmp_path):
        r = DatasetRepairer(messy_df)
        r.auto_repair(target_column="target")
        csv_path, _ = r.export(dataset_name="m", output_dir=tmp_path)
        assert r.to_csv_bytes() == csv_path.read_bytes()


# ---------------------------------------------------------------------------
# Other repair operations
# ---------------------------------------------------------------------------


class TestOtherOperations:
    """Operations that previously mutated state before they could fail."""

    def test_coerce_to_int_keeps_gaps_visible(self):
        # Previously filled unconvertible values with a real 0.
        df = pd.DataFrame({"v": ["1", "2", "abc"]})
        r = DatasetRepairer(df)
        out = r.coerce_types("v", "int")
        assert out["v"].isna().sum() == 1
        assert out["v"].tolist()[:2] == [1, 2]

    def test_unknown_coercion_type_raises_without_history(self):
        df = pd.DataFrame({"v": [1, 2]})
        r = DatasetRepairer(df)
        with pytest.raises(ValueError, match="Unknown target type"):
            r.coerce_types("v", "complex")
        assert r.get_log() == []

    def test_scaling_preserves_missing_values(self):
        df = pd.DataFrame({"v": [1.0, np.nan, 3.0, 5.0]})
        r = DatasetRepairer(df)
        out = r.scale_numeric("v", method="standard")
        assert out["v"].isna().sum() == 1

    def test_rare_category_grouping_preserves_nulls(self):
        df = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 49 + ["rare"] + [None]})
        r = DatasetRepairer(df)
        out = r.handle_rare_categories("c", threshold=0.02)
        assert out["c"].isna().sum() == 1
        assert "Other" in out["c"].tolist()

    def test_encoding_keeps_missing_as_null_not_minus_one(self):
        df = pd.DataFrame({"c": ["x", "y", None, "x"]})
        r = DatasetRepairer(df)
        out = r.encode_categorical("c")
        assert out["c"].isna().sum() == 1
        assert -1 not in out["c"].dropna().tolist()

    def test_preview_does_not_mutate_state(self, messy_df):
        r = DatasetRepairer(messy_df)
        before = int(r.df["age"].isna().sum())
        preview = r.preview_repair("impute_missing", column="age", strategy="median")
        assert int(preview["age"].isna().sum()) == 0
        assert int(r.df["age"].isna().sum()) == before
        assert r.get_log() == []


class TestDatetimeRepair:
    """Dates must survive imputation as dates.

    A real cafe-sales export exposed this: pd.to_numeric turns a datetime
    column into nanosecond integers, so the "median date" came back as
    1.6875648e+18 and 460 rows were written to CSV as that number.
    """

    @pytest.fixture
    def dated(self):
        dates = pd.to_datetime(
            ["2023-01-05", "2023-02-06", None, "2023-03-07", None, "2023-04-08"]
        )
        return pd.DataFrame({"id": range(6), "seen_on": dates})

    def test_median_fill_keeps_datetime_dtype(self, dated):
        r = DatasetRepairer(dated)
        out = r.impute_missing("seen_on", strategy="median")
        assert pd.api.types.is_datetime64_any_dtype(out["seen_on"].dtype)
        assert out["seen_on"].isna().sum() == 0

    def test_fill_value_is_a_real_date(self, dated):
        r = DatasetRepairer(dated)
        r.impute_missing("seen_on", strategy="median")
        filled = str(r.get_log()[-1]["details"]["fill_value"])
        assert "e+" not in filled.lower(), f"nanosecond integer leaked: {filled}"
        assert filled.startswith("2023-")

    def test_mean_fill_also_stays_a_date(self, dated):
        r = DatasetRepairer(dated)
        out = r.impute_missing("seen_on", strategy="mean")
        assert pd.api.types.is_datetime64_any_dtype(out["seen_on"].dtype)

    def test_auto_repair_keeps_dates_readable_on_export(self, dated, tmp_path):
        r = DatasetRepairer(dated)
        r.auto_repair()
        csv_path, _ = r.export(dataset_name="dates", output_dir=tmp_path)
        text = csv_path.read_text(encoding="utf-8-sig")
        assert "e+18" not in text
        assert "2023-" in text
