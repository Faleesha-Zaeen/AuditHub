"""
Tests for src.lineage
======================

Covers version comparison, rename detection, registry versioning (including
the in-place migration of an existing database) and pipeline lineage.
"""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.lineage.comparator import ChangeType, VersionComparator, VersionComparison
from src.lineage.tracker import LineageTracker
from src.utils.dataset_metadata import DatasetMetadata
from src.utils.dataset_registry import DatasetRegistry


@pytest.fixture
def rng():
    return np.random.default_rng(20260905)


@pytest.fixture
def v1(rng):
    """The older version of a customer dataset."""
    n = 600
    df = pd.DataFrame({
        "customer_id": range(n),
        "age": rng.normal(45, 12, n),
        "income": rng.normal(60000, 15000, n),
        "legacy_id": rng.integers(1, 99999, n),
        "city": rng.choice(["Delhi", "Mumbai", "Pune"], n),
        "churn": rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
    })
    df.loc[rng.choice(n, 12, replace=False), "age"] = np.nan
    return df


def _changes_of(comparison: VersionComparison, change_type: str):
    return [c for c in comparison.column_changes if c.change_type == change_type]


class TestHeadlineMetrics:
    """Row, column, missing and duplicate deltas."""

    def test_row_and_column_counts(self, v1):
        v2 = pd.concat([v1, v1.head(20)], ignore_index=True)
        v2["new_col"] = 1
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert result.left_rows == len(v1)
        assert result.right_rows == len(v1) + 20
        assert result.row_delta == 20
        assert result.column_delta == 1

    def test_duplicate_count_change(self, v1):
        v2 = pd.concat([v1, v1.head(37)], ignore_index=True)
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert result.left_duplicates == 0
        assert result.right_duplicates == 37

    def test_missing_percentage_change(self, v1):
        v2 = v1.copy()
        v2.loc[v2.sample(frac=0.3, random_state=1).index, "income"] = np.nan
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert result.right_missing_pct > result.left_missing_pct

    def test_headline_frame_has_four_metrics(self, v1):
        result = VersionComparator(detect_drift=False).compare(v1, v1.copy())
        assert len(result.headline_frame()) == 4


class TestSchemaChanges:
    """Added, removed and retyped columns."""

    def test_added_and_removed_columns(self, v1):
        v2 = v1.drop(columns=["legacy_id"]).copy()
        v2["customer_segment"] = "A"
        v2["account_age"] = 12

        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert result.removed_columns == ["legacy_id"]
        assert set(result.added_columns) == {"customer_segment", "account_age"}
        assert len(_changes_of(result, ChangeType.COLUMN_ADDED)) == 2
        assert len(_changes_of(result, ChangeType.COLUMN_REMOVED)) == 1

    def test_removing_the_target_is_critical(self, v1):
        v2 = v1.drop(columns=["churn"])
        result = VersionComparator(detect_drift=False).compare(
            v1, v2, target_column="churn"
        )
        removal = _changes_of(result, ChangeType.COLUMN_REMOVED)[0]
        assert removal.severity == "CRITICAL"

    def test_numeric_to_text_is_critical(self, v1):
        v2 = v1.copy()
        v2["income"] = v2["income"].astype(str)
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        change = _changes_of(result, ChangeType.TYPE_CHANGED)[0]
        assert change.severity == "CRITICAL"

    def test_int_to_float_is_only_informational(self, v1):
        v2 = v1.copy()
        v2["legacy_id"] = v2["legacy_id"].astype(float)
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        change = _changes_of(result, ChangeType.TYPE_CHANGED)[0]
        assert change.severity == "INFO"

    def test_cardinality_explosion_is_flagged(self, v1, rng):
        v2 = v1.copy()
        v2["city"] = [f"city_{i}" for i in range(len(v2))]
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert _changes_of(result, ChangeType.CARDINALITY_CHANGED)


class TestRenameDetection:
    """Renames are established from content, never guessed from names."""

    def test_identical_content_is_a_rename(self, v1):
        v2 = v1.rename(columns={"legacy_id": "client_reference"})
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert len(result.renamed_columns) == 1
        rename = result.renamed_columns[0]
        assert rename["from"] == "legacy_id"
        assert rename["to"] == "client_reference"
        assert rename["confidence"] == 1.0

    def test_renamed_column_is_not_double_counted(self, v1):
        v2 = v1.rename(columns={"legacy_id": "client_reference"})
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert result.added_columns == []
        assert result.removed_columns == []

    def test_unrelated_columns_are_not_called_a_rename(self, v1, rng):
        v2 = v1.drop(columns=["legacy_id"]).copy()
        v2["something_else"] = rng.normal(0, 1, len(v2))
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert result.renamed_columns == []
        assert result.removed_columns == ["legacy_id"]
        assert result.added_columns == ["something_else"]

    def test_rename_across_differing_types_is_rejected(self, v1):
        v2 = v1.drop(columns=["legacy_id"]).copy()
        v2["legacy_ref"] = v1["legacy_id"].astype(str)
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert result.renamed_columns == []

    def test_rename_is_reported_as_a_warning(self, v1):
        v2 = v1.rename(columns={"legacy_id": "client_reference"})
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        change = _changes_of(result, ChangeType.COLUMN_RENAMED)[0]
        assert change.severity == "WARNING"


class TestMissingAndTarget:
    """Missing-rate jumps and target changes."""

    def test_missing_jump_is_reported(self, v1):
        v2 = v1.copy()
        v2.loc[v2.sample(frac=0.4, random_state=2).index, "age"] = np.nan
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        change = _changes_of(result, ChangeType.MISSING_CHANGED)[0]
        assert change.column == "age"
        assert change.after > change.before

    def test_missing_change_is_not_double_reported(self, v1):
        # The drift engine also notices a missing-rate jump; the comparison
        # must report it once, not once per engine.
        v2 = v1.copy()
        v2.loc[v2.sample(frac=0.4, random_state=2).index, "age"] = np.nan
        result = VersionComparator(detect_drift=True).compare(v1, v2)
        age_changes = [c for c in result.column_changes if c.column == "age"]
        kinds = [c.change_type for c in age_changes]
        assert kinds.count(ChangeType.DISTRIBUTION_CHANGED) == 0
        assert kinds.count(ChangeType.MISSING_CHANGED) == 1

    def test_target_class_balance_shift(self, v1, rng):
        v2 = v1.copy()
        v2["churn"] = rng.choice(["yes", "no"], len(v2), p=[0.6, 0.4])
        result = VersionComparator(detect_drift=False).compare(
            v1, v2, target_column="churn"
        )
        change = _changes_of(result, ChangeType.TARGET_CHANGED)[0]
        assert "class balance moved" in change.detail

    def test_new_target_class_is_critical(self, v1, rng):
        v2 = v1.copy()
        v2["churn"] = rng.choice(["yes", "no", "maybe"], len(v2))
        result = VersionComparator(detect_drift=False).compare(
            v1, v2, target_column="churn"
        )
        change = _changes_of(result, ChangeType.TARGET_CHANGED)[0]
        assert change.severity == "CRITICAL"
        assert "maybe" in change.detail


class TestDriftIntegration:
    """Distribution comparison reuses the drift engine."""

    def test_distribution_shift_is_reported(self, v1, rng):
        v2 = v1.copy()
        v2["age"] = rng.normal(62, 12, len(v2))
        result = VersionComparator(detect_drift=True).compare(v1, v2)
        assert result.drift is not None
        assert _changes_of(result, ChangeType.DISTRIBUTION_CHANGED)

    def test_drift_can_be_disabled(self, v1, rng):
        v2 = v1.copy()
        v2["age"] = rng.normal(62, 12, len(v2))
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert result.drift is None


class TestComparisonShape:
    """Serialisation for the API, UI and report."""

    def test_to_dict_is_json_serialisable(self, v1):
        v2 = v1.copy()
        v2["extra"] = 1
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        json.dumps(result.to_dict())

    def test_to_frame_matches_change_count(self, v1):
        v2 = v1.drop(columns=["legacy_id"])
        result = VersionComparator(detect_drift=False).compare(v1, v2)
        assert len(result.to_frame()) == len(result.column_changes)

    def test_identical_versions_report_no_changes(self, v1):
        result = VersionComparator(detect_drift=False).compare(v1, v1.copy())
        assert result.column_changes == []
        assert result.row_delta == 0

    def test_missing_input_raises(self, v1):
        with pytest.raises(ValueError):
            VersionComparator().compare(v1, None)


# ---------------------------------------------------------------------------
# Registry versioning
# ---------------------------------------------------------------------------


def _meta(filename: str, rows: int = 10) -> DatasetMetadata:
    return DatasetMetadata(
        filename=filename, shape=(rows, 2),
        column_names=["a", "b"], column_types={"a": "int64", "b": "object"},
    )


class TestRegistryVersioning:
    """Versions are assigned automatically and queryable."""

    def test_versions_increment_within_a_group(self, tmp_path):
        registry = DatasetRegistry(db_path=str(tmp_path / "r.db"))
        registry.register(_meta("customers.csv"))
        registry.register(_meta("customers.csv"))
        registry.register(_meta("customers.csv"))

        versions = registry.list_versions("customers")
        assert [v.extra_metadata["version"] for v in versions] == [1, 2, 3]

    def test_versioned_filenames_share_a_group(self, tmp_path):
        registry = DatasetRegistry(db_path=str(tmp_path / "r.db"))
        registry.register(_meta("sales_v1.csv"))
        registry.register(_meta("sales_v2.csv"))
        assert len(registry.list_versions("sales")) == 2

    def test_different_datasets_get_separate_groups(self, tmp_path):
        registry = DatasetRegistry(db_path=str(tmp_path / "r.db"))
        registry.register(_meta("customers.csv"))
        registry.register(_meta("orders.csv"))
        groups = {g["version_group"] for g in registry.list_version_groups()}
        assert groups == {"customers", "orders"}

    def test_latest_version(self, tmp_path):
        registry = DatasetRegistry(db_path=str(tmp_path / "r.db"))
        registry.register(_meta("customers.csv", rows=10))
        registry.register(_meta("customers.csv", rows=99))
        latest = registry.latest_version("customers")
        assert latest.num_rows == 99
        assert latest.extra_metadata["version"] == 2

    def test_get_specific_version(self, tmp_path):
        registry = DatasetRegistry(db_path=str(tmp_path / "r.db"))
        registry.register(_meta("customers.csv", rows=10))
        registry.register(_meta("customers.csv", rows=99))
        assert registry.get_version("customers", 1).num_rows == 10
        assert registry.get_version("customers", 5) is None

    def test_parent_links_to_previous_version(self, tmp_path):
        registry = DatasetRegistry(db_path=str(tmp_path / "r.db"))
        first = registry.register(_meta("customers.csv"))
        registry.register(_meta("customers.csv"))
        second = registry.latest_version("customers")
        assert second.extra_metadata["parent_dataset_id"] == first

    def test_explicit_version_group_is_honoured(self, tmp_path):
        registry = DatasetRegistry(db_path=str(tmp_path / "r.db"))
        registry.register(_meta("anything.csv"), version_group="my_group")
        assert len(registry.list_versions("my_group")) == 1

    def test_existing_rows_survive_migration(self, tmp_path):
        """A pre-versioning database must gain the columns without losing data."""
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE dataset_registry ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id VARCHAR(36), "
            "filename VARCHAR(512), num_rows INTEGER, num_columns INTEGER, "
            "size_bytes INTEGER, checksum VARCHAR(128), source_format VARCHAR(16), "
            "description TEXT, tags TEXT, column_names TEXT, column_types TEXT, "
            "extra_metadata TEXT, created_at DATETIME, updated_at DATETIME)"
        )
        conn.execute(
            "INSERT INTO dataset_registry (dataset_id, filename, num_rows, num_columns) "
            "VALUES ('legacy-1', 'old.csv', 5, 2)"
        )
        conn.commit()
        conn.close()

        registry = DatasetRegistry(db_path=str(db))
        assert registry.count() == 1
        record = registry.get("legacy-1")
        assert record is not None
        assert record.extra_metadata["version"] == 1
        assert record.extra_metadata["version_group"] == "old.csv"

        # New registrations continue from the backfilled state.
        registry.register(_meta("old.csv"))
        assert registry.count() == 2

    def test_migration_is_idempotent(self, tmp_path):
        db = str(tmp_path / "r.db")
        DatasetRegistry(db_path=db)
        DatasetRegistry(db_path=db)
        registry = DatasetRegistry(db_path=db)
        registry.register(_meta("x.csv"))
        assert registry.count() == 1


# ---------------------------------------------------------------------------
# Lineage tracking
# ---------------------------------------------------------------------------


class TestLineageTracker:
    """Events are recorded and can be replayed per stage and per column."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return LineageTracker(db_path=str(tmp_path / "lineage.db"),
                              dataset_name="demo.csv", dataset_id="ds-1")

    def test_records_and_reads_back(self, tracker):
        tracker.record("ingestion", "Loaded 400 rows", event_type="load")
        trace = tracker.trace()
        assert len(trace.events) == 1
        assert trace.events[0].summary == "Loaded 400 rows"

    def test_column_history_is_ordered(self, tracker):
        tracker.record("ingestion", "143 missing value(s) detected", column="age")
        tracker.record("repair", "median imputation, 143 cell(s) changed", column="age")
        tracker.record("training", "included in training", column="age")

        history = tracker.trace().column_history("age")
        assert [e.stage for e in history] == ["ingestion", "repair", "training"]
        assert "143" in history[0].summary

    def test_unrelated_columns_are_separated(self, tracker):
        tracker.record("repair", "imputed", column="age")
        tracker.record("repair", "imputed", column="income")
        columns = tracker.trace().columns()
        assert set(columns) == {"age", "income"}

    def test_stages_summarise_in_order(self, tracker):
        tracker.record("ingestion", "Loaded")
        tracker.record("repair", "Repaired")
        tracker.record("training", "Trained")
        stages = tracker.trace().stages()
        assert [s["stage"] for s in stages] == ["ingestion", "repair", "training"]

    def test_flow_renders_as_a_chain(self, tracker):
        tracker.record("ingestion", "Loaded")
        tracker.record("repair", "Repaired")
        assert tracker.trace().flow() == "Raw Dataset -> Ingestion -> Repair"

    def test_trace_for_dataset(self, tracker):
        tracker.record("ingestion", "Loaded")
        trace = tracker.trace_for_dataset("ds-1")
        assert trace.events

    def test_list_runs(self, tracker):
        tracker.record("ingestion", "Loaded")
        runs = tracker.list_runs()
        assert runs[0]["run_id"] == tracker.run_id
        assert runs[0]["events"] == 1

    def test_unknown_run_is_empty_not_an_error(self, tracker):
        assert tracker.trace("does-not-exist").events == []

    def test_trace_is_json_serialisable(self, tracker):
        tracker.record("repair", "imputed", column="age", detail={"cells": 143})
        json.dumps(tracker.trace().to_dict())

    def test_record_many(self, tracker):
        tracker.record_many("repair", [
            {"summary": "a", "column": "x"},
            {"summary": "b", "column": "y"},
        ])
        assert len(tracker.trace().events) == 2

    def test_recording_failure_does_not_raise(self, tmp_path):
        # Lineage is an audit trail; a broken store must not stop the pipeline.
        tracker = LineageTracker(db_path=str(tmp_path / "l.db"))
        tracker._Session = lambda: (_ for _ in ()).throw(RuntimeError("db gone"))
        tracker.record("ingestion", "should not raise")


class TestDisplayTablesSerialise:
    """Tables shown in the UI must survive Arrow serialisation.

    Streamlit renders dataframes through Arrow, which rejects an object column
    mixing ints and strings. A real cafe-sales dataset crashed both the Repair
    and Version Compare pages this way -- and only once they had data, which is
    why a render-only test did not catch it.
    """

    def test_headline_frame_is_arrow_safe(self, v1):
        import pyarrow as pa

        v2 = v1.copy()
        v2["extra"] = 1
        comparison = VersionComparator(detect_drift=False).compare(v1, v2)
        pa.Table.from_pandas(comparison.headline_frame())

    def test_changes_frame_is_arrow_safe(self, v1):
        import pyarrow as pa

        v2 = v1.drop(columns=["legacy_id"])
        comparison = VersionComparator(detect_drift=False).compare(v1, v2)
        pa.Table.from_pandas(comparison.to_frame())

    def test_headline_columns_are_all_strings(self, v1):
        comparison = VersionComparator(detect_drift=False).compare(v1, v1.copy())
        frame = comparison.headline_frame()
        for column in ("before", "after", "change"):
            assert all(isinstance(v, str) for v in frame[column])
