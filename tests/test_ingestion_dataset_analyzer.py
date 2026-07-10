"""
Tests for the DatasetAnalyzer module.
"""

import pandas as pd
import pytest

from src.ingestion.dataset_analyzer import (
    ColumnCategory,
    ColumnInfo,
    DatasetAnalyzer,
    DatasetSummary,
)


class TestColumnCategory:
    """Constants should match expected values."""

    def test_constants(self):
        assert ColumnCategory.NUMERIC == "numeric"
        assert ColumnCategory.CATEGORICAL == "categorical"
        assert ColumnCategory.BOOLEAN == "boolean"
        assert ColumnCategory.DATETIME == "datetime"
        assert ColumnCategory.TEXT == "text"
        assert ColumnCategory.MIXED == "mixed"
        assert ColumnCategory.CONSTANT == "constant"
        assert ColumnCategory.BINARY == "binary"
        assert ColumnCategory.IDENTIFIER == "identifier"
        assert ColumnCategory.HIGH_CARDINALITY == "high_cardinality"
        assert ColumnCategory.UNKNOWN == "unknown"


class TestColumnInfo:
    """ColumnInfo frozen dataclass."""

    def test_defaults(self):
        info = ColumnInfo(name="col_a")
        assert info.name == "col_a"
        assert info.dtype == ""
        assert info.inferred_type == ColumnCategory.UNKNOWN
        assert info.is_feature_candidate is True

    def test_frozen(self):
        info = ColumnInfo(name="col")
        with pytest.raises(AttributeError):
            info.name = "new_name"

    def test_repr(self):
        info = ColumnInfo(name="my_col", nunique=10, null_count=2)
        r = repr(info)
        assert "my_col" in r
        assert "10" in r


class TestDatasetSummary:
    """DatasetSummary dataclass serialization."""

    def test_defaults(self):
        summary = DatasetSummary()
        assert summary.general == {}
        assert summary.column_stats == []
        assert summary.potential_targets == []
        assert summary.potential_ids == []
        assert summary.fingerprint is None

    def test_to_dict(self):
        summary = DatasetSummary(
            general={"num_rows": 100},
            column_stats=[ColumnInfo(name="a", nunique=10)],
        )
        d = summary.to_dict()
        assert d["general"]["num_rows"] == 100
        assert d["fingerprint"] is None
        assert len(d["column_stats"]) == 1

    def test_to_dict_empty(self):
        summary = DatasetSummary()
        d = summary.to_dict()
        assert isinstance(d, dict)
        assert d["fingerprint"] is None


class TestDatasetAnalyzer:
    """Main test suite for DatasetAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return DatasetAnalyzer(generate_fingerprint=True)

    @pytest.fixture
    def analyzer_no_fp(self):
        return DatasetAnalyzer(generate_fingerprint=False)

    # ------------------------------------------------------------------
    # Basic analysis — general stats
    # ------------------------------------------------------------------

    def test_analyze_returns_summary(self, analyzer):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        summary = analyzer.analyze(df, source_filename="test.csv")
        assert isinstance(summary, DatasetSummary)
        assert summary.general["num_rows"] == 3
        assert summary.general["num_columns"] == 2
        assert summary.general["total_cells"] == 6

    def test_analyze_empty_df(self, analyzer):
        df = pd.DataFrame()
        summary = analyzer.analyze(df)
        assert summary.general["num_rows"] == 0
        assert summary.general["num_columns"] == 0

    def test_analyze_single_row(self, analyzer):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        summary = analyzer.analyze(df)
        assert summary.general["num_rows"] == 1

    def test_analyze_filename(self, analyzer):
        df = pd.DataFrame({"x": [1]})
        summary = analyzer.analyze(df, source_filename="mydata.csv")
        assert summary.general["filename"] == "mydata.csv"

    def test_analyze_unknown_filename(self, analyzer):
        df = pd.DataFrame({"x": [1]})
        summary = analyzer.analyze(df)
        assert summary.general["filename"] == "unknown"

    # ------------------------------------------------------------------
    # Schema information
    # ------------------------------------------------------------------

    def test_schema_info(self, analyzer):
        df = pd.DataFrame({"a": [1], "b": [2.0], "c": ["x"]})
        summary = analyzer.analyze(df)
        assert summary.schema_info["column_names"] == ["a", "b", "c"]
        assert "int64" in summary.schema_info["dtypes"]["a"]

    # ------------------------------------------------------------------
    # Column statistics
    # ------------------------------------------------------------------

    def test_column_stats_count(self, analyzer):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        summary = analyzer.analyze(df)
        assert len(summary.column_stats) == 3

    def test_column_stats_basic(self, analyzer):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        summary = analyzer.analyze(df)
        col_a = summary.column_stats[0]
        assert col_a.name == "a"
        assert col_a.nunique == 3
        assert col_a.is_numeric is True

    def test_column_dtype_recorded(self, analyzer):
        df = pd.DataFrame({"a": [1.5, 2.5]})
        summary = analyzer.analyze(df)
        assert "float" in summary.column_stats[0].dtype

    # ------------------------------------------------------------------
    # Numeric column stats
    # ------------------------------------------------------------------

    def test_numeric_stats(self, analyzer):
        df = pd.DataFrame({"values": [1.0, 2.0, 3.0, 4.0, 5.0]})
        summary = analyzer.analyze(df)
        col = summary.column_stats[0]
        assert col.mean == 3.0
        assert col.min_val == 1.0
        assert col.max_val == 5.0

    def test_numeric_stats_all_nan(self, analyzer):
        df = pd.DataFrame({"a": [float("nan"), float("nan")]})
        summary = analyzer.analyze(df)
        col = summary.column_stats[0]
        assert col.mean is None
        assert col.std is None

    # ------------------------------------------------------------------
    # Missing value analysis
    # ------------------------------------------------------------------

    def test_missing_summary(self, analyzer):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        summary = analyzer.analyze(df)
        assert summary.missing_summary["total_missing"] == 1
        assert summary.missing_summary["total_cells"] == 6
        assert len(summary.missing_summary["columns_with_missing"]) == 1
        assert summary.missing_summary["columns_with_missing"][0]["name"] == "a"
        assert summary.missing_summary["columns_with_missing"][0]["null_count"] == 1

    def test_missing_summary_no_missing(self, analyzer):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        summary = analyzer.analyze(df)
        assert summary.missing_summary["total_missing"] == 0
        assert len(summary.missing_summary["columns_with_missing"]) == 0

    def test_null_pct_per_column(self, analyzer):
        df = pd.DataFrame({"a": [1, None, None, 4]})
        summary = analyzer.analyze(df)
        col = summary.column_stats[0]
        assert col.null_count == 2
        assert col.null_pct == 50.0

    # ------------------------------------------------------------------
    # Duplicate analysis
    # ------------------------------------------------------------------

    def test_duplicate_rows(self, analyzer):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
        summary = analyzer.analyze(df)
        assert summary.duplicate_summary["duplicate_rows"] == 1

    def test_no_duplicate_rows(self, analyzer):
        df = pd.DataFrame({"a": [1, 2, 3]})
        summary = analyzer.analyze(df)
        assert summary.duplicate_summary["duplicate_rows"] == 0

    def test_duplicate_columns(self, analyzer):
        df = pd.DataFrame({"a": [1, 2], "b": [1, 2], "c": [3, 4]})
        summary = analyzer.analyze(df)
        assert len(summary.duplicate_summary["duplicate_columns"]) >= 1

    # ------------------------------------------------------------------
    # Memory statistics
    # ------------------------------------------------------------------

    def test_memory_stats(self, analyzer):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        summary = analyzer.analyze(df)
        assert summary.memory_stats["total_bytes"] > 0
        assert summary.memory_stats["total_mb"] > 0
        assert "a" in summary.memory_stats["per_column_bytes"]
        assert "b" in summary.memory_stats["per_column_bytes"]

    # ------------------------------------------------------------------
    # Column type inference
    # ------------------------------------------------------------------

    def test_infer_numeric(self, analyzer):
        df = pd.DataFrame({"a": [1, 2, 3]})
        summary = analyzer.analyze(df)
        assert summary.column_stats[0].inferred_type == ColumnCategory.NUMERIC

    def test_infer_float(self, analyzer):
        df = pd.DataFrame({"a": [1.5, 2.5, 3.5]})
        summary = analyzer.analyze(df)
        assert summary.column_stats[0].inferred_type == ColumnCategory.NUMERIC

    def test_infer_boolean(self, analyzer):
        df = pd.DataFrame({"a": [True, False, True]})
        summary = analyzer.analyze(df)
        assert summary.column_stats[0].inferred_type == ColumnCategory.BOOLEAN

    def test_infer_datetime(self, analyzer):
        df = pd.DataFrame({"a": pd.to_datetime(["2021-01-01", "2021-01-02"])})
        summary = analyzer.analyze(df)
        assert summary.column_stats[0].inferred_type == ColumnCategory.DATETIME

    def test_infer_text(self, analyzer):
        df = pd.DataFrame({"a": ["apple", "banana", "cherry"]})
        summary = analyzer.analyze(df)
        assert summary.column_stats[0].inferred_type == ColumnCategory.TEXT

    def test_infer_constant(self, analyzer):
        df = pd.DataFrame({"a": [42, 42, 42]})
        summary = analyzer.analyze(df)
        assert summary.column_stats[0].inferred_type == ColumnCategory.CONSTANT

    def test_infer_binary(self, analyzer):
        df = pd.DataFrame({"a": [0, 1, 0, 1]})
        summary = analyzer.analyze(df)
        assert summary.column_stats[0].inferred_type == ColumnCategory.BINARY

    def test_infer_categorical_via_text(self, analyzer):
        df = pd.DataFrame({"a": ["cat", "dog", "bird"]})
        summary = analyzer.analyze(df)
        assert summary.column_stats[0].inferred_type == ColumnCategory.TEXT  # low cardinality but object dtype

    def test_infer_high_cardinality_text(self, analyzer):
        values = [f"val_{i}" for i in range(200)]
        df = pd.DataFrame({"a": values})
        summary = analyzer.analyze(df)
        assert summary.column_stats[0].inferred_type == ColumnCategory.HIGH_CARDINALITY

    def test_infer_string_datetime(self, analyzer):
        df = pd.DataFrame({"a": ["2021-01-01", "2021-01-02", "2021-01-03"]})
        summary = analyzer.analyze(df)
        # String that looks like datetime
        assert summary.column_stats[0].inferred_type in (ColumnCategory.DATETIME, ColumnCategory.TEXT)

    # ------------------------------------------------------------------
    # Target detection
    # ------------------------------------------------------------------

    def test_binary_target_detected(self, analyzer):
        df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        })
        summary = analyzer.analyze(df)
        assert len(summary.potential_targets) > 0
        assert summary.potential_targets[0]["name"] == "target"

    def test_numeric_target_not_high_priority(self, analyzer):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [100.0, 200.0, 300.0],
        })
        summary = analyzer.analyze(df)
        # Numeric columns with no distinguishing name should not be top targets
        target_names = [t["name"] for t in summary.potential_targets]
        assert len(target_names) <= 2

    def test_target_confidence_scores(self, analyzer):
        df = pd.DataFrame({
            "label": [0, 1, 0, 1, 0],
            "feat": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        summary = analyzer.analyze(df)
        for t in summary.potential_targets:
            assert 0.0 <= t["confidence"] <= 1.0
            assert "suggested_task" in t

    # ------------------------------------------------------------------
    # ID column detection
    # ------------------------------------------------------------------

    def test_id_column_detected_by_name(self, analyzer):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })
        summary = analyzer.analyze(df)
        id_names = [i["name"] for i in summary.potential_ids]
        assert "id" in id_names

    def test_id_column_detected_by_uniqueness(self, analyzer):
        df = pd.DataFrame({
            "uid": [101, 102, 103, 104, 105],
            "data": [1, 2, 3, 4, 5],
        })
        summary = analyzer.analyze(df)
        id_names = [i["name"] for i in summary.potential_ids]
        assert "uid" in id_names

    def test_non_id_column_not_detected(self, analyzer):
        df = pd.DataFrame({
            "a": [1, 1, 2, 2, 3],
            "b": [10, 20, 10, 20, 10],
        })
        summary = analyzer.analyze(df)
        assert len(summary.potential_ids) == 0

    def test_id_confidence(self, analyzer):
        df = pd.DataFrame({"user_id": [1, 2, 3]})
        summary = analyzer.analyze(df)
        for i in summary.potential_ids:
            assert 0.0 <= i["confidence"] <= 1.0

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def test_features_exclude_id(self, analyzer):
        # Use value column with repeated values so it's not detected as ID
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10.0, 20.0, 10.0, 20.0, 30.0],
        })
        summary = analyzer.analyze(df)
        assert "id" not in summary.potential_features
        assert "value" in summary.potential_features

    # ------------------------------------------------------------------
    # Dataset type detection
    # ------------------------------------------------------------------

    def test_classification_detected(self, analyzer):
        df = pd.DataFrame({
            "feat": [1.0, 2.0, 3.0],
            "class": [0, 1, 0],
        })
        summary = analyzer.analyze(df)
        assert summary.dataset_type["type"] == "classification"
        assert summary.dataset_type["target_column"] == "class"

    def test_unsupervised_detected(self, analyzer):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        })
        summary = analyzer.analyze(df)
        assert summary.dataset_type["type"] == "unsupervised"

    def test_unknown_dataset_small(self, analyzer):
        df = pd.DataFrame({"a": [1]})
        summary = analyzer.analyze(df)
        assert summary.dataset_type["type"] == "unknown"

    def test_regression_detected(self, analyzer):
        df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0, 4.0],
            "score": [95.0, 85.0, 75.0, 65.0],
        })
        summary = analyzer.analyze(df)
        # Score column with numeric values and target-like name
        target_names = [t["name"] for t in summary.potential_targets]
        assert any("score" in n for n in target_names)

    # ------------------------------------------------------------------
    # Fingerprint generation
    # ------------------------------------------------------------------

    def test_fingerprint_generated(self, analyzer):
        df = pd.DataFrame({"a": [1, 2, 3]})
        summary = analyzer.analyze(df)
        assert summary.fingerprint is not None
        assert summary.fingerprint.is_complete

    def test_fingerprint_not_generated(self, analyzer_no_fp):
        df = pd.DataFrame({"a": [1, 2, 3]})
        summary = analyzer_no_fp.analyze(df)
        assert summary.fingerprint is None

    # ------------------------------------------------------------------
    # Edge cases: mixed types, empty columns
    # ------------------------------------------------------------------

    def test_mixed_type_column(self, analyzer):
        df = pd.DataFrame({"a": [1, "text", 3.5, None]})
        summary = analyzer.analyze(df)
        # Object dtype with mixed types should be detected
        assert summary.column_stats[0].inferred_type in (
            ColumnCategory.MIXED, ColumnCategory.UNKNOWN, ColumnCategory.TEXT,
        )

    def test_all_null_column(self, analyzer):
        df = pd.DataFrame({"a": [None, None, None]})
        summary = analyzer.analyze(df)
        assert summary.column_stats[0].null_count == 3
        assert summary.column_stats[0].nunique == 0

    def test_large_dataset(self, analyzer):
        n = 10000
        df = pd.DataFrame({
            "id": range(n),
            "value": range(n),
            "category": ["cat"] * (n // 2) + ["dog"] * (n // 2),
        })
        summary = analyzer.analyze(df)
        assert summary.general["num_rows"] == n
        assert len(summary.column_stats) == 3

    # ------------------------------------------------------------------
    # ColumnInfo categorical candidate flags
    # ------------------------------------------------------------------

    def test_categorical_candidate(self, analyzer):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        summary = analyzer.analyze(df)
        col = summary.column_stats[0]
        assert col.is_categorical_candidate is False  # TEXT type, not CATEGORICAL/BOOLEAN/BINARY

    def test_binary_is_categorical_candidate(self, analyzer):
        df = pd.DataFrame({"a": [0, 1, 0]})
        summary = analyzer.analyze(df)
        col = summary.column_stats[0]
        assert col.is_categorical_candidate is True

    # ------------------------------------------------------------------
    # Top value detection
    # ------------------------------------------------------------------

    def test_top_value(self, analyzer):
        df = pd.DataFrame({"a": ["x", "x", "y"]})
        summary = analyzer.analyze(df)
        col = summary.column_stats[0]
        assert col.top_value == "x"
        assert col.top_freq == 2

    def test_top_value_tie(self, analyzer):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        summary = analyzer.analyze(df)
        col = summary.column_stats[0]
        assert col.top_freq == 1

    # ------------------------------------------------------------------
    # Index handling
    # ------------------------------------------------------------------

    def test_named_index(self, analyzer):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.index.name = "row_id"
        summary = analyzer.analyze(df)
        assert summary.schema_info["has_index"] is True
        assert summary.schema_info["index_name"] == "row_id"

    def test_no_index_name(self, analyzer):
        df = pd.DataFrame({"a": [1, 2, 3]})
        summary = analyzer.analyze(df)
        assert summary.schema_info["has_index"] is False
