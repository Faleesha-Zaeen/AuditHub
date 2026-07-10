"""
Tests for the DatasetFingerprint and FingerprintGenerator modules.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.dataset_fingerprint import DatasetFingerprint, FingerprintGenerator


class TestDatasetFingerprint:
    """Tests for the immutable DatasetFingerprint dataclass."""

    def test_defaults(self):
        """Should create a fingerprint with default values."""
        fp = DatasetFingerprint()
        assert fp.fingerprint_id is not None
        assert len(fp.fingerprint_id) == 36  # UUID4
        assert fp.dataset_checksum == ""
        assert fp.created_at is not None

    def test_frozen(self):
        """Should be immutable."""
        fp = DatasetFingerprint()
        with pytest.raises(AttributeError):
            fp.fingerprint_id = "changed"

    def test_is_complete_false(self):
        """Should return False when hashes are empty."""
        fp = DatasetFingerprint()
        assert fp.is_complete is False

    def test_is_complete_true(self):
        """Should return True when all core hashes are populated."""
        fp = DatasetFingerprint(
            dataset_checksum="abc",
            schema_hash="abc",
            column_hash="abc",
            shape_hash="abc",
            file_hash="abc",
            content_hash="abc",
        )
        assert fp.is_complete is True

    def test_is_complete_without_content(self):
        """Should be complete with file_hash but no content_hash."""
        fp = DatasetFingerprint(
            schema_hash="abc",
            column_hash="abc",
            shape_hash="abc",
            file_hash="abc",
        )
        assert fp.is_complete is True

    def test_summary_output(self):
        """Summary should include key fields."""
        fp = DatasetFingerprint(
            fingerprint_id="test-id",
            schema_hash="abc123def456",
        )
        s = fp.summary
        assert "test-id" in s
        assert "abc123def456" in s

    def test_to_dict(self):
        """Should serialize to a dictionary."""
        fp = DatasetFingerprint(
            fingerprint_id="fp-1",
            dataset_checksum="sum1",
            schema_hash="sch1",
            column_hash="col1",
            shape_hash="sha1",
            file_hash="file1",
            content_hash="con1",
        )
        d = fp.to_dict()
        assert d["fingerprint_id"] == "fp-1"
        assert d["dataset_checksum"] == "sum1"
        assert d["is_complete"] is True
        assert "extra" in d

    def test_to_dict_empty_fingerprint(self):
        """Empty fingerprint should still serialize."""
        fp = DatasetFingerprint()
        d = fp.to_dict()
        assert d["is_complete"] is False
        assert d["extra"] == {}

    def test_extra_metadata_preserved(self):
        """Extra dict should be preserved in to_dict."""
        fp = DatasetFingerprint(extra={"source": "test", "version": 2})
        d = fp.to_dict()
        assert d["extra"]["source"] == "test"
        assert d["extra"]["version"] == 2


class TestFingerprintGenerator:
    """Tests for FingerprintGenerator."""

    @pytest.fixture
    def generator(self):
        return FingerprintGenerator()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
            "c": ["x", "y", "z"],
        })

    # ------------------------------------------------------------------
    # from_dataframe
    # ------------------------------------------------------------------

    def test_from_dataframe_returns_fingerprint(self, generator, sample_df):
        fp = generator.from_dataframe(sample_df)
        assert isinstance(fp, DatasetFingerprint)
        assert fp.is_complete

    def test_from_dataframe_schema_hash(self, generator, sample_df):
        fp = generator.from_dataframe(sample_df)
        assert len(fp.schema_hash) == 32  # MD5
        assert fp.schema_hash != ""

    def test_from_dataframe_column_hash(self, generator, sample_df):
        fp = generator.from_dataframe(sample_df)
        assert len(fp.column_hash) == 32
        assert fp.column_hash != ""

    def test_from_dataframe_shape_hash(self, generator, sample_df):
        fp = generator.from_dataframe(sample_df)
        assert len(fp.shape_hash) == 32
        assert fp.shape_hash != ""

    def test_from_dataframe_content_hash(self, generator, sample_df):
        fp = generator.from_dataframe(sample_df)
        assert len(fp.content_hash) == 64  # SHA-256

    def test_from_dataframe_deterministic(self, generator, sample_df):
        fp1 = generator.from_dataframe(sample_df)
        fp2 = generator.from_dataframe(sample_df)
        assert fp1.schema_hash == fp2.schema_hash
        assert fp1.column_hash == fp2.column_hash
        assert fp1.shape_hash == fp2.shape_hash
        assert fp1.content_hash == fp2.content_hash

    def test_from_dataframe_different_schema_differs(self, generator):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "c": [3, 4]})
        fp1 = generator.from_dataframe(df1)
        fp2 = generator.from_dataframe(df2)
        assert fp1.schema_hash != fp2.schema_hash

    def test_from_dataframe_with_file_path(self, generator, sample_df, tmp_path):
        file_path = tmp_path / "data.csv"
        sample_df.to_csv(file_path, index=False)
        fp = generator.from_dataframe(sample_df, file_path=file_path)
        assert fp.dataset_checksum != ""
        assert fp.file_hash == fp.dataset_checksum
        assert "file_path" in fp.extra

    def test_from_dataframe_without_file_path(self, generator, sample_df):
        fp = generator.from_dataframe(sample_df)
        assert fp.file_hash == fp.content_hash  # Falls back to content hash

    def test_from_dataframe_empty(self, generator):
        df = pd.DataFrame()
        fp = generator.from_dataframe(df)
        assert fp.is_complete
        assert fp.extra["num_rows"] == 0
        assert fp.extra["num_columns"] == 0

    def test_from_dataframe_single_column(self, generator):
        df = pd.DataFrame({"a": [1, 2, 3]})
        fp = generator.from_dataframe(df)
        assert fp.column_hash is not None

    # ------------------------------------------------------------------
    # from_file
    # ------------------------------------------------------------------

    def test_from_file(self, generator, tmp_path):
        path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
        fp = generator.from_file(path)
        assert isinstance(fp, DatasetFingerprint)
        assert fp.dataset_checksum != ""
        assert len(fp.dataset_checksum) == 64  # SHA-256

    def test_from_file_file_not_found(self, generator, tmp_path):
        with pytest.raises(FileNotFoundError):
            generator.from_file(tmp_path / "nonexistent.csv")

    def test_from_file_with_dataframe(self, generator, tmp_path):
        path = tmp_path / "test.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(path, index=False)
        fp = generator.from_file(path, df=df)
        assert fp.is_complete
        assert fp.schema_hash != ""
        assert fp.content_hash != ""

    def test_from_file_without_dataframe(self, generator, tmp_path):
        path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
        fp = generator.from_file(path, df=None)
        assert fp.schema_hash == ""
        assert fp.content_hash == ""
        assert fp.is_complete is False

    def test_from_file_checks_consistency(self, generator, tmp_path):
        """File hash should be consistent for same file content."""
        path = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(path, index=False)
        fp1 = generator.from_file(path)
        fp2 = generator.from_file(path)
        assert fp1.dataset_checksum == fp2.dataset_checksum

    def test_from_file_extra_metadata(self, generator, tmp_path):
        path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1]}).to_csv(path, index=False)
        fp = generator.from_file(path)
        assert "file_path" in fp.extra
        assert "file_size" in fp.extra
        assert fp.extra["file_size"] > 0

    # ------------------------------------------------------------------
    # Hashing consistency
    # ------------------------------------------------------------------

    def test_consistent_hash_across_calls(self, generator):
        """Same columns/dtypes should produce same schema hash."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7.0, 8.0]})
        fp1 = generator.from_dataframe(df1)
        fp2 = generator.from_dataframe(df2)
        # Schema hash should match (same column names + dtypes)
        assert fp1.schema_hash == fp2.schema_hash
        # Content hash should differ (different values)
        assert fp1.content_hash != fp2.content_hash

    def test_shape_hash_same_shape(self, generator):
        """Same-shaped DataFrames should have same shape hash."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        fp1 = generator.from_dataframe(df1)
        fp2 = generator.from_dataframe(df2)
        assert fp1.shape_hash == fp2.shape_hash

    def test_shape_hash_different_shape(self, generator):
        """Different-shaped DataFrames should have different shape hash."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})
        fp1 = generator.from_dataframe(df1)
        fp2 = generator.from_dataframe(df2)
        assert fp1.shape_hash != fp2.shape_hash

    def test_column_hash_differs_by_order(self, generator):
        """Column hash should depend on column order."""
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"b": [1], "a": [2]})
        fp1 = generator.from_dataframe(df1)
        fp2 = generator.from_dataframe(df2)
        assert fp1.column_hash != fp2.column_hash

    # ------------------------------------------------------------------
    # Large DataFrame handling
    # ------------------------------------------------------------------

    def test_large_dataframe_content_hash(self, generator):
        n = 5000
        df = pd.DataFrame({
            "id": range(n),
            "value": [float(i) for i in range(n)],
            "label": ["A", "B"] * (n // 2),
        })
        fp = generator.from_dataframe(df)
        assert fp.is_complete
        assert len(fp.content_hash) == 64

    # ------------------------------------------------------------------
    # UUID stability check
    # ------------------------------------------------------------------

    def test_fingerprint_id_is_uuid4(self, generator, sample_df):
        fp = generator.from_dataframe(sample_df)
        parts = fp.fingerprint_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
