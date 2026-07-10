"""
Tests for the DatasetMetadata module.
"""

import pytest

from src.utils.dataset_metadata import DatasetMetadata


class TestDatasetMetadata:
    """Test suite for DatasetMetadata."""

    def test_default_creation(self):
        """DatasetMetadata should create with sensible defaults."""
        meta = DatasetMetadata()
        assert meta.dataset_id is not None
        assert meta.filename == ""
        assert meta.shape == (0, 0)
        assert meta.column_names == []
        assert meta.column_types == {}
        assert meta.size_bytes == 0
        assert meta.checksum == ""
        assert meta.upload_timestamp is not None
        assert meta.description is None

    def test_custom_creation(self):
        """DatasetMetadata should accept custom values."""
        meta = DatasetMetadata(
            dataset_id="test-123",
            filename="data.csv",
            shape=(100, 5),
            column_names=["a", "b", "c", "d", "e"],
            column_types={"a": "int64", "b": "float64"},
            size_bytes=2048,
            checksum="abc123",
            source_format=".csv",
            description="Test dataset",
            tags=("test", "sample"),
            extra_metadata={"source": "manual"},
        )
        assert meta.dataset_id == "test-123"
        assert meta.filename == "data.csv"
        assert meta.num_rows == 100
        assert meta.num_columns == 5
        assert meta.source_format == ".csv"
        assert meta.description == "Test dataset"
        assert "test" in meta.tags
        assert meta.extra_metadata["source"] == "manual"

    def test_num_rows(self):
        """num_rows should return the row count."""
        meta = DatasetMetadata(shape=(50, 10))
        assert meta.num_rows == 50

    def test_num_columns(self):
        """num_columns should return the column count."""
        meta = DatasetMetadata(shape=(50, 10))
        assert meta.num_columns == 10

    def test_size_kb(self):
        """size_kb should convert bytes to KB."""
        meta = DatasetMetadata(size_bytes=2048)
        assert meta.size_kb == 2.0

    def test_size_mb(self):
        """size_mb should convert bytes to MB."""
        meta = DatasetMetadata(size_bytes=1048576)
        assert meta.size_mb == 1.0

    def test_summary(self):
        """summary should return a non-empty string with key info."""
        meta = DatasetMetadata(
            filename="test.csv",
            shape=(100, 5),
            size_bytes=2048,
            source_format=".csv",
            checksum="abcdef1234567890abcdef1234567890",
        )
        summary = meta.summary
        assert isinstance(summary, str)
        assert "test.csv" in summary
        assert "100" in summary
        assert "5" in summary

    def test_summary_with_description(self):
        """summary should include the description if provided."""
        meta = DatasetMetadata(
            filename="data.csv",
            description="Important dataset",
        )
        summary = meta.summary
        assert "Important dataset" in summary

    def test_summary_with_tags(self):
        """summary should include tags if provided."""
        meta = DatasetMetadata(
            filename="data.csv",
            tags=("ml", "production"),
        )
        summary = meta.summary
        assert "ml" in summary
        assert "production" in summary

    def test_to_dict(self):
        """to_dict() should return a JSON-serializable dict."""
        meta = DatasetMetadata(
            dataset_id="test-456",
            filename="data.csv",
            shape=(100, 5),
            column_names=["a", "b"],
            column_types={"a": "int64"},
            size_bytes=1024,
            checksum="def789",
            source_format=".csv",
            description="Test",
            tags=("tag1",),
            extra_metadata={"key": "value"},
        )
        d = meta.to_dict()
        assert d["dataset_id"] == "test-456"
        assert d["filename"] == "data.csv"
        assert d["num_rows"] == 100
        assert d["num_columns"] == 5
        # Check list conversion
        assert d["shape"] == [100, 5]
        assert d["tags"] == ["tag1"]
        assert d["extra_metadata"]["key"] == "value"

    def test_from_dict(self):
        """from_dict() should create metadata from a dict."""
        data = {
            "dataset_id": "from-dict",
            "filename": "imported.csv",
            "shape": [200, 10],
            "column_names": ["col1", "col2"],
            "column_types": {"col1": "float64"},
            "size_bytes": 4096,
            "checksum": "aaaabbbb",
            "source_format": ".csv",
            "description": "Imported",
            "tags": ["imported", "test"],
            "extra_metadata": {"version": 2},
        }
        meta = DatasetMetadata.from_dict(data)
        assert meta.dataset_id == "from-dict"
        assert meta.filename == "imported.csv"
        assert meta.num_rows == 200
        assert meta.num_columns == 10
        assert tuple(meta.tags) == ("imported", "test")
        assert meta.extra_metadata["version"] == 2

    def test_from_dict_with_defaults(self):
        """from_dict() should use defaults for missing fields."""
        meta = DatasetMetadata.from_dict({})
        assert meta.dataset_id is not None
        assert meta.filename == ""
        assert meta.shape == (0, 0)

    def test_immutable(self):
        """DatasetMetadata should be immutable (frozen dataclass)."""
        meta = DatasetMetadata()
        with pytest.raises(AttributeError):
            meta.filename = "new_name.csv"  # type: ignore[misc]

    def test_unique_ids(self):
        """Each instance should have a unique dataset_id."""
        meta1 = DatasetMetadata()
        meta2 = DatasetMetadata()
        assert meta1.dataset_id != meta2.dataset_id

    def test_default_upload_timestamp(self):
        """upload_timestamp should be a valid ISO string."""
        meta = DatasetMetadata()
        assert "T" in meta.upload_timestamp  # ISO format
        assert meta.upload_timestamp.endswith("+00:00") or "+" in meta.upload_timestamp
