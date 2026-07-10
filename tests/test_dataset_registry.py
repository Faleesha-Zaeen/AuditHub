"""
Tests for the DatasetRegistry module.
"""

import tempfile
from pathlib import Path

import pytest

from src.utils.dataset_metadata import DatasetMetadata
from src.utils.dataset_registry import DatasetRegistry
from src.utils.exceptions import DatabaseException


class TestDatasetRegistry:
    """Test suite for DatasetRegistry."""

    @pytest.fixture
    def registry(self) -> DatasetRegistry:
        """Create a DatasetRegistry with a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        registry = DatasetRegistry(db_path=db_path)
        yield registry

        # Cleanup: close engine first so SQLite releases the file lock
        registry.close()
        try:
            Path(db_path).unlink()
        except (PermissionError, OSError):
            pass

    @pytest.fixture
    def sample_metadata(self) -> DatasetMetadata:
        """Create a sample metadata object."""
        return DatasetMetadata(
            dataset_id="test-001",
            filename="data.csv",
            shape=(100, 5),
            column_names=["id", "name", "value", "category", "date"],
            column_types={
                "id": "int64",
                "name": "object",
                "value": "float64",
                "category": "object",
                "date": "datetime64[ns]",
            },
            size_bytes=20480,
            checksum="abcdef1234567890",
            source_format=".csv",
            description="Test dataset",
            tags=("test", "regression"),
            extra_metadata={"source": "manual"},
        )

    def test_register(self, registry: DatasetRegistry, sample_metadata: DatasetMetadata):
        """register() should persist metadata and return the ID."""
        dataset_id = registry.register(sample_metadata)
        assert dataset_id == "test-001"

    def test_get(self, registry: DatasetRegistry, sample_metadata: DatasetMetadata):
        """get() should retrieve metadata by ID."""
        registry.register(sample_metadata)
        retrieved = registry.get("test-001")
        assert retrieved is not None
        assert retrieved.filename == "data.csv"
        assert retrieved.num_rows == 100
        assert retrieved.num_columns == 5
        assert retrieved.checksum == "abcdef1234567890"

    def test_get_not_found(self, registry: DatasetRegistry):
        """get() should return None for unknown IDs."""
        retrieved = registry.get("nonexistent")
        assert retrieved is None

    def test_list_all(self, registry: DatasetRegistry):
        """list_all() should return all registered datasets."""
        meta1 = DatasetMetadata(dataset_id="id-1", filename="a.csv", shape=(10, 2))
        meta2 = DatasetMetadata(dataset_id="id-2", filename="b.csv", shape=(20, 3))

        registry.register(meta1)
        registry.register(meta2)

        all_meta = registry.list_all()
        assert len(all_meta) == 2

    def test_list_all_empty(self, registry: DatasetRegistry):
        """list_all() should return empty list when no datasets."""
        assert registry.list_all() == []

    def test_delete(self, registry: DatasetRegistry, sample_metadata: DatasetMetadata):
        """delete() should remove metadata and return True."""
        registry.register(sample_metadata)
        result = registry.delete("test-001")
        assert result is True
        assert registry.get("test-001") is None

    def test_delete_not_found(self, registry: DatasetRegistry):
        """delete() should return False for unknown IDs."""
        result = registry.delete("nonexistent")
        assert result is False

    def test_count(self, registry: DatasetRegistry):
        """count() should return the number of registered datasets."""
        assert registry.count() == 0
        registry.register(DatasetMetadata(dataset_id="a", filename="a.csv"))
        assert registry.count() == 1
        registry.register(DatasetMetadata(dataset_id="b", filename="b.csv"))
        assert registry.count() == 2

    def test_get_by_filename(self, registry: DatasetRegistry):
        """get_by_filename() should find datasets by filename."""
        registry.register(DatasetMetadata(dataset_id="id-1", filename="same.csv"))
        registry.register(DatasetMetadata(dataset_id="id-2", filename="same.csv"))

        results = registry.get_by_filename("same.csv")
        assert len(results) == 2

    def test_get_by_filename_none(self, registry: DatasetRegistry):
        """get_by_filename() should return empty list for no matches."""
        results = registry.get_by_filename("nonexistent.csv")
        assert results == []

    def test_update_metadata(self, registry: DatasetRegistry, sample_metadata: DatasetMetadata):
        """update_metadata() should update fields."""
        registry.register(sample_metadata)
        updated = registry.update_metadata(
            "test-001",
            {"description": "Updated description", "tags": ["new-tag"]},
        )
        assert updated is not None
        assert updated.description == "Updated description"
        assert "new-tag" in updated.tags

    def test_update_metadata_not_found(self, registry: DatasetRegistry):
        """update_metadata() should return None for unknown IDs."""
        result = registry.update_metadata("nonexistent", {"description": "test"})
        assert result is None

    def test_update_extra_metadata(self, registry: DatasetRegistry, sample_metadata: DatasetMetadata):
        """update_metadata() should merge extra_metadata."""
        registry.register(sample_metadata)
        updated = registry.update_metadata(
            "test-001",
            {"extra_metadata": {"new_key": "new_value"}},
        )
        assert updated is not None
        assert updated.extra_metadata["source"] == "manual"
        assert updated.extra_metadata["new_key"] == "new_value"

    def test_duplicate_register(self, registry: DatasetRegistry, sample_metadata: DatasetMetadata):
        """register() should raise DatabaseException for duplicate IDs."""
        registry.register(sample_metadata)
        with pytest.raises(DatabaseException):
            registry.register(sample_metadata)

    def test_close(self, registry: DatasetRegistry):
        """close() should not raise."""
        registry.close()  # Should not raise

    def test_tags_roundtrip(self, registry: DatasetRegistry):
        """Tags should survive a save/load cycle."""
        meta = DatasetMetadata(
            dataset_id="tags-test",
            filename="tags.csv",
            tags=("a", "b", "c"),
        )
        registry.register(meta)
        retrieved = registry.get("tags-test")
        assert retrieved is not None
        assert tuple(retrieved.tags) == ("a", "b", "c")

    def test_column_names_roundtrip(self, registry: DatasetRegistry):
        """Column names should survive a save/load cycle."""
        meta = DatasetMetadata(
            dataset_id="cols-test",
            filename="cols.csv",
            column_names=["x", "y", "z"],
            column_types={"x": "int64", "y": "float64", "z": "object"},
        )
        registry.register(meta)
        retrieved = registry.get("cols-test")
        assert retrieved is not None
        assert retrieved.column_names == ["x", "y", "z"]
        assert retrieved.column_types == {"x": "int64", "y": "float64", "z": "object"}
