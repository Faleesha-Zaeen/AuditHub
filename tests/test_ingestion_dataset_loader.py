"""
Tests for the DatasetLoader module.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.exceptions import (
    CorruptedDatasetError,
    EmptyDatasetError,
    InvalidJsonError,
    UnsupportedFormatError,
)


class TestDatasetLoader:
    """Test suite for DatasetLoader."""

    @pytest.fixture
    def loader(self) -> DatasetLoader:
        return DatasetLoader()

    @pytest.fixture
    def csv_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(path, index=False)
        return path

    @pytest.fixture
    def excel_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "test.xlsx"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(path, index=False)
        return path

    @pytest.fixture
    def json_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "test.json"
        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        path.write_text(json.dumps(data))
        return path

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def test_load_csv(self, loader: DatasetLoader, csv_path: Path):
        """Should load a CSV file."""
        df, meta = loader.load(csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert meta.filename == "test.csv"
        assert meta.shape == (3, 2)
        assert meta.source_format == ".csv"

    def test_load_csv_with_headers(self, loader: DatasetLoader, tmp_path: Path):
        """Should load CSV with correct column names."""
        path = tmp_path / "headers.csv"
        pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(path, index=False)
        df, meta = loader.load(path)
        assert list(df.columns) == ["col_a", "col_b"]
        assert meta.column_names == ["col_a", "col_b"]

    def test_load_csv_encoding_fallback(self, loader: DatasetLoader, tmp_path: Path):
        """Should handle Latin-1 encoded CSV."""
        path = tmp_path / "latin1.csv"
        with open(path, "w", encoding="latin1") as f:
            f.write("a,b\n1,2\n3,4")
        df, _ = loader.load(path)
        assert len(df) == 2

    # ------------------------------------------------------------------
    # Excel loading
    # ------------------------------------------------------------------

    def test_load_excel(self, loader: DatasetLoader, excel_path: Path):
        """Should load an Excel file."""
        df, meta = loader.load(excel_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert meta.source_format == ".xlsx"

    def test_load_excel_xls(self, loader: DatasetLoader, tmp_path: Path):
        """Should load .xls files."""
        path = tmp_path / "old.xls"
        pd.DataFrame({"x": [1]}).to_excel(path, index=False)
        df, _ = loader.load(path)
        assert len(df) == 1

    # ------------------------------------------------------------------
    # JSON loading
    # ------------------------------------------------------------------

    def test_load_json_array(self, loader: DatasetLoader, json_path: Path):
        """Should load JSON array of objects."""
        df, meta = loader.load(json_path)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]
        assert meta.source_format == ".json"

    def test_load_json_empty_array(self, loader: DatasetLoader, tmp_path: Path):
        """Should raise EmptyDatasetError for empty JSON array (no rows)."""
        path = tmp_path / "empty_array.json"
        path.write_text("[]")
        with pytest.raises(EmptyDatasetError):
            loader.load(path)

    def test_load_json_records_key(self, loader: DatasetLoader, tmp_path: Path):
        """Should load JSON with a records key."""
        path = tmp_path / "records.json"
        path.write_text(json.dumps({"data": [{"x": 1}, {"x": 2}]}))
        df, _ = loader.load(path)
        assert len(df) == 2

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_file_not_found(self, loader: DatasetLoader, tmp_path: Path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.csv")

    def test_unsupported_format(self, loader: DatasetLoader, tmp_path: Path):
        """Should raise UnsupportedFormatError for bad extension."""
        path = tmp_path / "data.txt"
        path.write_text("a,b\n1,2")
        with pytest.raises(UnsupportedFormatError):
            loader.load(path)

    def test_corrupted_csv(self, loader: DatasetLoader, tmp_path: Path):
        """Should raise CorruptedDatasetError for CSV that pandas cannot parse."""
        path = tmp_path / "corrupt.csv"
        # Unterminated quote causes ParserError from the C parser
        path.write_text('a,b\n"unterminated,2\n3,4\n')
        with pytest.raises(CorruptedDatasetError):
            loader.load(path)

    def test_empty_csv_file(self, loader: DatasetLoader, tmp_path: Path):
        """Should raise EmptyDatasetError for empty file."""
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(EmptyDatasetError):
            loader.load(path)

    def test_header_only_csv(self, loader: DatasetLoader, tmp_path: Path):
        """Should raise EmptyDatasetError for header-only CSV."""
        path = tmp_path / "header_only.csv"
        path.write_text("a,b,c\n")
        with pytest.raises(EmptyDatasetError):
            loader.load(path)

    def test_invalid_json(self, loader: DatasetLoader, tmp_path: Path):
        """Should raise InvalidJsonError for bad JSON."""
        path = tmp_path / "bad.json"
        path.write_text("{invalid json}")
        with pytest.raises(InvalidJsonError):
            loader.load(path)

    def test_non_tabular_json(self, loader: DatasetLoader, tmp_path: Path):
        """Should raise InvalidJsonError for non-tabular JSON."""
        path = tmp_path / "scalar.json"
        path.write_text(json.dumps("just a string"))
        with pytest.raises(InvalidJsonError):
            loader.load(path)

    def test_nested_json_not_records(self, loader: DatasetLoader, tmp_path: Path):
        """Should raise InvalidJsonError for nested dict without records key."""
        path = tmp_path / "nested.json"
        path.write_text(json.dumps({"metadata": {"version": 1}, "not_records": "x"}))
        with pytest.raises(InvalidJsonError):
            loader.load(path)

    # ------------------------------------------------------------------
    # Metadata checks
    # ------------------------------------------------------------------

    def test_metadata_checksum(self, loader: DatasetLoader, csv_path: Path):
        """Metadata should include a valid checksum."""
        _, meta = loader.load(csv_path)
        assert len(meta.checksum) == 64  # SHA-256 hex
        assert isinstance(meta.checksum, str)

    def test_metadata_column_types(self, loader: DatasetLoader, csv_path: Path):
        """Metadata should include column types."""
        _, meta = loader.load(csv_path)
        assert "a" in meta.column_types
        assert "b" in meta.column_types

    def test_metadata_file_size(self, loader: DatasetLoader, csv_path: Path):
        """Metadata should include file size."""
        _, meta = loader.load(csv_path)
        assert meta.size_bytes > 0

    def test_supported_extensions(self, loader: DatasetLoader):
        """supported_extensions() should return a frozenset."""
        exts = loader.supported_extensions()
        assert ".csv" in exts
        assert ".xlsx" in exts
        assert ".json" in exts

    def test_no_columns(self, loader: DatasetLoader, tmp_path: Path):
        """Should raise EmptyDatasetError for DataFrame with no columns."""
        path = tmp_path / "no_cols.csv"
        pd.DataFrame().to_csv(path, index=False)
        with pytest.raises(EmptyDatasetError):
            loader.load(path)
