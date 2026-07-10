"""
Tests for the FileManager module.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.utils.file_manager import FileManager
from src.utils.exceptions import DatasetException


class TestFileManager:
    """Test suite for FileManager."""

    def test_init_with_base_dir(self, tmp_project_dir: Path):
        """FileManager should accept a base directory."""
        fm = FileManager(base_dir=tmp_project_dir)
        assert fm._base_dir == tmp_project_dir.resolve()

    def test_init_without_base_dir(self):
        """FileManager should work without a base directory."""
        fm = FileManager()
        assert fm._base_dir is None

    # ------------------------------------------------------------------
    # DataFrame save/load
    # ------------------------------------------------------------------

    def test_save_dataframe_csv(self, tmp_project_dir: Path, sample_dataframe: pd.DataFrame):
        """save_dataframe() should save CSV files."""
        fm = FileManager()
        path = tmp_project_dir / "test.csv"
        result = fm.save_dataframe(sample_dataframe, path)
        assert result.exists()
        assert result.suffix == ".csv"
        loaded = pd.read_csv(path)
        assert len(loaded) == 3

    def test_save_dataframe_tsv(self, tmp_project_dir: Path, sample_dataframe: pd.DataFrame):
        """save_dataframe() should save TSV files."""
        fm = FileManager()
        path = tmp_project_dir / "test.tsv"
        fm.save_dataframe(sample_dataframe, path)
        loaded = pd.read_csv(path, sep="\t")
        assert len(loaded) == 3

    def test_save_dataframe_json(self, tmp_project_dir: Path, sample_dataframe: pd.DataFrame):
        """save_dataframe() should save JSON files."""
        fm = FileManager()
        path = tmp_project_dir / "test.json"
        fm.save_dataframe(sample_dataframe, path)
        loaded = pd.read_json(path)
        assert len(loaded) == 3

    def test_save_dataframe_parquet(self, tmp_project_dir: Path, sample_dataframe: pd.DataFrame):
        """save_dataframe() should save Parquet files."""
        fm = FileManager()
        path = tmp_project_dir / "test.parquet"
        fm.save_dataframe(sample_dataframe, path)
        loaded = pd.read_parquet(path)
        assert len(loaded) == 3

    def test_save_dataframe_pickle(self, tmp_project_dir: Path, sample_dataframe: pd.DataFrame):
        """save_dataframe() should save pickle files."""
        fm = FileManager()
        path = tmp_project_dir / "test.pkl"
        fm.save_dataframe(sample_dataframe, path)
        loaded = pd.read_pickle(path)
        assert len(loaded) == 3

    def test_save_dataframe_excel(self, tmp_project_dir: Path, sample_dataframe: pd.DataFrame):
        """save_dataframe() should save Excel files."""
        fm = FileManager()
        path = tmp_project_dir / "test.xlsx"
        fm.save_dataframe(sample_dataframe, path)
        loaded = pd.read_excel(path)
        assert len(loaded) == 3

    def test_save_dataframe_unsupported_format(self, tmp_project_dir: Path, sample_dataframe: pd.DataFrame):
        """save_dataframe() should raise for unsupported formats."""
        fm = FileManager()
        path = tmp_project_dir / "test.txt"
        with pytest.raises(DatasetException):
            fm.save_dataframe(sample_dataframe, path)

    def test_load_dataframe_csv(self, sample_dataframe_path: Path):
        """load_dataframe() should load CSV files."""
        fm = FileManager()
        df = fm.load_dataframe(sample_dataframe_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_load_dataframe_file_not_found(self, tmp_project_dir: Path):
        """load_dataframe() should raise for missing files."""
        fm = FileManager()
        with pytest.raises(DatasetException):
            fm.load_dataframe(tmp_project_dir / "nonexistent.csv")

    def test_load_dataframe_unsupported_format(self, tmp_project_dir: Path):
        """load_dataframe() should raise for unsupported formats."""
        fm = FileManager()
        path = tmp_project_dir / "test.xyz"
        path.write_text("dummy")
        with pytest.raises(DatasetException):
            fm.load_dataframe(path)

    # ------------------------------------------------------------------
    # JSON save/load
    # ------------------------------------------------------------------

    def test_save_json(self, tmp_project_dir: Path):
        """save_json() should save JSON files."""
        fm = FileManager()
        data = {"key": "value", "number": 42}
        path = tmp_project_dir / "data.json"
        result = fm.save_json(data, path)
        assert result.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_save_json_creates_dirs(self, tmp_project_dir: Path):
        """save_json() should create parent directories."""
        fm = FileManager()
        path = tmp_project_dir / "nested" / "sub" / "data.json"
        fm.save_json({"a": 1}, path)
        assert path.exists()

    def test_load_json(self, tmp_project_dir: Path):
        """load_json() should load JSON files."""
        fm = FileManager()
        path = tmp_project_dir / "data.json"
        path.write_text(json.dumps({"hello": "world"}))
        data = fm.load_json(path)
        assert data == {"hello": "world"}

    def test_load_json_file_not_found(self, tmp_project_dir: Path):
        """load_json() should raise for missing files."""
        fm = FileManager()
        with pytest.raises(FileNotFoundError):
            fm.load_json(tmp_project_dir / "missing.json")

    # ------------------------------------------------------------------
    # Pickle save/load
    # ------------------------------------------------------------------

    def test_save_pickle(self, tmp_project_dir: Path):
        """save_pickle() should save pickle files."""
        fm = FileManager()
        obj = {"data": [1, 2, 3], "name": "test"}
        path = tmp_project_dir / "obj.pkl"
        result = fm.save_pickle(obj, path)
        assert result.exists()
        loaded = pickle.loads(path.read_bytes())
        assert loaded == obj

    def test_load_pickle(self, tmp_project_dir: Path):
        """load_pickle() should load pickle files."""
        fm = FileManager()
        obj = {"hello": "world"}
        path = tmp_project_dir / "obj.pkl"
        path.write_bytes(pickle.dumps(obj))
        loaded = fm.load_pickle(path)
        assert loaded == obj

    def test_load_pickle_file_not_found(self, tmp_project_dir: Path):
        """load_pickle() should raise for missing files."""
        fm = FileManager()
        with pytest.raises(FileNotFoundError):
            fm.load_pickle(tmp_project_dir / "missing.pkl")

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def test_file_exists(self, tmp_project_dir: Path):
        """file_exists() should check file existence."""
        fm = FileManager()
        path = tmp_project_dir / "exists.txt"
        assert fm.file_exists(path) is False
        path.write_text("hello")
        assert fm.file_exists(path) is True

    def test_safe_delete(self, tmp_project_dir: Path):
        """safe_delete() should delete existing files."""
        fm = FileManager()
        path = tmp_project_dir / "delete_me.txt"
        path.write_text("to be deleted")
        assert fm.safe_delete(path) is True
        assert path.exists() is False

    def test_safe_delete_missing(self, tmp_project_dir: Path):
        """safe_delete() should return False for missing files."""
        fm = FileManager()
        assert fm.safe_delete(tmp_project_dir / "missing.txt") is False

    def test_create_backup(self, tmp_project_dir: Path):
        """create_backup() should create a timestamped backup."""
        fm = FileManager()
        path = tmp_project_dir / "original.txt"
        path.write_text("backup me")
        backup = fm.create_backup(path)
        assert backup is not None
        assert backup.exists()
        assert backup.name.startswith("original_")
        assert backup.suffix == ".txt"

    def test_create_backup_missing(self, tmp_project_dir: Path):
        """create_backup() should return None for missing files."""
        fm = FileManager()
        assert fm.create_backup(tmp_project_dir / "missing.txt") is None

    def test_get_size(self, tmp_project_dir: Path):
        """get_size() should return file size in bytes."""
        fm = FileManager()
        path = tmp_project_dir / "sized.txt"
        content = "x" * 1024
        path.write_text(content)
        assert fm.get_size(path) == 1024

    def test_get_size_missing(self, tmp_project_dir: Path):
        """get_size() should return 0 for missing files."""
        fm = FileManager()
        assert fm.get_size(tmp_project_dir / "missing.txt") == 0

    def test_save_dataframe_auto_creates_dir(self, tmp_project_dir: Path, sample_dataframe: pd.DataFrame):
        """save_dataframe() should auto-create parent directories."""
        fm = FileManager()
        path = tmp_project_dir / "deep" / "nested" / "data.csv"
        fm.save_dataframe(sample_dataframe, path)
        assert path.exists()

    def test_repr(self):
        """__repr__ should include base_dir."""
        fm = FileManager()
        assert "FileManager" in repr(fm)
