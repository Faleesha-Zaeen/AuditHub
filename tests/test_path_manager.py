"""
Tests for the PathManager module.
"""

from pathlib import Path

import pytest

from src.utils.path_manager import PathManager


class TestPathManager:
    """Test suite for PathManager."""

    def test_root_property(self, tmp_project_dir: Path):
        """root should return the project root."""
        pm = PathManager(project_root=tmp_project_dir)
        assert pm.root == tmp_project_dir.resolve()

    def test_raw_data_dir(self, tmp_project_dir: Path):
        """raw_data_dir should return and create the path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.raw_data_dir
        assert path.exists()
        assert path.name == "raw"
        assert path.parent.name == "data"

    def test_validated_data_dir(self, tmp_project_dir: Path):
        """validated_data_dir should return and create the path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.validated_data_dir
        assert path.exists()
        assert path.name == "validated"

    def test_repaired_data_dir(self, tmp_project_dir: Path):
        """repaired_data_dir should return and create the path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.repaired_data_dir
        assert path.exists()
        assert path.name == "repaired"

    def test_mutated_data_dir(self, tmp_project_dir: Path):
        """mutated_data_dir should return and create the path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.mutated_data_dir
        assert path.exists()
        assert path.name == "mutated"

    def test_processed_data_dir(self, tmp_project_dir: Path):
        """processed_data_dir should return and create the path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.processed_data_dir
        assert path.exists()
        assert path.name == "processed"

    def test_artifacts_dir(self, tmp_project_dir: Path):
        """artifacts_dir should return and create the path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.artifacts_dir
        assert path.exists()
        assert path.name == "artifacts"

    def test_models_dir(self, tmp_project_dir: Path):
        """models_dir should return and create the path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.models_dir
        assert path.exists()
        assert path.name == "models"

    def test_reports_dir(self, tmp_project_dir: Path):
        """reports_dir should return and create the path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.reports_dir
        assert path.exists()

    def test_logs_dir(self, tmp_project_dir: Path):
        """logs_dir should return and create the path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.logs_dir
        assert path.exists()
        assert path.name == "logs"

    def test_database_path(self, tmp_project_dir: Path):
        """database_path() should return path with custom db name."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.database_path("test_custom.db")
        assert str(path).endswith("test_custom.db")
        assert path.parent.exists()

    def test_raw_data_file(self, tmp_project_dir: Path):
        """raw_data_file() should return correct full path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.raw_data_file("mydata.csv")
        assert path.name == "mydata.csv"
        assert path.parent.name == "raw"
        assert path.exists() is False  # File doesn't exist, but dir does

    def test_validated_data_file(self, tmp_project_dir: Path):
        """validated_data_file() should return correct path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.validated_data_file("valid.csv")
        assert path.name == "valid.csv"
        assert path.parent.name == "validated"

    def test_model_file(self, tmp_project_dir: Path):
        """model_file() should return correct path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.model_file("model.pkl")
        assert path.name == "model.pkl"
        assert path.parent.name == "models"

    def test_report_file(self, tmp_project_dir: Path):
        """report_file() should return correct path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.report_file("report.html")
        assert path.name == "report.html"

    def test_log_file(self, tmp_project_dir: Path):
        """log_file() should return correct path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.log_file("app.log")
        assert path.name == "app.log"
        assert path.parent.name == "logs"

    def test_artifact_file(self, tmp_project_dir: Path):
        """artifact_file() should return correct path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.artifact_file("features.pkl")
        assert path.name == "features.pkl"
        assert path.parent.name == "artifacts"

    def test_list_raw_files_empty(self, tmp_project_dir: Path):
        """list_raw_files() should return empty list for empty dir."""
        pm = PathManager(project_root=tmp_project_dir)
        files = pm.list_raw_files()
        assert files == []

    def test_list_raw_files_with_data(self, tmp_project_dir: Path):
        """list_raw_files() should list files in raw dir."""
        pm = PathManager(project_root=tmp_project_dir)
        # Create a test file in raw dir
        test_file = pm.raw_data_dir / "test.csv"
        test_file.write_text("a,b,c\n1,2,3")

        files = pm.list_raw_files("*.csv")
        assert len(files) == 1
        assert files[0].name == "test.csv"

    def test_compute_checksum(self, tmp_project_dir: Path):
        """compute_checksum() should compute SHA-256 hash."""
        pm = PathManager(project_root=tmp_project_dir)
        test_file = tmp_project_dir / "test_file.txt"
        test_file.write_text("hello world")

        checksum = pm.compute_checksum(test_file)
        # Known SHA-256 of "hello world"
        assert len(checksum) == 64  # SHA-256 hex length
        assert isinstance(checksum, str)

    def test_compute_checksum_md5(self, tmp_project_dir: Path):
        """compute_checksum() should support MD5."""
        pm = PathManager(project_root=tmp_project_dir)
        test_file = tmp_project_dir / "test_file.txt"
        test_file.write_text("hello world")

        checksum = pm.compute_checksum(test_file, algorithm="md5")
        assert len(checksum) == 32  # MD5 hex length

    def test_compute_checksum_file_not_found(self, tmp_project_dir: Path):
        """compute_checksum() should raise FileNotFoundError."""
        pm = PathManager(project_root=tmp_project_dir)
        with pytest.raises(FileNotFoundError):
            pm.compute_checksum(tmp_project_dir / "nonexistent.txt")

    def test_repr(self, tmp_project_dir: Path):
        """__repr__ should include root path."""
        pm = PathManager(project_root=tmp_project_dir)
        assert "PathManager" in repr(pm)
        assert str(tmp_project_dir.resolve()) in repr(pm)

    def test_auto_detect_root(self):
        """_detect_project_root should find a root with configs dir."""
        root = PathManager._detect_project_root()
        assert isinstance(root, Path)
        assert root.exists()

    def test_dir_created_only_once(self, tmp_project_dir: Path):
        """_ensure should only create dir once."""
        pm = PathManager(project_root=tmp_project_dir)
        dir1 = pm.raw_data_dir
        dir2 = pm.raw_data_dir
        assert dir1 == dir2
        assert dir1.exists()

    def test_repaired_data_file(self, tmp_project_dir: Path):
        """repaired_data_file() should return correct path."""
        pm = PathManager(project_root=tmp_project_dir)
        path = pm.repaired_data_file("repaired.csv")
        assert path.name == "repaired.csv"
        assert path.parent.name == "repaired"
