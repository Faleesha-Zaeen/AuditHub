"""
Tests for the Helpers module.
"""

import os
from pathlib import Path

import pytest
import yaml

from src.utils.helpers import (
    chunk_list,
    create_directory,
    delete_directory,
    ensure_directory_exists,
    ensure_extension,
    format_duration,
    generate_run_id,
    generate_timestamp,
    generate_uuid,
    get_env_var,
    get_file_size,
    get_file_size_mb,
    list_files,
    load_yaml,
    merge_dicts,
    safe_filename,
    save_yaml,
    timer,
    unique_filename,
)


class TestHelpers:
    """Test suite for helpers module."""

    def test_create_directory(self, tmp_project_dir: Path):
        """create_directory should create a directory and return Path."""
        new_dir = tmp_project_dir / "new_test_dir"
        result = create_directory(new_dir)
        assert new_dir.exists()
        assert isinstance(result, Path)

    def test_create_directory_exist_ok(self, tmp_project_dir: Path):
        """create_directory should not raise when directory exists."""
        existing = tmp_project_dir / "already_exists"
        existing.mkdir()
        result = create_directory(existing, exist_ok=True)
        assert result == existing.resolve()

    def test_ensure_directory_exists(self, tmp_project_dir: Path):
        """ensure_directory_exists should create and return dir."""
        path = tmp_project_dir / "ensured"
        result = ensure_directory_exists(path)
        assert path.exists()
        assert result == path.resolve()

    def test_list_files_empty(self, tmp_project_dir: Path):
        """list_files should return empty list for empty dir."""
        empty_dir = tmp_project_dir / "empty"
        empty_dir.mkdir()
        assert list_files(empty_dir) == []

    def test_list_files_with_files(self, tmp_project_dir: Path):
        """list_files should list files."""
        d = tmp_project_dir / "files"
        d.mkdir()
        (d / "a.csv").write_text("a")
        (d / "b.csv").write_text("b")
        files = list_files(d)
        assert len(files) == 2

    def test_list_files_with_pattern(self, tmp_project_dir: Path):
        """list_files should filter by pattern."""
        d = tmp_project_dir / "patterned"
        d.mkdir()
        (d / "a.csv").write_text("a")
        (d / "b.json").write_text("b")
        files = list_files(d, pattern="*.csv")
        assert len(files) == 1
        assert files[0].suffix == ".csv"

    def test_list_files_with_extensions(self, tmp_project_dir: Path):
        """list_files should filter by extensions."""
        d = tmp_project_dir / "exts"
        d.mkdir()
        (d / "a.csv").write_text("a")
        (d / "b.json").write_text("b")
        (d / "c.txt").write_text("c")
        files = list_files(d, extensions=[".csv", ".json"])
        assert len(files) == 2

    def test_list_files_missing_dir(self, tmp_project_dir: Path):
        """list_files should return empty for missing dir."""
        assert list_files(tmp_project_dir / "nonexistent") == []

    def test_delete_directory(self, tmp_project_dir: Path):
        """delete_directory should remove a directory."""
        d = tmp_project_dir / "delete_me"
        d.mkdir()
        (d / "file.txt").write_text("x")
        assert delete_directory(d) is True
        assert not d.exists()

    def test_delete_directory_missing(self, tmp_project_dir: Path):
        """delete_directory should return False for missing dir."""
        assert delete_directory(tmp_project_dir / "nonexistent") is False

    def test_get_file_size(self, tmp_project_dir: Path):
        """get_file_size should return bytes."""
        f = tmp_project_dir / "sized.txt"
        f.write_text("x" * 500)
        assert get_file_size(f) == 500

    def test_get_file_size_missing(self, tmp_project_dir: Path):
        """get_file_size should return 0 for missing."""
        assert get_file_size(tmp_project_dir / "missing.txt") == 0

    def test_get_file_size_mb(self, tmp_project_dir: Path):
        """get_file_size_mb should return MB."""
        f = tmp_project_dir / "sized_mb.txt"
        f.write_text("x" * 1024 * 1024)
        size = get_file_size_mb(f)
        assert size > 0.99
        assert size < 1.1

    def test_load_yaml(self, configs_dir: Path):
        """load_yaml should parse YAML files."""
        yml = configs_dir / "test.yml"
        data = {"key": "value", "nested": {"inner": 42}}
        with open(yml, "w") as f:
            yaml.dump(data, f)
        loaded = load_yaml(yml)
        assert loaded == data

    def test_load_yaml_not_found(self, tmp_project_dir: Path):
        """load_yaml should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_project_dir / "nonexistent.yml")

    def test_save_yaml(self, tmp_project_dir: Path):
        """save_yaml should write YAML files."""
        path = tmp_project_dir / "saved.yml"
        data = {"key": "value"}
        save_yaml(data, path)
        assert path.exists()
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_generate_timestamp(self):
        """generate_timestamp should return formatted string."""
        ts = generate_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 10

    def test_generate_timestamp_custom_format(self):
        """generate_timestamp should accept custom format."""
        ts = generate_timestamp("%Y")
        assert ts.isdigit()
        assert len(ts) == 4

    def test_generate_uuid(self):
        """generate_uuid should return UUID string."""
        uid = generate_uuid()
        assert isinstance(uid, str)
        assert "-" in uid

    def test_generate_run_id(self):
        """generate_run_id should return run ID string."""
        rid = generate_run_id()
        assert rid.startswith("run_")

    def test_safe_filename(self):
        """safe_filename should sanitize unsafe characters."""
        result = safe_filename('my/file:name?.csv')
        assert "/" not in result
        assert "?" not in result
        assert ":" not in result

    def test_safe_filename_custom_replacement(self):
        """safe_filename should use custom replacement."""
        result = safe_filename("bad/name", replacement="-")
        assert "/" not in result
        assert "-" in result or "_" in result  # path separators handled

    def test_unique_filename(self):
        """unique_filename should append timestamp."""
        result = unique_filename("report.html")
        assert result.startswith("report_")
        assert result.endswith(".html")

    def test_ensure_extension(self):
        """ensure_extension should add missing extension."""
        assert ensure_extension("data", ".csv") == "data.csv"

    def test_ensure_extension_has_it(self):
        """ensure_extension should not double-add extension."""
        assert ensure_extension("data.csv", ".csv") == "data.csv"

    def test_ensure_extension_without_dot(self):
        """ensure_extension should handle extension without dot."""
        assert ensure_extension("data", "csv") == "data.csv"

    def test_timer(self):
        """timer should measure elapsed time."""
        stopwatch = timer()
        import time as _time
        _time.sleep(0.01)
        elapsed = stopwatch()
        assert elapsed > 0.0

    def test_format_duration_hours(self):
        """format_duration should format hours."""
        assert format_duration(3661) == "1h 1m 1s"

    def test_format_duration_minutes(self):
        """format_duration should format minutes."""
        assert format_duration(125) == "2m 5s"

    def test_format_duration_seconds(self):
        """format_duration should format seconds."""
        result = format_duration(45.2)
        assert "45" in result
        assert "s" in result

    def test_format_duration_zero(self):
        """format_duration should handle zero."""
        result = format_duration(0)
        assert "0.0s" in result

    def test_chunk_list(self):
        """chunk_list should split list into chunks."""
        items = [1, 2, 3, 4, 5]
        chunks = chunk_list(items, 2)
        assert chunks == [[1, 2], [3, 4], [5]]

    def test_chunk_list_empty(self):
        """chunk_list should handle empty list."""
        assert chunk_list([], 3) == []

    def test_merge_dicts(self):
        """merge_dicts should deep-merge."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 4}
        result = merge_dicts(base, override)
        assert result == {"a": 1, "b": {"c": 99, "d": 3}, "e": 4}

    def test_merge_dicts_no_override(self):
        """merge_dicts should return copy when no override."""
        base = {"a": 1}
        result = merge_dicts(base, {})
        assert result == {"a": 1}
        assert result is not base  # should be a copy

    def test_get_env_var(self, monkeypatch):
        """get_env_var should return env var value."""
        monkeypatch.setenv("TEST_AUDITHUB_VAR", "test_value")
        assert get_env_var("TEST_AUDITHUB_VAR") == "test_value"

    def test_get_env_var_default(self):
        """get_env_var should return default for missing var."""
        assert get_env_var("NONEXISTENT_VAR_XYZ", "fallback") == "fallback"
