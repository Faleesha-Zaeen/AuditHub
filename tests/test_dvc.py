"""
AuditHub Tests - DVC Module
=============================

Tests for the DVCManager and fallback simulated dataset versioning.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.utils.dvc_manager import DVCManager


def test_dvc_manager_simulated(tmp_path):
    """Test DVCManager version tracking when DVC falls back to simulation."""
    # Write a test file
    test_file = tmp_path / "dataset.csv"
    test_file.write_text("id,value\n1,10.0\n2,20.0")

    # Force simulated mode by using an empty folder as root without DVC
    manager = DVCManager(root_dir=tmp_path)
    metadata = manager.track_file(test_file)

    assert isinstance(metadata, dict)
    assert metadata["status"] == "success"
    assert "checksum" in metadata
    assert metadata["file"] == "dataset.csv"
    assert metadata["size_bytes"] > 0
    assert metadata["engine"] in ("dvc", "simulated")
