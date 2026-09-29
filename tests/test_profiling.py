"""
AuditHub Tests - Profiling Module
===================================

Tests for the DatasetProfiler class and profiling summaries.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.profiling import DatasetProfiler


def test_dataset_profiler(sample_dataframe, tmp_path, config_manager):
    """Test that DatasetProfiler successfully generates a profile report."""
    profiler = DatasetProfiler()
    # Force minimal mode for speed in tests
    profiler.minimal = True

    summary = profiler.profile(
        sample_dataframe,
        dataset_name="test_profile",
        output_dir=tmp_path
    )

    assert isinstance(summary, dict)
    assert summary["dataset_name"] == "test_profile"
    assert "report_path" in summary
    assert Path(summary["report_path"]).exists()
    assert summary["n_records"] == 3
    assert summary["n_features"] == 3
    assert "variables" in summary
    assert "id" in summary["variables"]
