"""
AuditHub Tests - Health scoring Module
========================================

Tests for the HealthScoreCalculator class and its composite scores and deductions.
"""

import pytest
import pandas as pd

from src.health import HealthScoreCalculator, DatasetHealthReport


def test_health_calculator_clean(sample_dataframe):
    """Test that a clean dataset gets a high health score and an A or B grade."""
    calculator = HealthScoreCalculator()
    report = calculator.calculate(sample_dataframe)

    assert isinstance(report, DatasetHealthReport)
    assert report.overall_score >= 80.0
    assert report.grade in ("A", "B")
    assert isinstance(report.to_dict(), dict)


def test_health_calculator_damaged():
    """Test score deductions and grade drop on damaged dataset."""
    df_damaged = pd.DataFrame({
        "id": [1, 2, 3, 4, 4, 4, 4],  # Contains actual duplicate row (3 and 6)
        "missing_col": [None, None, None, None, None, 1.0, None],  # 85% nulls
        "constant_col": [9, 9, 9, 9, 9, 9, 9],  # Constant
        "target": [1, 0, 1, 0, 1, 0, 0],
    })

    calculator = HealthScoreCalculator()
    report = calculator.calculate(df_damaged, target_column="target")

    # Score should drop significantly
    assert report.overall_score < 90.0
    assert len(report.explanations) > 0
    # Deductions should mention duplicates and missingness
    explanations_str = " ".join(report.explanations)
    assert "duplicates" in explanations_str.lower()
    assert "missing" in explanations_str.lower()
    assert "constant" in explanations_str.lower()
