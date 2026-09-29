"""
AuditHub Tests - Quality Audit Module
=======================================

Tests for the DatasetAuditor class and its quality dimensions checks.
"""

import pytest
import pandas as pd
import numpy as np

from src.quality import DatasetAuditor, QualityAuditReport, AuditFinding


def test_dataset_auditor_clean(sample_dataframe):
    """Test that a clean dataset has minimal warning/error findings."""
    auditor = DatasetAuditor()
    report = auditor.audit(sample_dataframe)

    assert isinstance(report, QualityAuditReport)
    assert report.summary["ERROR"] == 0


def test_dataset_auditor_anomalies():
    """Test detection of outliers, duplicates, target leakage, constant columns."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 6],  # Contains a duplicate row (5 and 6)
        "constant": [42, 42, 42, 42, 42, 42, 42],  # Constant column
        "outliers": [1.0, 1.2, 0.9, 1.1, 100.0, 1.0, 1.0],  # Outlier (100.0)
        "leakage": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 60.0],  # Leakage candidate
        "target": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 60.0],
    })

    auditor = DatasetAuditor()
    report = auditor.audit(df, target_column="target")

    findings_dimensions = [f.dimension for f in report.findings]
    findings_severities = [f.severity for f in report.findings]

    # Check for duplicates (Uniqueness)
    assert "Uniqueness" in findings_dimensions

    # Check for constant column (Reliability / WARNING)
    assert any(f.column == "constant" and f.severity == "WARNING" for f in report.findings)

    # Check for outliers (Accuracy / INFO or WARNING)
    assert "Accuracy" in findings_dimensions

    # Check for target leakage (Reliability / ERROR)
    assert any(f.column == "leakage" and f.severity == "ERROR" for f in report.findings)
