"""
AuditHub Tests - Reporting Module
===================================

Tests for the ReportGenerator class, HTML rendering, and JSON exports.
"""

import pytest
import json
from pathlib import Path

from src.reporting import ReportGenerator


def test_report_generator_html_and_json(tmp_path):
    """Test generating HTML and JSON reports."""
    generator = ReportGenerator()

    # Mock inputs
    validation_report = {"success": True, "score": 95.0}
    health_report = {
        "overall_score": 90.0,
        "grade": "A",
        "dimensions": {"completeness": {"score": 100.0}, "uniqueness": {"score": 90.0}},
        "explanations": ["-10 points from Uniqueness due to duplicates"]
    }
    audit_report = {
        "findings": [
            {"dimension": "Uniqueness", "severity": "WARNING", "column": "id", "message": "Duplicate entries found"}
        ],
        "summary": {"INFO": 0, "WARNING": 1, "ERROR": 0}
    }
    repair_log = [
        {"step": 1, "action": "remove_duplicates", "details": {"rows_removed": 2}, "timestamp": "2026-07-17"}
    ]

    # Test HTML report generation
    html_content = generator.generate_html(
        dataset_name="test_dataset",
        validation_report=validation_report,
        health_report=health_report,
        audit_report=audit_report,
        repair_log=repair_log
    )

    assert "test_dataset" in html_content
    assert "Overall Health" in html_content
    assert "remove_duplicates" in html_content
    assert "Duplicate entries found" in html_content

    # Test JSON Export
    json_path = generator.export_report(
        dataset_name="test_dataset",
        validation_report=validation_report,
        health_report=health_report,
        audit_report=audit_report,
        repair_log=repair_log,
        format_type="json"
    )

    assert json_path.exists()
    with open(json_path, "r") as f:
        data = json.load(f)
        assert data["dataset_name"] == "test_dataset"
        assert data["health"]["grade"] == "A"
        assert len(data["repairs"]) == 1
