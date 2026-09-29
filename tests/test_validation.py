"""
AuditHub Tests - Validation Module
=====================================

Tests for the schema validator, expectation builder, dataset validator, and report compiler.
"""

import pytest
import pandas as pd
import numpy as np

from src.validation import (
    DatasetValidator,
    SchemaValidator,
    ExpectationBuilder,
    ValidationReport,
    SchemaValidationError,
)
from src.utils.exceptions import ValidationException


def test_schema_validator_success(sample_dataframe):
    """Test that SchemaValidator passes for valid schema."""
    validator = SchemaValidator.from_reference_dataframe(sample_dataframe)
    results = validator.validate(sample_dataframe)
    assert results["success"] is True
    assert len(results["errors"]) == 0
    assert len(results["warnings"]) == 0


def test_schema_validator_mismatch():
    """Test that SchemaValidator catches missing columns and type mismatches."""
    schema = {
        "dataset": {
            "expected_columns": [
                {"name": "col_int", "type": "integer", "nullable": False, "unique": True},
                {"name": "col_float", "type": "float", "nullable": True, "unique": False},
            ],
            "constraints": {
                "min_rows": 2,
            },
            "column_types": {
                "integer": ["int64", "int32"],
                "float": ["float64", "float32"],
            }
        }
    }
    validator = SchemaValidator(schema=schema)

    # DataFrame with missing col_float and wrong type in col_int (string instead of int)
    df_invalid = pd.DataFrame({
        "col_int": ["not_int", "also_not_int"],
    })

    results = validator.validate(df_invalid)
    assert results["success"] is False
    assert any("Missing expected column" in err for err in results["errors"])
    assert any("Type mismatch" in err or "Column 'col_int'" in err for err in results["errors"])


def test_expectation_builder(sample_dataframe):
    """Test that ExpectationBuilder constructs an expectation suite successfully."""
    from src.ingestion.dataset_analyzer import DatasetAnalyzer
    analyzer = DatasetAnalyzer()
    summary = analyzer.analyze(sample_dataframe)

    builder = ExpectationBuilder(summary)
    ge_dataset = builder.build_suite(sample_dataframe)

    # Perform validation
    validation_results = ge_dataset.validate()
    assert validation_results["success"] is True


def test_dataset_validator_pipeline(sample_dataframe, config_manager):
    """Test the full DatasetValidator pipeline."""
    validator = DatasetValidator()
    report = validator.validate(sample_dataframe, dataset_name="test_dataset")

    assert isinstance(report, ValidationReport)
    assert report.score >= 90.0
    assert isinstance(report.to_dict(), dict)


def test_dataset_validator_anomalies(config_manager):
    """Test custom anomaly detection: whitespace, leakage, mixed types."""
    # Dataframe with:
    # 1. Whitespace strings in 'name'
    # 2. Mixed types in 'mixed' (numeric and text)
    # 3. Low variance in 'low_var'
    # 4. Target leakage in 'leakage' (identical to target)
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", " ", "Bob", "   ", "Charlie"],
        "mixed": [1, "two", 3, "four", 5],
        "low_var": [1.0, 1.00001, 1.0, 1.0, 1.0],
        "leakage": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    validator = DatasetValidator()
    report = validator.validate(df, target_column="target", dataset_name="anomalous")

    assert isinstance(report, ValidationReport)
    # Deductions should occur due to custom anomalies
    assert report.score < 100.0
    
    recommendations_str = " ".join(report.recommendations)
    assert "leakage" in recommendations_str.lower()
    assert "whitespace-only" in recommendations_str.lower()
    assert "mixed datatypes" in recommendations_str.lower()


def test_dataset_validator_empty(config_manager):
    """Test that DatasetValidator raises validation error on empty DataFrame."""
    validator = DatasetValidator()
    df_empty = pd.DataFrame()
    with pytest.raises(ValidationException):
        validator.validate(df_empty)
