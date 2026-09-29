"""
AuditHub Validation - Dataset Validator
========================================

The core orchestration engine for schema and content validation of any structured dataset.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.ingestion.dataset_analyzer import ColumnCategory, DatasetAnalyzer, DatasetSummary
from src.utils.logger import get_logger
from src.validation.exceptions import ValidationException
from src.validation.expectation_builder import ExpectationBuilder
from src.validation.schema_validator import SchemaValidator
from src.validation.validation_report import ValidationReport

logger = get_logger(__name__)


class DatasetValidator:
    """Orchestrates comprehensive validation checks on tabular datasets.

    Coordinates Great Expectations validation, schema structural validation,
    and custom quality checks.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the DatasetValidator.

        Parameters
        ----------
        schema : dict | None
            Expected schema configuration dictionary. If None, loads default.
        """
        self.schema_validator = SchemaValidator(schema=schema)
        self.analyzer = DatasetAnalyzer()

    def validate(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> ValidationReport:
        """Run all validation pipelines on the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        target_column : str | None
            Name of the target column. If None, inferred dynamically.
        dataset_name : str | None
            Name of the dataset for tracking.

        Returns
        -------
        ValidationReport
            Aggregated report containing score, results, and recommendations.
        """
        logger.info("Starting validation pipeline for dataset: %s", dataset_name or "Unnamed")

        if df.empty:
            raise ValidationException("Cannot validate an empty DataFrame.")

        # 1. Analyze the dataset to get metadata and inferred categories
        try:
            summary = self.analyzer.analyze(df, source_filename=dataset_name)
        except Exception as exc:
            raise ValidationException("Failed to profile dataset during validation phase", cause=exc)

        # 2. Schema structure checks
        try:
            schema_results = self.schema_validator.validate(df)
        except Exception as exc:
            raise ValidationException("Failed during schema structural check", cause=exc)

        # 3. Great Expectations validations
        try:
            builder = ExpectationBuilder(summary)
            ge_dataset = builder.build_suite(df)
            ge_results = ge_dataset.validate()
        except Exception as exc:
            raise ValidationException("Failed during Great Expectations content checks", cause=exc)

        # 4. Custom quality anomalies
        try:
            custom_results = self._run_custom_checks(df, summary, target_column)
        except Exception as exc:
            raise ValidationException("Failed during custom quality checks", cause=exc)

        # 5. Compile final ValidationReport
        report = ValidationReport.compile(
            schema_results=schema_results,
            ge_validation_result=ge_results,
            custom_results=custom_results,
            dataset_name=dataset_name,
        )

        logger.info("Validation completed. Score: %.2f (Success: %s)", report.score, report.success)
        return report

    def _run_custom_checks(
        self,
        df: pd.DataFrame,
        summary: DatasetSummary,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform custom validation checks not covered by standard GE expectations.

        Checks for mixed datatypes, empty/whitespace columns, low variance,
        and target leakage candidates.
        """
        failed_columns: Dict[str, List[str]] = {}

        # Resolve target column
        resolved_target = target_column
        if not resolved_target and summary.potential_targets:
            resolved_target = summary.potential_targets[0].get("column")
            logger.debug("Inferred target column for leakage detection: '%s'", resolved_target)

        for col_info in summary.column_stats:
            col_name = col_info.name
            issues: List[str] = []

            # Check if column is empty
            if col_info.null_pct == 100.0:
                issues.append("Empty column (100% null)")

            # Check mixed datatype
            # If the column dtype is 'object' but contains different types (e.g. ints and strings)
            if col_info.dtype == "object":
                non_null_series = df[col_name].dropna()
                if not non_null_series.empty:
                    types = non_null_series.apply(type).unique()
                    if len(types) > 1:
                        issues.append(f"Mixed datatypes detected (found types: {[t.__name__ for t in types]})")

            # Check whitespace-only values in object columns
            if col_info.dtype == "object":
                non_null_series = df[col_name].dropna()
                if not non_null_series.empty:
                    # check if any string consists only of spaces
                    try:
                        whitespace_mask = non_null_series.astype(str).str.strip() == ""
                        whitespace_count = whitespace_mask.sum()
                        if whitespace_count > 0:
                            pct = round((whitespace_count / len(df)) * 100, 2)
                            issues.append(f"Whitespace-only strings detected in {whitespace_count} rows ({pct}%)")
                    except Exception:
                        pass

            # Check low variance (except constant)
            if col_info.is_numeric and col_info.std is not None:
                if 0.0 < col_info.std < 1e-4:
                    issues.append(f"Low variance detected (std = {col_info.std})")

            # Check leakage candidates
            # If a column is extremely correlated with the target variable (> 0.95 absolute correlation)
            if resolved_target and col_name != resolved_target and col_info.is_numeric:
                try:
                    target_series = df[resolved_target]
                    if pd.api.types.is_numeric_dtype(target_series.dtype):
                        corr = df[col_name].corr(target_series)
                        if not pd.isna(corr) and abs(corr) > 0.95:
                            issues.append(f"Potential target leakage: high correlation with target ({corr:.4f})")
                except Exception:
                    pass

            if issues:
                failed_columns[col_name] = issues

        return {
            "failed_columns": failed_columns,
            "resolved_target": resolved_target,
        }
