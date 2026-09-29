"""
AuditHub Validation - Validation Report
=========================================

Data structures and logic for aggregating validation outcomes, scoring, and recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Consolidated report detailing all validation checks, scores, and recommendations.

    Attributes
    ----------
    success : bool
        Overall validation success.
    score : float
        Composite score (0.0 to 100.0) indicating dataset quality.
    total_checks : int
        Total number of validations performed.
    passed_checks : int
        Number of passed validations.
    failed_checks : int
        Number of failed validations.
    schema_results : dict
        Results of schema checks (missing columns, type consistency).
    ge_results : list
        List of expectation results from Great Expectations.
    custom_results : dict
        Results of custom validations (leakage, whitespace-only, constant columns).
    recommendations : list[str]
        Actionable recommendations for repairing the dataset.
    dataset_name : str | None
        Name of the validated dataset.
    timestamp : str
        ISO format timestamp of validation.
    """

    success: bool = False
    score: float = 0.0
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    schema_results: Dict[str, Any] = field(default_factory=dict)
    ge_results: List[Dict[str, Any]] = field(default_factory=list)
    custom_results: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    dataset_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the report.
        """
        return {
            "success": self.success,
            "score": round(self.score, 2),
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "schema_results": self.schema_results,
            "ge_results": self.ge_results,
            "custom_results": self.custom_results,
            "recommendations": self.recommendations,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
        }

    @classmethod
    def compile(
        cls,
        schema_results: Dict[str, Any],
        ge_validation_result: Dict[str, Any],
        custom_results: Dict[str, Any],
        dataset_name: Optional[str] = None,
    ) -> "ValidationReport":
        """Compile reports from schema, GE, and custom checks into a ValidationReport.

        Parameters
        ----------
        schema_results : dict
            Dict returned by SchemaValidator.validate.
        ge_validation_result : dict
            Dict returned by Great Expectations validation run.
        custom_results : dict
            Dict from custom validation checks.
        dataset_name : str | None
            Name of the validated dataset.

        Returns
        -------
        ValidationReport
            Aggregated validation report.
        """
        ge_results_list = []
        ge_passed = 0
        ge_failed = 0

        # Parse Great Expectations results
        results = ge_validation_result.get("results", [])
        for r in results:
            success = r.get("success", False)
            expectation_config = r.get("expectation_config", {})
            kwargs = expectation_config.get("kwargs", {})
            column = kwargs.get("column")
            expectation_type = expectation_config.get("expectation_type")

            if success:
                ge_passed += 1
            else:
                ge_failed += 1

            ge_results_list.append({
                "expectation": expectation_type,
                "column": column,
                "success": success,
                "kwargs": kwargs,
                "exception_info": r.get("exception_info"),
            })

        # Schema checks counts
        schema_success = schema_results.get("success", False)
        schema_errors = schema_results.get("errors", [])
        schema_warnings = schema_results.get("warnings", [])

        # Custom checks counts
        custom_failed_cols = custom_results.get("failed_columns", {})
        custom_failed_count = sum(len(issues) for issues in custom_failed_cols.values())

        # Total counts
        total_checks = len(ge_results_list) + len(schema_errors) + 1  # 1 for schema structure
        failed_checks = ge_failed + len(schema_errors) + (1 if not schema_success else 0)
        passed_checks = total_checks - failed_checks

        # Calculate Score (0-100)
        if total_checks > 0:
            score = (passed_checks / total_checks) * 100.0
        else:
            score = 100.0

        # Adjust score for custom anomalies
        deduction = custom_failed_count * 5.0
        score = max(0.0, score - deduction)

        # Generate Actionable Recommendations
        recommendations = []

        # Schema recommendations
        if not schema_success:
            recommendations.append("Fix dataset schema violations: verify column existence and ensure correct datatypes.")

        for err in schema_errors:
            if "Type mismatch" in err:
                recommendations.append(f"Cast column datatype: {err}")
            elif "Nullability violation" in err:
                recommendations.append(f"Handle unexpected nulls: {err}")
            elif "Uniqueness violation" in err:
                recommendations.append(f"Remove duplicates or use a unique identifier: {err}")

        # GE recommendations
        for res in ge_results_list:
            if not res["success"]:
                col = res["column"]
                exp = res["expectation"]
                if exp == "expect_column_values_to_not_be_null":
                    recommendations.append(f"Column '{col}': Impute missing values (mean/median/mode) or drop null rows.")
                elif exp == "expect_column_values_to_be_unique":
                    recommendations.append(f"Column '{col}': Ensure uniqueness of identifier column by removing duplicate entries.")
                elif exp == "expect_column_unique_value_count_to_be_between":
                    recommendations.append(f"Column '{col}': Remove constant columns or verify variance.")
                elif exp == "expect_column_values_to_not_match_regex":
                    recommendations.append(f"Column '{col}': Clean whitespace-only strings from the text entries.")
                elif exp == "expect_column_values_to_be_between":
                    recommendations.append(f"Column '{col}': Clamp or inspect values outside of standard standard deviation limits.")

        # Custom recommendations
        for col, issues in custom_failed_cols.items():
            for issue in issues:
                if "leakage" in issue.lower():
                    recommendations.append(f"Column '{col}': Potential target leakage candidate. Consider dropping this column.")
                elif "mixed" in issue.lower():
                    recommendations.append(f"Column '{col}': Mixed datatypes detected. Standardize values to a single type.")
                elif "low variance" in issue.lower():
                    recommendations.append(f"Column '{col}': Low variance. Consider dropping if it provides no information.")
                elif "empty" in issue.lower():
                    recommendations.append(f"Column '{col}': Column is completely empty. Drop it from the features.")

        # Deduplicate recommendations list
        recommendations = list(dict.fromkeys(recommendations))

        # Overall Success
        success = failed_checks == 0 and score >= 90.0

        return cls(
            success=success,
            score=score,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            schema_results=schema_results,
            ge_results=ge_results_list,
            custom_results=custom_results,
            recommendations=recommendations,
            dataset_name=dataset_name,
        )
