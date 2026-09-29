"""
AuditHub Health - Health Score Calculator
===========================================

Computes composite dataset health score and logs score deductions transparently.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from src.ingestion.dataset_analyzer import ColumnCategory, DatasetAnalyzer, DatasetSummary
from src.quality.auditor import DatasetAuditor, QualityAuditReport
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthDimensionScore:
    """Represents a score breakdown for a single health dimension."""

    name: str
    score: float
    deductions: List[str] = field(default_factory=list)


@dataclass
class DatasetHealthReport:
    """Overall dataset health score report.

    Attributes
    ----------
    overall_score : float
        Overall health score (0.0 - 100.0).
    grade : str
        Grade letter (A, B, C, D, F).
    dimensions : dict[str, HealthDimensionScore]
        Score breakdown by dimension.
    explanations : list[str]
        List of all deductions across dimensions.
    """

    overall_score: float = 0.0
    grade: str = "F"
    dimensions: Dict[str, HealthDimensionScore] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the health report to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the report.
        """
        return {
            "overall_score": round(self.overall_score, 2),
            "grade": self.grade,
            "dimensions": {
                name: {
                    "score": round(dim.score, 2),
                    "deductions": dim.deductions
                }
                for name, dim in self.dimensions.items()
            },
            "explanations": self.explanations,
        }


def _resolve_grade(score: float, ranges: Dict[str, Dict[str, Any]]) -> str:
    """Map a health score to a grade letter.

    Parameters
    ----------
    score : float
        Overall health score (0.0 - 100.0).
    ranges : dict
        Mapping of grade letter to a dict with at least a ``min`` key.

    Returns
    -------
    str
        The highest grade whose lower bound ``score`` reaches. Falls back to
        the lowest-bounded grade so that a score can never match nothing.
    """
    if not ranges:
        return "F"

    ordered = sorted(
        ranges.items(),
        key=lambda item: float(item[1].get("min", 0.0)),
        reverse=True,
    )
    for letter, bounds in ordered:
        if score >= float(bounds.get("min", 0.0)):
            return letter

    return ordered[-1][0]


class HealthScoreCalculator:
    """Computes multidimensional dataset health scores based on audit findings."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the HealthScoreCalculator.

        Parameters
        ----------
        params : dict | None
            Health score parameters. If None, loads from ConfigManager.
        """
        if params is None:
            try:
                self.params = ConfigManager().get_params()
            except Exception:
                self.params = {}
        else:
            self.params = params

        self.analyzer = DatasetAnalyzer()
        self.auditor = DatasetAuditor(params=self.params)

    def calculate(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> DatasetHealthReport:
        """Calculate the multidimensional dataset health report.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to score.
        target_column : str | None
            Name of the target column.

        Returns
        -------
        DatasetHealthReport
            Consolidated health report.
        """
        logger.info("Calculating dataset health score (shape: %s)", df.shape)

        # 1. Profile and audit the dataset
        summary = self.analyzer.analyze(df)
        audit_report = self.auditor.audit(df, target_column=target_column)

        # 2. Get weights from parameters
        data_params = self.params.get("data", {})
        quality_params = data_params.get("quality", {})
        weights = quality_params.get("health_score_weights", {
            "completeness": 0.25,
            "uniqueness": 0.15,
            "consistency": 0.20,
            "accuracy": 0.25,
            "timeliness": 0.15,  # Note: Timeliness mapped to Reliability in dimensions
        })

        # Mapped dimensions
        # Completeness, Uniqueness, Consistency, Accuracy, Reliability, Readiness, Integrity
        dim_scores: Dict[str, HealthDimensionScore] = {}

        # -------------------------------------------------------------
        # Pillar 1: Completeness (0 - 100)
        # Deduct based on missing value percentages
        # -------------------------------------------------------------
        comp_deductions = []
        comp_score = 100.0
        for col_info in summary.column_stats:
            if col_info.null_pct > 0:
                ded = col_info.null_pct * 0.5  # Max 50 points per column
                comp_score -= ded
                comp_deductions.append(
                    f"Deducted {ded:.1f} points from Completeness: Column '{col_info.name}' is {col_info.null_pct}% missing."
                )
        comp_score = max(0.0, comp_score)
        dim_scores["completeness"] = HealthDimensionScore("Completeness", comp_score, comp_deductions)

        # -------------------------------------------------------------
        # Pillar 2: Uniqueness (0 - 100)
        # Deduct based on duplicate rows
        # -------------------------------------------------------------
        uniq_deductions = []
        dup_sum = summary.duplicate_summary
        dup_pct = dup_sum.get("duplicate_rows_pct", 0.0)
        uniq_score = max(0.0, 100.0 - (dup_pct * 2.0))  # 2 points per % duplicate
        if dup_pct > 0:
            uniq_deductions.append(
                f"Deducted {dup_pct * 2.0:.1f} points from Uniqueness: {dup_pct}% of rows are duplicates."
            )
        dim_scores["uniqueness"] = HealthDimensionScore("Uniqueness", uniq_score, uniq_deductions)

        # -------------------------------------------------------------
        # Pillar 3: Consistency (0 - 100)
        # Deduct based on class imbalance, mixed datatypes, rare categories
        # -------------------------------------------------------------
        cons_deductions = []
        cons_score = 100.0
        for finding in audit_report.findings:
            if finding.dimension == "Consistency":
                ded = 15.0 if finding.severity == "WARNING" else 5.0
                cons_score -= ded
                cons_deductions.append(
                    f"Deducted {ded:.1f} points from Consistency: {finding.message}"
                )
        cons_score = max(0.0, cons_score)
        dim_scores["consistency"] = HealthDimensionScore("Consistency", cons_score, cons_deductions)

        # -------------------------------------------------------------
        # Pillar 4: Accuracy (0 - 100)
        # Deduct based on outliers, skewness
        # -------------------------------------------------------------
        acc_deductions = []
        acc_score = 100.0
        for finding in audit_report.findings:
            if finding.dimension == "Accuracy":
                ded = 10.0 if finding.severity == "WARNING" else 3.0
                acc_score -= ded
                acc_deductions.append(
                    f"Deducted {ded:.1f} points from Accuracy: {finding.message}"
                )
        acc_score = max(0.0, acc_score)
        dim_scores["accuracy"] = HealthDimensionScore("Accuracy", acc_score, acc_deductions)

        # -------------------------------------------------------------
        # Pillar 5: Reliability (0 - 100)
        # Deduct based on leakage candidates, constant columns, multicollinearity
        # -------------------------------------------------------------
        rel_deductions = []
        rel_score = 100.0
        for finding in audit_report.findings:
            if finding.dimension == "Reliability":
                ded = 20.0 if finding.severity == "ERROR" else (10.0 if finding.severity == "WARNING" else 5.0)
                rel_score -= ded
                rel_deductions.append(
                    f"Deducted {ded:.1f} points from Reliability: {finding.message}"
                )
        rel_score = max(0.0, rel_score)
        dim_scores["reliability"] = HealthDimensionScore("Reliability", rel_score, rel_deductions)

        # -------------------------------------------------------------
        # Pillar 6: Readiness & Integrity (0 - 100)
        # Check target existence, valid ID uniqueness
        # -------------------------------------------------------------
        read_deductions = []
        read_score = 100.0

        # No target column found/inferred
        resolved_target = target_column
        if not resolved_target and summary.potential_targets:
            resolved_target = summary.potential_targets[0].get("column")
        if not resolved_target:
            read_score -= 20.0
            read_deductions.append("Deducted 20.0 points from Readiness: No target column identified in dataset.")

        # ID column integrity check: verify if ID column contains duplicates
        for col_info in summary.column_stats:
            if col_info.is_id_candidate and col_info.null_pct > 0:
                read_score -= 10.0
                read_deductions.append(
                    f"Deducted 10.0 points from Readiness: Identifier column '{col_info.name}' contains nulls."
                )

        read_score = max(0.0, read_score)
        dim_scores["readiness"] = HealthDimensionScore("Readiness", read_score, read_deductions)

        # 3. Compute overall score using weights
        # Map weights dictionary
        w_completeness = weights.get("completeness", 0.25)
        w_uniqueness = weights.get("uniqueness", 0.15)
        w_consistency = weights.get("consistency", 0.20)
        w_accuracy = weights.get("accuracy", 0.25)
        w_reliability = weights.get("reliability", weights.get("timeliness", 0.15))

        # Re-normalize weights to sum to 1.0
        total_w = w_completeness + w_uniqueness + w_consistency + w_accuracy + w_reliability
        if total_w > 0:
            w_completeness /= total_w
            w_uniqueness /= total_w
            w_consistency /= total_w
            w_accuracy /= total_w
            w_reliability /= total_w

        overall_score = (
            dim_scores["completeness"].score * w_completeness +
            dim_scores["uniqueness"].score * w_uniqueness +
            dim_scores["consistency"].score * w_consistency +
            dim_scores["accuracy"].score * w_accuracy +
            dim_scores["reliability"].score * w_reliability
        )

        # Apply Readiness penalty to overall score
        # e.g., if readiness is low, overall score gets scaled down or penalized
        if dim_scores["readiness"].score < 100.0:
            penalty = (100.0 - dim_scores["readiness"].score) * 0.1
            overall_score = max(0.0, overall_score - penalty)

        # 4. Map score to Grade
        ranges = self.params.get("schema", {}).get("health_schema", {}).get("grade_ranges", {
            "A": {"min": 90, "max": 100},
            "B": {"min": 75, "max": 90},
            "C": {"min": 60, "max": 75},
            "D": {"min": 40, "max": 60},
            "F": {"min": 0, "max": 40},
        })

        # Award the highest grade whose lower bound the score reaches. Testing
        # `min <= score <= max` instead would drop any score landing between two
        # bands (89.5, 74.6, ...) through to the default "F".
        grade = _resolve_grade(overall_score, ranges)

        # Compile all explanations
        explanations = []
        for name, dim in dim_scores.items():
            explanations.extend(dim.deductions)

        return DatasetHealthReport(
            overall_score=overall_score,
            grade=grade,
            dimensions=dim_scores,
            explanations=explanations,
        )
