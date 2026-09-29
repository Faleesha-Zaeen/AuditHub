"""
AuditHub Quality - Quality Audit Engine
========================================

Analyzes datasets across multiple quality dimensions and flags anomalies by severity.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.ingestion.dataset_analyzer import ColumnCategory, DatasetAnalyzer, DatasetSummary
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AuditFinding:
    """Represents a single data quality finding.

    Attributes
    ----------
    dimension : str
        Quality dimension (e.g. Completeness, Consistency, Uniqueness, Accuracy).
    severity : str
        Severity level (e.g. INFO, WARNING, ERROR).
    column : str | None
        Associated column, if applicable.
    message : str
        Detailed description of the finding.
    metrics : dict
        Associated metrics (e.g. exact percentages, counts).
    """

    dimension: str
    severity: str
    column: Optional[str]
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAuditReport:
    """Aggregates all audit findings.

    Attributes
    ----------
    findings : list[AuditFinding]
        List of findings.
    summary : dict
        Counts of findings by severity.
    timestamp : str
        Generation timestamp.
    """

    findings: List[AuditFinding] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the report.
        """
        return {
            "findings": [
                {
                    "dimension": f.dimension,
                    "severity": f.severity,
                    "column": f.column,
                    "message": f.message,
                    "metrics": f.metrics,
                }
                for f in self.findings
            ],
            "summary": self.summary,
            "timestamp": self.timestamp,
        }


class DatasetAuditor:
    """Core auditing engine detecting statistical anomalies and quality issues."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the DatasetAuditor with parameters.

        Parameters
        ----------
        params : dict | None
            Auditing parameters. If None, loads from ConfigManager.
        """
        if params is None:
            try:
                self.params = ConfigManager().get_params()
            except Exception:
                self.params = {}
        else:
            self.params = params

        self.analyzer = DatasetAnalyzer()

    def audit(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> QualityAuditReport:
        """Audit the DataFrame and generate a QualityAuditReport.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to audit.
        target_column : str | None
            Name of the target column. If None, inferred dynamically.

        Returns
        -------
        QualityAuditReport
            Report containing all audit findings categorized by severity.
        """
        logger.info("Auditing dataset (shape: %s)", df.shape)
        findings: List[AuditFinding] = []

        # 1. Analyze the dataset to get metadata
        summary = self.analyzer.analyze(df)

        # Resolve target
        resolved_target = target_column
        if not resolved_target and summary.potential_targets:
            resolved_target = summary.potential_targets[0].get("column")

        # Load parameter thresholds
        data_params = self.params.get("data", {})
        quality_params = data_params.get("quality", {})
        ingest_params = data_params.get("ingestion", {})
        profiling_params = data_params.get("profiling", {})

        missing_threshold = ingest_params.get("missing_threshold", 0.5)
        corr_threshold = profiling_params.get("correlation_threshold", 0.8)

        # --- A. Completeness checks ---
        for col_info in summary.column_stats:
            col_name = col_info.name
            if col_info.null_pct > 0:
                if col_info.null_pct == 100.0:
                    findings.append(AuditFinding(
                        dimension="Completeness",
                        severity="ERROR",
                        column=col_name,
                        message=f"Column '{col_name}' is completely empty (100% missing values).",
                        metrics={"null_count": col_info.null_count, "null_pct": col_info.null_pct}
                    ))
                elif col_info.null_pct / 100.0 > missing_threshold:
                    findings.append(AuditFinding(
                        dimension="Completeness",
                        severity="WARNING",
                        column=col_name,
                        message=f"Column '{col_name}' has a high percentage of missing values: {col_info.null_pct}%.",
                        metrics={"null_count": col_info.null_count, "null_pct": col_info.null_pct}
                    ))
                else:
                    findings.append(AuditFinding(
                        dimension="Completeness",
                        severity="INFO",
                        column=col_name,
                        message=f"Column '{col_name}' contains missing values: {col_info.null_pct}%.",
                        metrics={"null_count": col_info.null_count, "null_pct": col_info.null_pct}
                    ))

        # --- B. Uniqueness checks (Duplicate rows/columns) ---
        dup_sum = summary.duplicate_summary
        duplicate_rows_count = dup_sum.get("duplicate_rows", 0)
        duplicate_rows_pct = dup_sum.get("duplicate_rows_pct", 0.0)

        if duplicate_rows_count > 0:
            severity = "WARNING" if duplicate_rows_pct > 5.0 else "INFO"
            findings.append(AuditFinding(
                dimension="Uniqueness",
                severity=severity,
                column=None,
                message=f"Dataset contains {duplicate_rows_count} duplicate rows ({duplicate_rows_pct}%).",
                metrics={"duplicate_count": duplicate_rows_count, "duplicate_pct": duplicate_rows_pct}
            ))

        # --- C. Distribution & Accuracy checks (Skew, Kurtosis, Outliers) ---
        for col_info in summary.column_stats:
            col_name = col_info.name
            if not col_info.is_numeric or col_info.inferred_type == ColumnCategory.CONSTANT:
                continue

            series = df[col_name].dropna()
            if series.empty:
                continue

            # Outlier detection (IQR method)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())
            outlier_pct = round((outlier_count / len(df)) * 100, 2)

            if outlier_count > 0:
                severity = "WARNING" if outlier_pct > 5.0 else "INFO"
                findings.append(AuditFinding(
                    dimension="Accuracy",
                    severity=severity,
                    column=col_name,
                    message=f"Column '{col_name}' contains {outlier_count} statistical outliers ({outlier_pct}%).",
                    metrics={"outlier_count": outlier_count, "outlier_pct": outlier_pct}
                ))

            # Skewness and Kurtosis
            try:
                skew = float(series.skew())
                kurt = float(series.kurt())
                if abs(skew) > 1.5:
                    findings.append(AuditFinding(
                        dimension="Consistency",
                        severity="WARNING" if abs(skew) > 3.0 else "INFO",
                        column=col_name,
                        message=f"Column '{col_name}' exhibits high skewness: {skew:.2f}.",
                        metrics={"skewness": skew}
                    ))
                if abs(kurt) > 3.0:
                    findings.append(AuditFinding(
                        dimension="Consistency",
                        severity="INFO",
                        column=col_name,
                        message=f"Column '{col_name}' exhibits high kurtosis: {kurt:.2f}.",
                        metrics={"kurtosis": kurt}
                    ))
            except Exception:
                pass

        # --- D. Class Imbalance checks ---
        if resolved_target and resolved_target in df.columns:
            target_series = df[resolved_target].dropna()
            if not target_series.empty and (
                pd.api.types.is_string_dtype(target_series.dtype) or
                pd.api.types.is_integer_dtype(target_series.dtype) or
                pd.api.types.is_bool_dtype(target_series.dtype)
            ):
                vc = target_series.value_counts()
                if len(vc) > 1:
                    min_class_pct = (vc.min() / vc.sum()) * 100.0
                    if min_class_pct < 10.0:
                        findings.append(AuditFinding(
                            dimension="Consistency",
                            severity="WARNING" if min_class_pct < 5.0 else "INFO",
                            column=resolved_target,
                            message=f"Class imbalance detected in target '{resolved_target}': minority class represents only {min_class_pct:.2f}% of labels.",
                            metrics={"class_distribution": vc.to_dict(), "minority_pct": min_class_pct}
                        ))

        # --- E. Multicollinearity & Redundancy checks ---
        numeric_cols = [c.name for c in summary.column_stats if c.is_numeric and c.inferred_type != ColumnCategory.CONSTANT]
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                redundant_pairs = [
                    (col, row)
                    for col in upper.columns
                    for row in upper.index
                    if upper.loc[row, col] > corr_threshold
                ]

                for col1, col2 in redundant_pairs:
                    val = float(corr_matrix.loc[col1, col2])
                    findings.append(AuditFinding(
                        dimension="Reliability",
                        severity="WARNING" if val > 0.95 else "INFO",
                        column=col1,
                        message=f"High multicollinearity detected between '{col1}' and '{col2}' (correlation = {val:.4f}).",
                        metrics={"correlated_column": col2, "correlation": val}
                    ))
            except Exception:
                pass

        # --- F. Constant columns check ---
        for col_info in summary.column_stats:
            if col_info.inferred_type == ColumnCategory.CONSTANT or col_info.nunique <= 1:
                findings.append(AuditFinding(
                    dimension="Reliability",
                    severity="WARNING",
                    column=col_info.name,
                    message=f"Column '{col_info.name}' is constant (contains only 1 unique value).",
                    metrics={"nunique": col_info.nunique}
                ))

        # --- G. Rare Categories check ---
        for col_info in summary.column_stats:
            col_name = col_info.name
            if col_info.inferred_type == ColumnCategory.CATEGORICAL and col_info.nunique > 1:
                series = df[col_name].dropna()
                vc = series.value_counts(normalize=True)
                rare_cats = vc[vc < 0.01]  # < 1%
                if not rare_cats.empty:
                    findings.append(AuditFinding(
                        dimension="Consistency",
                        severity="INFO",
                        column=col_name,
                        message=f"Column '{col_name}' contains {len(rare_cats)} rare category/categories (representing < 1% of values each).",
                        metrics={"rare_categories": rare_cats.to_dict()}
                    ))

        # --- H. Target Leakage check ---
        if resolved_target:
            for col_info in summary.column_stats:
                col_name = col_info.name
                if col_name == resolved_target or not col_info.is_numeric:
                    continue

                try:
                    corr = float(df[col_name].corr(df[resolved_target]))
                    if not pd.isna(corr) and abs(corr) > 0.95:
                        findings.append(AuditFinding(
                            dimension="Reliability",
                            severity="ERROR",
                            column=col_name,
                            message=f"High target leakage candidate: '{col_name}' correlation with target is {corr:.4f}.",
                            metrics={"target_correlation": corr}
                        ))
                except Exception:
                    pass

        # Compile Summary
        summary_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
        for f in findings:
            summary_counts[f.severity] = summary_counts.get(f.severity, 0) + 1

        return QualityAuditReport(findings=findings, summary=summary_counts)
