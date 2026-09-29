"""
AuditHub Drift - Data Drift Detection
======================================

Answers one question: *does my new data still look like the data my model was
trained on?*

Two datasets are compared column by column:

* **Population Stability Index (PSI)** is the headline score. It is computed
  the same way for both column kinds -- numeric columns are discretised into
  quantile bins taken from the reference data, categorical columns use their
  own categories -- so a single number is comparable across the whole dataset.
* A **statistical test** runs alongside it for significance: two-sample
  Kolmogorov-Smirnov for numeric columns, chi-square for categorical ones.

PSI leads because significance is not the same as importance: with 500,000
rows a KS test flags a shift far too small to matter, while PSI measures how
much the population actually moved. The test is reported so a small PSI backed
by a very low p-value is still visible.

Columns that only exist on one side are reported as added or removed rather
than scored, and columns whose type changed are flagged explicitly -- a
silently re-typed column is a data contract break, not a distribution shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.ingestion.dataset_analyzer import ColumnCategory, DatasetAnalyzer
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Drift status levels
# ---------------------------------------------------------------------------


class DriftStatus:
    """Verdict levels for a column or a whole dataset."""

    STABLE = "STABLE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    # Structural outcomes -- not distribution shifts.
    NEW_COLUMN = "NEW_COLUMN"
    REMOVED_COLUMN = "REMOVED_COLUMN"
    TYPE_CHANGED = "TYPE_CHANGED"
    NOT_COMPARABLE = "NOT_COMPARABLE"

    #: Statuses that represent an actual measured distribution shift.
    SCORED = (STABLE, WARNING, CRITICAL)
    #: Statuses that should draw the user's attention.
    ACTIONABLE = (WARNING, CRITICAL, TYPE_CHANGED, REMOVED_COLUMN)


# Default thresholds, overridden by configs/params.yaml -> data.drift
_DEFAULTS: Dict[str, Any] = {
    "psi_warning_threshold": 0.10,
    "psi_critical_threshold": 0.25,
    "significance_level": 0.05,
    "numeric_bins": 10,
    "rare_category_threshold": 0.01,
    "min_samples": 30,
    "dataset_critical_share": 0.30,
    "dataset_warning_share": 0.10,
}

# Laplace-style floor applied to bin proportions. PSI divides by the reference
# proportion, so an empty bin would otherwise make the score infinite.
_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ColumnDriftResult:
    """Drift verdict for a single column.

    Attributes
    ----------
    column : str
        Column name.
    status : str
        One of the :class:`DriftStatus` constants.
    column_kind : str
        ``"numeric"``, ``"categorical"`` or ``"unknown"``.
    psi : float | None
        Population Stability Index. ``None`` when not comparable.
    test_name : str | None
        Statistical test applied (``"ks_2samp"`` or ``"chi2_contingency"``).
    statistic : float | None
        Test statistic.
    p_value : float | None
        Test p-value.
    significant : bool | None
        Whether the test rejected the null at the configured alpha.
    reference_stats, current_stats : dict
        Summary statistics for each side, used by the UI and report.
    top_changes : list[dict]
        The bins or categories that moved most, with their before/after share.
    distribution : list[dict]
        Every bin or category with its reference and current share. Charts are
        built from this, so the UI and the report never re-bin the data
        themselves and therefore cannot disagree with the score.
    explanation : str
        Plain-language description of what changed and why it was flagged.
    """

    column: str
    status: str
    column_kind: str = "unknown"
    psi: Optional[float] = None
    test_name: Optional[str] = None
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    significant: Optional[bool] = None
    reference_stats: Dict[str, Any] = field(default_factory=dict)
    current_stats: Dict[str, Any] = field(default_factory=dict)
    top_changes: List[Dict[str, Any]] = field(default_factory=list)
    distribution: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    #: True when the verdict is driven by the missing-value rate rather than by
    #: a shift in the observed values. Lets callers avoid reporting the same
    #: change twice under two different headings.
    missing_rate_drift: bool = False

    def distribution_frame(self) -> pd.DataFrame:
        """Return the binned distribution as a long-format DataFrame.

        Shaped for direct plotting: one row per (bin, dataset) pair.
        """
        rows = []
        for entry in self.distribution:
            rows.append({"bin": entry["bin"], "dataset": "reference",
                         "percent": entry["reference_pct"]})
            rows.append({"bin": entry["bin"], "dataset": "current",
                         "percent": entry["current_pct"]})
        return pd.DataFrame(rows, columns=["bin", "dataset", "percent"])

    @property
    def drifted(self) -> bool:
        """True when the column moved enough to warrant attention."""
        return self.status in (DriftStatus.WARNING, DriftStatus.CRITICAL)

    def to_dict(self) -> Dict[str, Any]:
        """Return the result as a JSON-serialisable dictionary."""
        return {
            "column": self.column,
            "status": self.status,
            "column_kind": self.column_kind,
            "psi": self.psi,
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "reference_stats": self.reference_stats,
            "current_stats": self.current_stats,
            "top_changes": self.top_changes,
            "distribution": self.distribution,
            "explanation": self.explanation,
            "missing_rate_drift": self.missing_rate_drift,
        }


@dataclass
class DriftReport:
    """Whole-dataset drift outcome.

    Attributes
    ----------
    columns : list[ColumnDriftResult]
        Per-column results, ordered most-drifted first.
    overall_score : float
        Mean PSI across comparable columns.
    max_score : float
        Highest single-column PSI.
    status : str
        Dataset-level verdict.
    added_columns, removed_columns : list[str]
        Columns present on only one side.
    type_changed_columns : list[str]
        Columns whose inferred kind changed between datasets.
    reference_rows, current_rows : int
        Row counts of each dataset.
    thresholds : dict
        Thresholds in force for this run.
    timestamp : str
        Generation time.
    """

    columns: List[ColumnDriftResult] = field(default_factory=list)
    overall_score: float = 0.0
    max_score: float = 0.0
    status: str = DriftStatus.STABLE
    added_columns: List[str] = field(default_factory=list)
    removed_columns: List[str] = field(default_factory=list)
    type_changed_columns: List[str] = field(default_factory=list)
    reference_rows: int = 0
    current_rows: int = 0
    reference_name: str = "reference"
    current_name: str = "current"
    thresholds: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def drifted_columns(self) -> List[ColumnDriftResult]:
        """Columns flagged WARNING or CRITICAL, worst first."""
        return [c for c in self.columns if c.drifted]

    @property
    def compared_columns(self) -> List[ColumnDriftResult]:
        """Columns that actually received a drift score."""
        return [c for c in self.columns if c.status in DriftStatus.SCORED]

    def summary_counts(self) -> Dict[str, int]:
        """Return a count of columns per status."""
        counts: Dict[str, int] = {}
        for col in self.columns:
            counts[col.status] = counts.get(col.status, 0) + 1
        return counts

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        drifted = len(self.drifted_columns)
        compared = len(self.compared_columns)
        parts = [f"{drifted}/{compared} column(s) drifted", f"overall PSI {self.overall_score:.4f}"]
        if self.added_columns:
            parts.append(f"{len(self.added_columns)} added")
        if self.removed_columns:
            parts.append(f"{len(self.removed_columns)} removed")
        if self.type_changed_columns:
            parts.append(f"{len(self.type_changed_columns)} type change(s)")
        return f"{self.status}: " + "; ".join(parts) + "."

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return {
            "status": self.status,
            "overall_score": self.overall_score,
            "max_score": self.max_score,
            "reference_name": self.reference_name,
            "current_name": self.current_name,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "added_columns": self.added_columns,
            "removed_columns": self.removed_columns,
            "type_changed_columns": self.type_changed_columns,
            "summary_counts": self.summary_counts(),
            "summary": self.summary(),
            "thresholds": self.thresholds,
            "timestamp": self.timestamp,
            "columns": [c.to_dict() for c in self.columns],
        }

    def to_frame(self) -> pd.DataFrame:
        """Return per-column results as a DataFrame for display."""
        if not self.columns:
            return pd.DataFrame(
                columns=["column", "status", "column_kind", "psi", "p_value", "explanation"]
            )
        return pd.DataFrame([
            {
                "column": c.column,
                "status": c.status,
                "column_kind": c.column_kind,
                "psi": c.psi,
                "p_value": c.p_value,
                "explanation": c.explanation,
            }
            for c in self.columns
        ])


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Compares a current dataset against a reference dataset.

    Parameters
    ----------
    params : dict | None
        Full params mapping. Drift settings are read from
        ``data.drift``. Loaded from :class:`ConfigManager` when omitted.
    thresholds : dict | None
        Direct threshold overrides, applied on top of the config values.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> None:
        if params is None:
            try:
                params = ConfigManager().get_params()
            except Exception:
                params = {}

        configured = (params or {}).get("data", {}).get("drift", {}) or {}
        self.thresholds: Dict[str, Any] = {**_DEFAULTS, **configured, **(thresholds or {})}
        self.analyzer = DatasetAnalyzer(generate_fingerprint=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        reference_name: str = "reference",
        current_name: str = "current",
    ) -> DriftReport:
        """Compare two datasets and produce a drift report.

        Parameters
        ----------
        reference_df : pd.DataFrame
            The baseline -- typically the training data.
        current_df : pd.DataFrame
            The new data to check.
        columns : list[str] | None
            Restrict the comparison to these columns. Defaults to every
            column the two datasets share.
        reference_name, current_name : str
            Labels used in the report.

        Returns
        -------
        DriftReport
            Per-column verdicts plus a dataset-level score.

        Raises
        ------
        ValueError
            If either dataset is empty.
        """
        if reference_df is None or current_df is None:
            raise ValueError("Both a reference and a current dataset are required.")
        if reference_df.empty or current_df.empty:
            raise ValueError("Drift cannot be measured against an empty dataset.")

        ref_cols = list(reference_df.columns)
        cur_cols = list(current_df.columns)
        shared = [c for c in ref_cols if c in cur_cols]
        if columns is not None:
            shared = [c for c in shared if c in columns]

        report = DriftReport(
            reference_rows=int(len(reference_df)),
            current_rows=int(len(current_df)),
            reference_name=reference_name,
            current_name=current_name,
            thresholds=dict(self.thresholds),
        )

        # Structural differences are reported, never silently ignored.
        report.removed_columns = [str(c) for c in ref_cols if c not in cur_cols]
        report.added_columns = [str(c) for c in cur_cols if c not in ref_cols]

        for col in report.removed_columns:
            report.columns.append(ColumnDriftResult(
                column=col,
                status=DriftStatus.REMOVED_COLUMN,
                explanation=(
                    f"'{col}' exists in the {reference_name} dataset but not in "
                    f"{current_name}. Any model relying on it will fail or silently "
                    "receive an imputed value."
                ),
            ))
        for col in report.added_columns:
            report.columns.append(ColumnDriftResult(
                column=col,
                status=DriftStatus.NEW_COLUMN,
                explanation=(
                    f"'{col}' is new in the {current_name} dataset. It was not "
                    "available at training time, so it cannot be used by the "
                    "existing model without retraining."
                ),
            ))

        logger.info(
            "Comparing %d shared column(s) between %s (%d rows) and %s (%d rows)",
            len(shared), reference_name, len(reference_df), current_name, len(current_df),
        )

        for col in shared:
            result = self._compare_column(
                str(col), reference_df[col], current_df[col], reference_name, current_name
            )
            if result.status == DriftStatus.TYPE_CHANGED:
                report.type_changed_columns.append(str(col))
            report.columns.append(result)

        self._score_dataset(report)

        # Most severe and most drifted first, so the UI leads with what matters.
        severity = {
            DriftStatus.CRITICAL: 0,
            DriftStatus.TYPE_CHANGED: 1,
            DriftStatus.REMOVED_COLUMN: 2,
            DriftStatus.WARNING: 3,
            DriftStatus.NEW_COLUMN: 4,
            DriftStatus.NOT_COMPARABLE: 5,
            DriftStatus.STABLE: 6,
        }
        report.columns.sort(key=lambda c: (severity.get(c.status, 9), -(c.psi or 0.0)))

        logger.info("Drift detection complete. %s", report.summary())
        return report

    # ------------------------------------------------------------------
    # Column comparison
    # ------------------------------------------------------------------

    def _column_kind(self, reference: pd.Series, current: pd.Series) -> str:
        """Decide whether a column should be treated as numeric or categorical.

        Both sides must agree. A column that is numeric in one dataset and text
        in the other is a schema change, not a distribution shift.
        """
        ref_numeric = pd.api.types.is_numeric_dtype(reference.dtype)
        cur_numeric = pd.api.types.is_numeric_dtype(current.dtype)

        if ref_numeric and cur_numeric:
            # A numeric column with very few distinct values behaves like a
            # category; binning it into deciles would produce empty bins.
            distinct = len(set(reference.dropna().unique()) | set(current.dropna().unique()))
            if distinct <= max(2, int(self.thresholds["numeric_bins"]) // 2):
                return "categorical"
            return "numeric"
        if not ref_numeric and not cur_numeric:
            return "categorical"
        return "mismatch"

    def _compare_column(
        self,
        column: str,
        reference: pd.Series,
        current: pd.Series,
        reference_name: str,
        current_name: str,
    ) -> ColumnDriftResult:
        """Compare one column across the two datasets."""
        ref_clean = reference.dropna()
        cur_clean = current.dropna()
        min_samples = int(self.thresholds["min_samples"])

        kind = self._column_kind(reference, current)
        if kind == "mismatch":
            return ColumnDriftResult(
                column=column,
                status=DriftStatus.TYPE_CHANGED,
                column_kind="mismatch",
                reference_stats={"dtype": str(reference.dtype)},
                current_stats={"dtype": str(current.dtype)},
                explanation=(
                    f"'{column}' changed type: {reference.dtype} in {reference_name} "
                    f"but {current.dtype} in {current_name}. This is a schema break -- "
                    "distributions cannot be meaningfully compared."
                ),
            )

        if len(ref_clean) < min_samples or len(cur_clean) < min_samples:
            return ColumnDriftResult(
                column=column,
                status=DriftStatus.NOT_COMPARABLE,
                column_kind=kind,
                reference_stats={"non_null": int(len(ref_clean))},
                current_stats={"non_null": int(len(cur_clean))},
                explanation=(
                    f"'{column}' has too few non-null values to judge "
                    f"({len(ref_clean)} reference, {len(cur_clean)} current; "
                    f"{min_samples} required)."
                ),
            )

        if kind == "numeric":
            result = self._compare_numeric(column, ref_clean, cur_clean)
        else:
            result = self._compare_categorical(column, ref_clean, cur_clean)

        # Missing-value rate is part of drift: a column that suddenly arrives
        # 40% empty has changed even if its observed values have not.
        ref_missing = float(reference.isna().mean() * 100)
        cur_missing = float(current.isna().mean() * 100)
        result.reference_stats["missing_pct"] = round(ref_missing, 2)
        result.current_stats["missing_pct"] = round(cur_missing, 2)
        if abs(cur_missing - ref_missing) >= 5.0:
            if result.status == DriftStatus.STABLE:
                # Rewrite rather than append: leaving the "is stable" opener in
                # place would contradict the WARNING the column is about to get.
                result.status = DriftStatus.WARNING
                result.missing_rate_drift = True
                result.explanation = (
                    f"'{column}' has a stable distribution among the values that are "
                    f"present (PSI {result.psi:.4f}), but its missing rate moved from "
                    f"{ref_missing:.1f}% to {cur_missing:.1f}%."
                )
            else:
                result.missing_rate_drift = True
                result.explanation += (
                    f" Missing values also moved from {ref_missing:.1f}% to {cur_missing:.1f}%."
                )

        return result

    def _compare_numeric(
        self, column: str, reference: pd.Series, current: pd.Series
    ) -> ColumnDriftResult:
        """Compare a numeric column using quantile-bin PSI plus a KS test."""
        ref = pd.to_numeric(reference, errors="coerce").dropna().astype(float)
        cur = pd.to_numeric(current, errors="coerce").dropna().astype(float)

        edges = self._quantile_edges(ref, int(self.thresholds["numeric_bins"]))
        ref_props = self._binned_proportions(ref, edges)
        cur_props = self._binned_proportions(cur, edges)
        psi, contributions = self._psi(ref_props, cur_props)

        labels = [f"[{edges[i]:.4g}, {edges[i + 1]:.4g})" for i in range(len(edges) - 1)]
        top_changes = self._top_changes(labels, ref_props, cur_props, contributions)
        distribution = self._distribution(labels, ref_props, cur_props, contributions)

        statistic, p_value = stats.ks_2samp(ref, cur)
        alpha = float(self.thresholds["significance_level"])

        result = ColumnDriftResult(
            column=column,
            status=self._classify(psi),
            column_kind="numeric",
            psi=round(float(psi), 6),
            test_name="ks_2samp",
            statistic=round(float(statistic), 6),
            p_value=float(p_value),
            significant=bool(p_value < alpha),
            reference_stats=self._numeric_stats(ref),
            current_stats=self._numeric_stats(cur),
            top_changes=top_changes,
            distribution=distribution,
        )
        result.explanation = self._explain_numeric(result)
        return result

    def _compare_categorical(
        self, column: str, reference: pd.Series, current: pd.Series
    ) -> ColumnDriftResult:
        """Compare a categorical column using PSI plus a chi-square test."""
        ref_counts = reference.astype(str).value_counts()
        cur_counts = current.astype(str).value_counts()
        categories = sorted(set(ref_counts.index) | set(cur_counts.index))

        ref_props = np.array([ref_counts.get(c, 0) for c in categories], dtype=float)
        cur_props = np.array([cur_counts.get(c, 0) for c in categories], dtype=float)
        ref_props = ref_props / ref_props.sum()
        cur_props = cur_props / cur_props.sum()

        psi, contributions = self._psi(ref_props, cur_props)
        top_changes = self._top_changes(categories, ref_props, cur_props, contributions)
        # Cap the charted categories so a high-cardinality column does not
        # produce an unreadable plot; the score still uses every category.
        distribution = self._distribution(
            categories, ref_props, cur_props, contributions, limit=25
        )

        statistic, p_value = self._chi_square(ref_counts, cur_counts, categories)
        alpha = float(self.thresholds["significance_level"])

        new_cats = [c for c in cur_counts.index if c not in set(ref_counts.index)]
        gone_cats = [c for c in ref_counts.index if c not in set(cur_counts.index)]

        result = ColumnDriftResult(
            column=column,
            status=self._classify(psi),
            column_kind="categorical",
            psi=round(float(psi), 6),
            test_name="chi2_contingency",
            statistic=None if statistic is None else round(float(statistic), 6),
            p_value=None if p_value is None else float(p_value),
            significant=None if p_value is None else bool(p_value < alpha),
            reference_stats={
                "distinct": int(len(ref_counts)),
                "top": ref_counts.head(5).to_dict(),
            },
            current_stats={
                "distinct": int(len(cur_counts)),
                "top": cur_counts.head(5).to_dict(),
                "new_categories": new_cats[:10],
                "missing_categories": gone_cats[:10],
            },
            top_changes=top_changes,
            distribution=distribution,
        )
        result.explanation = self._explain_categorical(result, new_cats, gone_cats)
        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _quantile_edges(reference: pd.Series, bins: int) -> np.ndarray:
        """Build bin edges from the reference distribution's quantiles.

        Quantile bins keep every reference bin equally populated, which makes
        PSI stable for skewed columns where fixed-width bins would leave most
        bins empty. Edges are taken from the reference only -- the current data
        must be measured against the baseline, not against itself.
        """
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.nanquantile(reference.values, quantiles))
        if len(edges) < 2:
            # Constant column: one bin either side of the single value.
            value = float(reference.iloc[0])
            edges = np.array([value - 0.5, value + 0.5])
        # Open the outer edges so values beyond the reference range still land
        # in a bin instead of being dropped.
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    @staticmethod
    def _binned_proportions(series: pd.Series, edges: np.ndarray) -> np.ndarray:
        """Return the share of values falling in each bin."""
        counts, _ = np.histogram(series.values, bins=edges)
        total = counts.sum()
        if total == 0:
            return np.zeros(len(counts), dtype=float)
        return counts.astype(float) / float(total)

    @staticmethod
    def _psi(reference: np.ndarray, current: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute the Population Stability Index and each bin's contribution.

        ``PSI = sum((current - reference) * ln(current / reference))``

        Both proportion vectors are floored at a small epsilon: an empty
        reference bin would otherwise divide by zero and report infinite drift
        for what may be a single new value.
        """
        ref = np.clip(reference.astype(float), _EPSILON, None)
        cur = np.clip(current.astype(float), _EPSILON, None)
        ref = ref / ref.sum()
        cur = cur / cur.sum()

        contributions = (cur - ref) * np.log(cur / ref)
        return float(np.sum(contributions)), contributions

    def _chi_square(
        self,
        ref_counts: pd.Series,
        cur_counts: pd.Series,
        categories: List[str],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Run a chi-square test of independence over the category counts.

        Rare categories are pooled into a single bucket first: chi-square is
        unreliable when expected cell counts fall below ~5, and a long tail of
        one-off values would otherwise dominate the statistic.
        """
        threshold = float(self.thresholds["rare_category_threshold"])
        ref_total = float(ref_counts.sum())
        cur_total = float(cur_counts.sum())

        kept, pooled_ref, pooled_cur = [], 0.0, 0.0
        for cat in categories:
            r = float(ref_counts.get(cat, 0))
            c = float(cur_counts.get(cat, 0))
            if (r / ref_total < threshold) and (c / cur_total < threshold):
                pooled_ref += r
                pooled_cur += c
            else:
                kept.append((r, c))

        if pooled_ref or pooled_cur:
            kept.append((pooled_ref, pooled_cur))
        if len(kept) < 2:
            return None, None

        table = np.array([[r for r, _ in kept], [c for _, c in kept]], dtype=float)
        # Drop categories absent from both sides after pooling.
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:
            return None, None

        try:
            statistic, p_value, _, _ = stats.chi2_contingency(table)
        except ValueError as exc:
            logger.debug("Chi-square not applicable: %s", exc)
            return None, None
        return float(statistic), float(p_value)

    def _classify(self, psi: float) -> str:
        """Map a PSI value to a status using the configured thresholds."""
        if psi >= float(self.thresholds["psi_critical_threshold"]):
            return DriftStatus.CRITICAL
        if psi >= float(self.thresholds["psi_warning_threshold"]):
            return DriftStatus.WARNING
        return DriftStatus.STABLE

    def _score_dataset(self, report: DriftReport) -> None:
        """Set the dataset-level score and status from the column results."""
        scored = report.compared_columns
        if not scored:
            report.overall_score = 0.0
            report.max_score = 0.0
            # Structural breaks still matter even when nothing could be scored.
            report.status = (
                DriftStatus.CRITICAL
                if (report.removed_columns or report.type_changed_columns)
                else DriftStatus.STABLE
            )
            return

        psis = [c.psi or 0.0 for c in scored]
        report.overall_score = round(float(np.mean(psis)), 6)
        report.max_score = round(float(np.max(psis)), 6)

        drifted_share = len([c for c in scored if c.drifted]) / len(scored)
        critical_count = len([c for c in scored if c.status == DriftStatus.CRITICAL])

        if (
            critical_count
            and drifted_share >= float(self.thresholds["dataset_critical_share"])
        ) or report.type_changed_columns or report.removed_columns:
            report.status = DriftStatus.CRITICAL
        elif drifted_share >= float(self.thresholds["dataset_warning_share"]) or critical_count:
            report.status = DriftStatus.WARNING
        else:
            report.status = DriftStatus.STABLE

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_stats(series: pd.Series) -> Dict[str, Any]:
        """Summary statistics shown beside a numeric column's verdict."""
        return {
            "count": int(len(series)),
            "mean": round(float(series.mean()), 6),
            "std": round(float(series.std()), 6) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 6),
            "p25": round(float(series.quantile(0.25)), 6),
            "median": round(float(series.median()), 6),
            "p75": round(float(series.quantile(0.75)), 6),
            "max": round(float(series.max()), 6),
        }

    @staticmethod
    def _top_changes(
        labels: List[str],
        reference: np.ndarray,
        current: np.ndarray,
        contributions: np.ndarray,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the bins/categories that contributed most to the PSI."""
        order = np.argsort(-np.abs(contributions))[:limit]
        changes = []
        for idx in order:
            if idx >= len(labels):
                continue
            changes.append({
                "bin": str(labels[idx]),
                "reference_pct": round(float(reference[idx]) * 100, 2),
                "current_pct": round(float(current[idx]) * 100, 2),
                "psi_contribution": round(float(contributions[idx]), 6),
            })
        return changes

    @staticmethod
    def _distribution(
        labels: List[str],
        reference: np.ndarray,
        current: np.ndarray,
        contributions: np.ndarray,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return every bin's reference/current share, for charting.

        When ``limit`` is set, the bins contributing most to the PSI are kept
        so that a high-cardinality column still charts readably.
        """
        indices = list(range(min(len(labels), len(reference), len(current))))
        if limit is not None and len(indices) > limit:
            ranked = sorted(indices, key=lambda i: -abs(float(contributions[i])))[:limit]
            indices = sorted(ranked)

        return [
            {
                "bin": str(labels[i]),
                "reference_pct": round(float(reference[i]) * 100, 3),
                "current_pct": round(float(current[i]) * 100, 3),
                "psi_contribution": round(float(contributions[i]), 6),
            }
            for i in indices
        ]

    def _explain_numeric(self, result: ColumnDriftResult) -> str:
        """Describe a numeric column's shift in plain language."""
        ref, cur = result.reference_stats, result.current_stats
        mean_shift = cur["mean"] - ref["mean"]
        direction = "higher" if mean_shift > 0 else "lower"

        if result.status == DriftStatus.STABLE:
            base = f"'{result.column}' is stable (PSI {result.psi:.4f})."
        else:
            base = (
                f"'{result.column}' shifted {direction}: mean moved from "
                f"{ref['mean']:.4g} to {cur['mean']:.4g} and the median from "
                f"{ref['median']:.4g} to {cur['median']:.4g} (PSI {result.psi:.4f})."
            )

        if result.top_changes:
            worst = result.top_changes[0]
            base += (
                f" The largest move is the {worst['bin']} range, "
                f"{worst['reference_pct']:.1f}% -> {worst['current_pct']:.1f}% of rows."
            )

        # A significant test with a negligible PSI is usually sample size, not
        # a real shift -- say so rather than leaving a bare p-value.
        if result.significant and result.status == DriftStatus.STABLE:
            base += (
                f" The KS test is significant (p={result.p_value:.2}), but the PSI is "
                "below the warning threshold, so the shift is statistically "
                "detectable yet small."
            )
        elif not result.significant and result.status != DriftStatus.STABLE:
            base += (
                f" The KS test is not significant (p={result.p_value:.2}), so treat "
                "this as a tentative signal."
            )
        return base

    def _explain_categorical(
        self,
        result: ColumnDriftResult,
        new_categories: List[str],
        missing_categories: List[str],
    ) -> str:
        """Describe a categorical column's shift in plain language."""
        if result.status == DriftStatus.STABLE:
            base = f"'{result.column}' is stable (PSI {result.psi:.4f})."
        else:
            base = f"'{result.column}' changed its category mix (PSI {result.psi:.4f})."

        if result.top_changes:
            worst = result.top_changes[0]
            base += (
                f" '{worst['bin']}' moved from {worst['reference_pct']:.1f}% to "
                f"{worst['current_pct']:.1f}% of rows."
            )
        if new_categories:
            base += f" New categories appeared: {', '.join(map(str, new_categories[:5]))}."
        if missing_categories:
            base += f" Categories no longer present: {', '.join(map(str, missing_categories[:5]))}."
        if result.p_value is not None and result.significant and result.status == DriftStatus.STABLE:
            base += (
                f" The chi-square test is significant (p={result.p_value:.2}) but the PSI "
                "is small, which usually means a large sample rather than a real shift."
            )
        return base


__all__ = [
    "ColumnDriftResult",
    "DriftDetector",
    "DriftReport",
    "DriftStatus",
]
