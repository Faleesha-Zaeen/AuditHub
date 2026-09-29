"""
AuditHub Lineage - Dataset Version Comparison
==============================================

Compares two versions of a dataset and reports what actually changed between
them: shape, schema, types, missing values, duplicates, distributions and the
target column.

Distribution changes are delegated to :class:`~src.drift.detector.DriftDetector`
rather than reimplemented -- "did this column's distribution move?" is the same
question whether it is asked of two versions or of training vs production data.

Renames are only reported when they can be established from the data itself: a
dropped column and a new column are treated as a rename when their *contents*
match, not because their names look similar. A guess would be worse than
reporting an add and a drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.drift.detector import DriftDetector, DriftStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)

# A dropped/added column pair is only called a rename above this similarity.
_RENAME_CONFIDENCE_THRESHOLD = 0.90

# Percentage-point change in a column's missing rate that warrants a warning.
_MISSING_WARNING_DELTA = 5.0
_MISSING_CRITICAL_DELTA = 20.0


class ChangeType:
    """Kinds of change reported between two dataset versions."""

    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_RENAMED = "column_renamed"
    TYPE_CHANGED = "type_changed"
    MISSING_CHANGED = "missing_changed"
    DISTRIBUTION_CHANGED = "distribution_changed"
    CARDINALITY_CHANGED = "cardinality_changed"
    TARGET_CHANGED = "target_changed"


@dataclass
class ColumnChange:
    """A single column-level difference between two versions.

    Attributes
    ----------
    column : str
        Column the change concerns.
    change_type : str
        One of the :class:`ChangeType` constants.
    severity : str
        ``INFO``, ``WARNING`` or ``CRITICAL``.
    detail : str
        Plain-language description.
    before, after : Any
        The values either side of the change, where meaningful.
    """

    column: str
    change_type: str
    severity: str = "INFO"
    detail: str = ""
    before: Any = None
    after: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the change as a JSON-serialisable dictionary."""
        return {
            "column": self.column,
            "change_type": self.change_type,
            "severity": self.severity,
            "detail": self.detail,
            "before": self.before,
            "after": self.after,
        }


@dataclass
class VersionComparison:
    """Full comparison between two dataset versions."""

    left_name: str = "V1"
    right_name: str = "V2"
    left_rows: int = 0
    right_rows: int = 0
    left_columns: int = 0
    right_columns: int = 0
    left_missing_pct: float = 0.0
    right_missing_pct: float = 0.0
    left_duplicates: int = 0
    right_duplicates: int = 0
    added_columns: List[str] = field(default_factory=list)
    removed_columns: List[str] = field(default_factory=list)
    renamed_columns: List[Dict[str, Any]] = field(default_factory=list)
    column_changes: List[ColumnChange] = field(default_factory=list)
    target_column: Optional[str] = None
    drift: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def row_delta(self) -> int:
        """Change in row count."""
        return self.right_rows - self.left_rows

    @property
    def column_delta(self) -> int:
        """Change in column count."""
        return self.right_columns - self.left_columns

    def changes_by_severity(self, severity: str) -> List[ColumnChange]:
        """Return changes at a given severity."""
        return [c for c in self.column_changes if c.severity == severity]

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        parts = [
            f"rows {self.left_rows:,} -> {self.right_rows:,} ({self.row_delta:+,})",
            f"columns {self.left_columns} -> {self.right_columns} ({self.column_delta:+d})",
        ]
        if self.added_columns:
            parts.append(f"{len(self.added_columns)} added")
        if self.removed_columns:
            parts.append(f"{len(self.removed_columns)} removed")
        if self.renamed_columns:
            parts.append(f"{len(self.renamed_columns)} renamed")
        critical = len(self.changes_by_severity("CRITICAL"))
        if critical:
            parts.append(f"{critical} critical change(s)")
        return "; ".join(parts) + "."

    def to_dict(self) -> Dict[str, Any]:
        """Return the comparison as a JSON-serialisable dictionary."""
        return {
            "left_name": self.left_name,
            "right_name": self.right_name,
            "left_rows": self.left_rows,
            "right_rows": self.right_rows,
            "row_delta": self.row_delta,
            "left_columns": self.left_columns,
            "right_columns": self.right_columns,
            "column_delta": self.column_delta,
            "left_missing_pct": self.left_missing_pct,
            "right_missing_pct": self.right_missing_pct,
            "left_duplicates": self.left_duplicates,
            "right_duplicates": self.right_duplicates,
            "added_columns": self.added_columns,
            "removed_columns": self.removed_columns,
            "renamed_columns": self.renamed_columns,
            "target_column": self.target_column,
            "column_changes": [c.to_dict() for c in self.column_changes],
            "drift": self.drift,
            "summary": self.summary(),
            "timestamp": self.timestamp,
        }

    def to_frame(self) -> pd.DataFrame:
        """Return the column-level changes as a DataFrame."""
        if not self.column_changes:
            return pd.DataFrame(columns=["column", "change_type", "severity", "detail"])
        return pd.DataFrame([
            {
                "column": c.column,
                "change_type": c.change_type,
                "severity": c.severity,
                "detail": c.detail,
            }
            for c in self.column_changes
        ])

    def headline_frame(self) -> pd.DataFrame:
        """Return the V1 -> V2 headline metrics as a DataFrame.

        Every cell is a formatted string. Mixing integers and strings in one
        column produces an object column that Arrow refuses to serialise, which
        crashed the page that displays this table.
        """
        return pd.DataFrame([
            {"metric": "Rows", "before": f"{self.left_rows:,}",
             "after": f"{self.right_rows:,}", "change": f"{self.row_delta:+,}"},
            {"metric": "Columns", "before": f"{self.left_columns:,}",
             "after": f"{self.right_columns:,}", "change": f"{self.column_delta:+d}"},
            {"metric": "Missing %", "before": f"{self.left_missing_pct:.2f}%",
             "after": f"{self.right_missing_pct:.2f}%",
             "change": f"{self.right_missing_pct - self.left_missing_pct:+.2f} pp"},
            {"metric": "Duplicate rows", "before": f"{self.left_duplicates:,}",
             "after": f"{self.right_duplicates:,}",
             "change": f"{self.right_duplicates - self.left_duplicates:+d}"},
        ])


class VersionComparator:
    """Compares two versions of a dataset.

    Parameters
    ----------
    detect_drift : bool
        Also run distribution drift over the shared columns. Enabled by
        default; disable it for a fast schema-only comparison.
    drift_detector : DriftDetector | None
        Detector to reuse. One is created when omitted.
    """

    def __init__(
        self,
        detect_drift: bool = True,
        drift_detector: Optional[DriftDetector] = None,
    ) -> None:
        self.detect_drift = detect_drift
        self._detector = drift_detector or DriftDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_name: str = "V1",
        right_name: str = "V2",
        target_column: Optional[str] = None,
    ) -> VersionComparison:
        """Compare two dataset versions.

        Parameters
        ----------
        left : pd.DataFrame
            The older version.
        right : pd.DataFrame
            The newer version.
        left_name, right_name : str
            Labels used in the output.
        target_column : str | None
            Target column, checked for presence and class changes.

        Returns
        -------
        VersionComparison
            Everything that differs between the two versions.

        Raises
        ------
        ValueError
            If either version is ``None``.
        """
        if left is None or right is None:
            raise ValueError("Two dataset versions are required for comparison.")

        comparison = VersionComparison(
            left_name=left_name,
            right_name=right_name,
            left_rows=int(len(left)),
            right_rows=int(len(right)),
            left_columns=int(len(left.columns)),
            right_columns=int(len(right.columns)),
            left_missing_pct=_missing_pct(left),
            right_missing_pct=_missing_pct(right),
            left_duplicates=int(left.duplicated().sum()),
            right_duplicates=int(right.duplicated().sum()),
            target_column=target_column,
        )

        left_cols = [str(c) for c in left.columns]
        right_cols = [str(c) for c in right.columns]
        removed = [c for c in left_cols if c not in right_cols]
        added = [c for c in right_cols if c not in left_cols]

        # Renames are resolved first so a renamed column is not double-counted
        # as both an addition and a removal.
        renames = self._detect_renames(left, right, removed, added)
        renamed_pairs = {(r["from"], r["to"]) for r in renames}
        comparison.renamed_columns = renames

        renamed_from = {f for f, _ in renamed_pairs}
        renamed_to = {t for _, t in renamed_pairs}
        comparison.removed_columns = [c for c in removed if c not in renamed_from]
        comparison.added_columns = [c for c in added if c not in renamed_to]

        for rename in renames:
            comparison.column_changes.append(ColumnChange(
                column=rename["to"],
                change_type=ChangeType.COLUMN_RENAMED,
                severity="WARNING",
                detail=(
                    f"'{rename['from']}' appears to have been renamed to '{rename['to']}' "
                    f"({rename['evidence']}). Downstream code referring to the old name "
                    "will break."
                ),
                before=rename["from"],
                after=rename["to"],
            ))

        for col in comparison.added_columns:
            comparison.column_changes.append(ColumnChange(
                column=col,
                change_type=ChangeType.COLUMN_ADDED,
                severity="INFO",
                detail=f"'{col}' is new in {right_name} ({right[col].dtype}).",
                after=str(right[col].dtype),
            ))
        for col in comparison.removed_columns:
            severity = "CRITICAL" if col == target_column else "WARNING"
            comparison.column_changes.append(ColumnChange(
                column=col,
                change_type=ChangeType.COLUMN_REMOVED,
                severity=severity,
                detail=(
                    f"'{col}' was present in {left_name} but is gone from {right_name}."
                    + (" This is the target column." if col == target_column else "")
                ),
                before=str(left[col].dtype),
            ))

        shared = [c for c in left_cols if c in right_cols]
        for col in shared:
            comparison.column_changes.extend(
                self._compare_shared_column(col, left[col], right[col], left_name, right_name)
            )

        self._compare_target(comparison, left, right, target_column)

        if self.detect_drift and shared:
            try:
                drift = self._detector.detect(
                    left[shared], right[shared],
                    reference_name=left_name, current_name=right_name,
                )
                comparison.drift = drift.to_dict()
                for result in drift.columns:
                    # A column flagged purely because its missing rate moved is
                    # already reported as a MISSING_CHANGED entry; repeating it
                    # here under "distribution changed" would double-count it
                    # and contradict its own PSI.
                    if result.drifted and not result.missing_rate_drift:
                        comparison.column_changes.append(ColumnChange(
                            column=result.column,
                            change_type=ChangeType.DISTRIBUTION_CHANGED,
                            severity="CRITICAL" if result.status == DriftStatus.CRITICAL else "WARNING",
                            detail=result.explanation,
                            before=result.reference_stats.get("mean"),
                            after=result.current_stats.get("mean"),
                        ))
            except ValueError as exc:
                # Too small or empty to compare: schema results still stand.
                logger.info("Skipping distribution comparison: %s", exc)

        logger.info("Version comparison complete: %s", comparison.summary())
        return comparison

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_renames(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        removed: List[str],
        added: List[str],
    ) -> List[Dict[str, Any]]:
        """Pair dropped and new columns whose contents match.

        Two forms of evidence are accepted:

        * identical values in the same order, when the row counts match --
          conclusive;
        * a very high overlap between the two value sets -- strong enough when
          rows were added or removed alongside the rename.

        Each dropped column is matched at most once, best candidate first.
        """
        if not removed or not added:
            return []

        candidates: List[Tuple[float, str, str, str]] = []
        for old in removed:
            for new in added:
                score, evidence = self._rename_similarity(left[old], right[new])
                if score >= _RENAME_CONFIDENCE_THRESHOLD:
                    candidates.append((score, old, new, evidence))

        candidates.sort(key=lambda c: -c[0])
        renames: List[Dict[str, Any]] = []
        used_old, used_new = set(), set()
        for score, old, new, evidence in candidates:
            if old in used_old or new in used_new:
                continue
            used_old.add(old)
            used_new.add(new)
            renames.append({
                "from": old, "to": new,
                "confidence": round(float(score), 4),
                "evidence": evidence,
            })
        return renames

    @staticmethod
    def _rename_similarity(old: pd.Series, new: pd.Series) -> Tuple[float, str]:
        """Score how likely it is that ``new`` is ``old`` under another name."""
        old_numeric = pd.api.types.is_numeric_dtype(old.dtype)
        new_numeric = pd.api.types.is_numeric_dtype(new.dtype)
        if old_numeric != new_numeric:
            return 0.0, ""

        if len(old) == len(new):
            left_values = old.reset_index(drop=True)
            right_values = new.reset_index(drop=True)
            both_null = left_values.isna() & right_values.isna()
            equal = (left_values == right_values) | both_null
            if bool(equal.all()):
                return 1.0, "identical values in the same order"

        old_values = set(old.dropna().astype(str).unique())
        new_values = set(new.dropna().astype(str).unique())
        if not old_values or not new_values:
            return 0.0, ""

        overlap = len(old_values & new_values)
        union = len(old_values | new_values)
        jaccard = overlap / union if union else 0.0
        if jaccard >= _RENAME_CONFIDENCE_THRESHOLD:
            return jaccard, f"{jaccard:.0%} of distinct values are shared"
        return jaccard, ""

    def _compare_shared_column(
        self,
        column: str,
        left: pd.Series,
        right: pd.Series,
        left_name: str,
        right_name: str,
    ) -> List[ColumnChange]:
        """Report type, missing-rate and cardinality changes for one column."""
        changes: List[ColumnChange] = []

        if str(left.dtype) != str(right.dtype):
            # A numeric column becoming text is a contract break; widening an
            # int to a float is routine.
            both_numeric = (
                pd.api.types.is_numeric_dtype(left.dtype)
                and pd.api.types.is_numeric_dtype(right.dtype)
            )
            changes.append(ColumnChange(
                column=column,
                change_type=ChangeType.TYPE_CHANGED,
                severity="INFO" if both_numeric else "CRITICAL",
                detail=(
                    f"'{column}' changed type from {left.dtype} in {left_name} to "
                    f"{right.dtype} in {right_name}."
                    + ("" if both_numeric else " Parsing downstream may now fail.")
                ),
                before=str(left.dtype),
                after=str(right.dtype),
            ))

        left_missing = float(left.isna().mean() * 100)
        right_missing = float(right.isna().mean() * 100)
        delta = right_missing - left_missing
        if abs(delta) >= _MISSING_WARNING_DELTA:
            severity = "CRITICAL" if abs(delta) >= _MISSING_CRITICAL_DELTA else "WARNING"
            direction = "rose" if delta > 0 else "fell"
            changes.append(ColumnChange(
                column=column,
                change_type=ChangeType.MISSING_CHANGED,
                severity=severity if delta > 0 else "INFO",
                detail=(
                    f"Missing values in '{column}' {direction} from "
                    f"{left_missing:.1f}% to {right_missing:.1f}% ({delta:+.1f} pp)."
                ),
                before=round(left_missing, 2),
                after=round(right_missing, 2),
            ))

        # Cardinality is how a categorical column's shape changes; a jump can
        # break one-hot encoding trained on the old set.
        if not pd.api.types.is_numeric_dtype(left.dtype):
            left_unique = int(left.nunique(dropna=True))
            right_unique = int(right.nunique(dropna=True))
            if left_unique and right_unique:
                ratio = right_unique / left_unique
                if ratio >= 2.0 or ratio <= 0.5:
                    changes.append(ColumnChange(
                        column=column,
                        change_type=ChangeType.CARDINALITY_CHANGED,
                        severity="WARNING",
                        detail=(
                            f"'{column}' went from {left_unique} to {right_unique} distinct "
                            "values. Encoders fitted on the old set will not cover the new one."
                        ),
                        before=left_unique,
                        after=right_unique,
                    ))

        return changes

    @staticmethod
    def _compare_target(
        comparison: VersionComparison,
        left: pd.DataFrame,
        right: pd.DataFrame,
        target_column: Optional[str],
    ) -> None:
        """Report changes to the target column's class balance."""
        if not target_column:
            return
        if target_column not in left.columns or target_column not in right.columns:
            return

        left_counts = left[target_column].value_counts(normalize=True)
        right_counts = right[target_column].value_counts(normalize=True)
        if len(left_counts) > 25 or len(right_counts) > 25:
            return  # continuous target; distribution drift already covers it

        new_classes = [c for c in right_counts.index if c not in set(left_counts.index)]
        gone_classes = [c for c in left_counts.index if c not in set(right_counts.index)]

        details: List[str] = []
        if new_classes:
            details.append(f"new class(es): {', '.join(map(str, new_classes[:5]))}")
        if gone_classes:
            details.append(f"class(es) no longer present: {', '.join(map(str, gone_classes[:5]))}")

        shifts = []
        for cls in set(left_counts.index) & set(right_counts.index):
            before = float(left_counts[cls]) * 100
            after = float(right_counts[cls]) * 100
            if abs(after - before) >= 5.0:
                shifts.append(f"'{cls}' {before:.1f}% -> {after:.1f}%")
        if shifts:
            details.append("class balance moved: " + "; ".join(shifts[:5]))

        if details:
            comparison.column_changes.append(ColumnChange(
                column=target_column,
                change_type=ChangeType.TARGET_CHANGED,
                severity="CRITICAL" if (new_classes or gone_classes) else "WARNING",
                detail=f"Target '{target_column}': " + "; ".join(details) + ".",
                before=left_counts.round(4).to_dict(),
                after=right_counts.round(4).to_dict(),
            ))


def _missing_pct(df: pd.DataFrame) -> float:
    """Return the share of cells that are null, as a percentage."""
    total = len(df) * len(df.columns)
    if total == 0:
        return 0.0
    return round(float(df.isna().sum().sum()) / total * 100, 4)


__all__ = [
    "ChangeType",
    "ColumnChange",
    "VersionComparator",
    "VersionComparison",
]
