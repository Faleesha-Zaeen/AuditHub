"""
AuditHub Repair - Intelligent Repair Engine
============================================

Implements dataset repairs including imputation, outlier handling, and type coercion
with a fully reversible execution history.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.ingestion.text_columns import classify_text_column
from src.utils.constants import REPAIRED_DATA_DIR
from src.utils.helpers import ensure_directory_exists, safe_filename, unique_filename
from src.utils.logger import get_logger

logger = get_logger(__name__)

#: Written into prose and identifier columns where a value is missing.
#: Deliberately conspicuous and parenthesised so it can never be mistaken for
#: real content, and so a reader scanning the file sees at once that nothing
#: was inferred there.
MISSING_TEXT_MARKER = "(missing)"

# Fraction of a text column's values that must parse as numbers before a
# mean/median request is honoured by coercing the column.
_NUMERIC_COERCION_THRESHOLD = 0.9


def _jsonable(value: Any) -> Any:
    """Convert a numpy/pandas scalar into something JSON and Streamlit accept."""
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


@dataclass
class AutoRepairResult:
    """Summary of a whole-dataset automatic repair run.

    Attributes
    ----------
    actions : list[dict]
        The repair log entries produced by this run.
    columns_imputed : dict[str, dict]
        Per-column record of strategy and fill value used.
    rows_before, rows_after : int
        Row counts either side of the run.
    duplicates_removed : int
        Number of duplicate rows dropped.
    target_rows_dropped : int
        Rows dropped because the target label was missing.
    skipped : list[str]
        Columns that could not be repaired, with the reason.
    """

    actions: List[Dict[str, Any]] = field(default_factory=list)
    columns_imputed: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rows_before: int = 0
    rows_after: int = 0
    duplicates_removed: int = 0
    target_rows_dropped: int = 0
    skipped: List[str] = field(default_factory=list)
    #: Identities found between numeric columns, e.g. total = quantity * price.
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    #: Per-column record of gaps computed exactly rather than estimated.
    derived: List[Dict[str, Any]] = field(default_factory=list)
    #: Per-column record of prose and identifier gaps marked rather than guessed.
    marked: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def cells_derived(self) -> int:
        """Number of gaps filled with an exactly computed value."""
        return sum(int(d.get("cells_filled", 0)) for d in self.derived)

    @property
    def cells_imputed(self) -> int:
        """Number of gaps filled with a statistical estimate."""
        return sum(int(i.get("cells_filled", 0)) for i in self.columns_imputed.values())

    @property
    def cells_marked(self) -> int:
        """Number of gaps filled with a visible placeholder instead of a guess.

        Counted apart from :attr:`cells_imputed` because these are explicitly
        *not* estimates -- nothing was inferred about them.
        """
        return sum(int(m.get("cells_filled", 0)) for m in self.marked.values())

    def to_dict(self) -> Dict[str, Any]:
        """Return the result as a plain dictionary."""
        return {
            "actions": self.actions,
            "columns_imputed": self.columns_imputed,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "duplicates_removed": self.duplicates_removed,
            "target_rows_dropped": self.target_rows_dropped,
            "skipped": self.skipped,
            "relationships": self.relationships,
            "derived": self.derived,
            "marked": self.marked,
            "cells_derived": self.cells_derived,
            "cells_imputed": self.cells_imputed,
            "cells_marked": self.cells_marked,
        }

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        parts = []
        if self.cells_derived:
            parts.append(f"{self.cells_derived} cell(s) derived exactly")
        parts.append(f"{len(self.columns_imputed)} column(s) imputed")
        if self.cells_marked:
            parts.append(
                f"{self.cells_marked} cell(s) marked '{MISSING_TEXT_MARKER}' "
                f"in {len(self.marked)} text column(s)"
            )
        if self.duplicates_removed:
            parts.append(f"{self.duplicates_removed} duplicate row(s) removed")
        if self.target_rows_dropped:
            parts.append(f"{self.target_rows_dropped} row(s) dropped for missing target")
        if self.skipped:
            parts.append(f"{len(self.skipped)} column(s) skipped")
        return "; ".join(parts) + "."


class DatasetRepairer:
    """Orchestrates automated and manual dataset repairs.

    Maintains a reversible step-by-step history of applied modifications.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the DatasetRepairer.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to repair. A copy will be stored internally.
        """
        self._current_df = df.copy()
        self._history: List[pd.DataFrame] = []  # Stack of past df states
        self._repair_log: List[Dict[str, Any]] = []  # History log entries

    @property
    def df(self) -> pd.DataFrame:
        """Get the current state of the repaired DataFrame."""
        return self._current_df

    def _save_state(self, action: str, details: Dict[str, Any]) -> None:
        """Save current state to history stack before applying changes.

        Prefer :meth:`_commit`, which records history only once the new frame
        has been computed successfully. Calling this directly risks leaving an
        undo entry behind for a repair that then failed or did nothing.
        """
        self._history.append(self._current_df.copy())
        self._repair_log.append({
            "step": len(self._history),
            "action": action,
            "details": details,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        logger.debug("Saved state before action: %s", action)

    def _commit(self, action: str, details: Dict[str, Any], new_df: pd.DataFrame) -> pd.DataFrame:
        """Atomically record history and adopt a successfully computed frame.

        The new frame is built *before* this is called, so an operation that
        raises or turns out to be a no-op never pollutes the undo stack -- the
        previous behaviour left a log entry and a history frame behind even
        when the repair had failed, making the next Undo revert the wrong step.
        """
        self._history.append(self._current_df.copy())
        self._repair_log.append({
            "step": len(self._history),
            "action": action,
            "details": details,
            "timestamp": pd.Timestamp.now().isoformat(),
        })
        self._current_df = new_df
        logger.debug("Committed repair action: %s", action)
        return self._current_df

    def revert(self) -> pd.DataFrame:
        """Revert the last applied repair action.

        Returns
        -------
        pd.DataFrame
            The restored DataFrame state.
        """
        if not self._history:
            logger.warning("No repair steps to revert.")
            return self._current_df

        self._current_df = self._history.pop()
        log_entry = self._repair_log.pop()
        logger.info("Reverted repair step %d: %s", log_entry["step"], log_entry["action"])
        return self._current_df

    def get_log(self) -> List[Dict[str, Any]]:
        """Return the list of applied repair log steps."""
        return self._repair_log

    # ------------------------------------------------------------------
    # Repair Operations
    # ------------------------------------------------------------------

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows from the dataset.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame.
        """
        before_count = len(self._current_df)
        duplicates_mask = self._current_df.duplicated()
        duplicates_count = duplicates_mask.sum()

        if duplicates_count == 0:
            logger.info("No duplicates found, skipping.")
            return self._current_df

        new_df = self._current_df.drop_duplicates().reset_index(drop=True)
        self._commit("remove_duplicates", {"rows_removed": int(duplicates_count)}, new_df)
        logger.info("Removed %d duplicate rows (reduced from %d to %d)", duplicates_count, before_count, len(self._current_df))
        return self._current_df

    @staticmethod
    def _as_numeric(series: pd.Series) -> Optional[pd.Series]:
        """Return ``series`` as numbers, or ``None`` if it is not numeric data.

        A column of numbers stored as text -- the usual result of one stray
        ``"?"`` in a CSV -- is coerced so that mean/median remain available.
        Genuinely textual columns return ``None`` so the caller can fall back.
        """
        if pd.api.types.is_numeric_dtype(series.dtype):
            return series
        if pd.api.types.is_bool_dtype(series.dtype):
            return series.astype("Float64")

        # Datetimes must never take this path: pd.to_numeric happily turns them
        # into nanosecond integers, so the "median date" comes back as
        # 1.6875648e+18 and the whole column degrades to object.
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return None

        non_null = series.dropna()
        if non_null.empty:
            return None

        coerced = pd.to_numeric(series, errors="coerce")
        parse_rate = float(coerced.notna().sum()) / float(len(non_null))
        if parse_rate >= _NUMERIC_COERCION_THRESHOLD:
            return coerced
        return None

    @staticmethod
    def _cast_fill_value(series: pd.Series, value: Any) -> Any:
        """Coerce a user-supplied constant to the column's own dtype.

        Streamlit text inputs hand back strings, so filling a float column with
        the typed value ``"0"`` used to turn the whole column into ``object``
        and break every downstream numeric step.
        """
        if value is None:
            return None

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(numeric):
                raise ValueError(
                    f"Cannot fill numeric column '{series.name}' with non-numeric value {value!r}."
                )
            if pd.api.types.is_integer_dtype(series.dtype) and float(numeric) % 1 == 0:
                return int(numeric)
            return numeric

        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            parsed = pd.to_datetime(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(parsed):
                raise ValueError(
                    f"Cannot fill datetime column '{series.name}' with unparseable value {value!r}."
                )
            return parsed

        return value

    @staticmethod
    def _restore_dtype(original: pd.Series, filled: pd.Series) -> pd.Series:
        """Cast an imputed column back to the original column's dtype.

        Keeps integer columns integral: the mean of an ``age`` column is
        ``32.5``, but writing that into a column of whole numbers produces a
        dataset no one would call clean.
        """
        if pd.api.types.is_integer_dtype(original.dtype):
            rounded = filled.round()
            try:
                return rounded.astype(original.dtype)
            except (TypeError, ValueError):
                return rounded.astype("Int64")

        if pd.api.types.is_numeric_dtype(original.dtype):
            try:
                return filled.astype(original.dtype)
            except (TypeError, ValueError):
                return filled

        return filled

    def _resolve_fill(
        self,
        series: pd.Series,
        strategy: str,
        fill_value: Any,
    ) -> Tuple[Any, str, Optional[str]]:
        """Work out the value to impute with.

        Returns
        -------
        tuple[Any, str, str | None]
            The fill value, the strategy actually used, and an optional note
            explaining any fallback that was applied.
        """
        note: Optional[str] = None

        if strategy == "constant":
            if fill_value is None or (isinstance(fill_value, str) and not fill_value.strip()):
                raise ValueError(
                    f"Strategy 'constant' needs a fill value for column '{series.name}'."
                )
            return self._cast_fill_value(series, fill_value), "constant", None

        if strategy in ("mean", "median"):
            # Dates have a genuine mean and median; pandas computes both
            # natively and returns a Timestamp, so the column keeps its dtype.
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                observed = series.dropna()
                if observed.empty:
                    raise ValueError(
                        f"Cannot compute {strategy} for column '{series.name}': "
                        "no observed dates."
                    )
                value = observed.mean() if strategy == "mean" else observed.median()
                return value, strategy, None

            numeric = self._as_numeric(series)
            if numeric is None:
                # A text column has no mean; mode is the honest equivalent.
                note = f"'{strategy}' is undefined for non-numeric column; used mode instead"
                strategy = "mode"
            else:
                value = numeric.mean() if strategy == "mean" else numeric.median()
                if pd.isna(value):
                    raise ValueError(
                        f"Cannot compute {strategy} for column '{series.name}': no observed values."
                    )
                if not pd.api.types.is_numeric_dtype(series.dtype):
                    note = f"column was stored as text and coerced to numeric to compute the {strategy}"
                return value, strategy, note

        if strategy == "mode":
            mode_vals = series.mode(dropna=True)
            if mode_vals.empty:
                raise ValueError(
                    f"Cannot compute mode for column '{series.name}': no observed values."
                )
            return mode_vals.iloc[0], "mode", note

        raise ValueError(
            f"Unknown imputation strategy: '{strategy}'. "
            "Expected one of: mean, median, mode, constant."
        )

    def impute_missing(
        self,
        column: str,
        strategy: str = "mean",
        fill_value: Any = None,
    ) -> pd.DataFrame:
        """Impute missing values in a column.

        Type-aware: numbers stored as text are coerced before mean/median are
        computed, ``mean``/``median`` on a genuinely textual column falls back
        to ``mode`` rather than raising, integer columns stay integral, and a
        constant is cast to the column's dtype instead of silently turning the
        column into ``object``.

        Parameters
        ----------
        column : str
            Column name.
        strategy : str
            Imputation strategy: 'mean', 'median', 'mode', or 'constant'.
        fill_value : Any
            Value to use if strategy is 'constant'.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame. Unchanged if the column has no missing
            values, in which case nothing is added to the undo history.

        Raises
        ------
        ValueError
            If the strategy is unknown, or the requested fill is impossible
            (e.g. a non-numeric constant for a numeric column).
        """
        if column not in self._current_df.columns:
            logger.warning("Column '%s' not found.", column)
            return self._current_df

        series = self._current_df[column]
        null_count = int(series.isna().sum())
        if null_count == 0:
            logger.debug("No missing values in column '%s'.", column)
            return self._current_df

        value, used_strategy, note = self._resolve_fill(series, strategy, fill_value)

        # Round before filling, not after: a nullable Int64 column rejects a
        # float outright, so the mean of an integer column must be rounded
        # first rather than repaired afterwards.
        if pd.api.types.is_integer_dtype(series.dtype) and isinstance(value, (int, float, np.number)):
            rounded = int(round(float(value)))
            if float(value) != float(rounded):
                note = (note + "; " if note else "") + (
                    f"{used_strategy} was {float(value):g}, rounded to {rounded} to keep the column integral"
                )
            value = rounded

        filled = series.fillna(value)
        filled = self._restore_dtype(series, filled)

        # Report the value actually written, which may differ from the raw
        # statistic once integer rounding has been applied.
        applied_value = filled[series.isna()].iloc[0] if null_count else value

        new_df = self._current_df.copy()
        new_df[column] = filled

        details: Dict[str, Any] = {
            "column": column,
            "strategy": used_strategy,
            "null_count": null_count,
            "fill_value": _jsonable(applied_value),
        }
        if used_strategy != strategy:
            details["requested_strategy"] = strategy
        if note:
            details["note"] = note

        self._commit("impute_missing", details, new_df)
        logger.info(
            "Imputed %d missing value(s) in '%s' using %s strategy (value=%s).",
            null_count, column, used_strategy, applied_value,
        )
        return self._current_df

    def coerce_types(self, column: str, target_type: str) -> pd.DataFrame:
        """Coerce column values to target datatype.

        Parameters
        ----------
        column : str
            Column name.
        target_type : str
            Target datatype (e.g. 'int', 'float', 'datetime', 'str').

        Returns
        -------
        pd.DataFrame
            The updated DataFrame.
        """
        if column not in self._current_df.columns:
            return self._current_df

        original = self._current_df[column]
        original_type = str(original.dtype)

        if target_type in ("int", "int64"):
            # Nullable Int64 keeps genuine gaps visible instead of turning
            # every missing value into a real, indistinguishable 0.
            converted = pd.to_numeric(original, errors="coerce").round().astype("Int64")
        elif target_type in ("float", "float64"):
            converted = pd.to_numeric(original, errors="coerce")
        elif target_type == "datetime":
            converted = pd.to_datetime(original, errors="coerce")
        elif target_type == "str":
            converted = original.astype("string")
        else:
            raise ValueError(
                f"Unknown target type: '{target_type}'. "
                "Expected one of: int, float, datetime, str."
            )

        lost = int(converted.isna().sum() - original.isna().sum())
        details: Dict[str, Any] = {
            "column": column,
            "target_type": target_type,
            "original_type": original_type,
        }
        if lost > 0:
            details["values_not_convertible"] = lost

        new_df = self._current_df.copy()
        new_df[column] = converted
        self._commit("coerce_types", details, new_df)

        if lost > 0:
            logger.warning(
                "Coercing '%s' to %s made %d value(s) null; they could not be converted.",
                column, target_type, lost,
            )
        logger.info("Coerced column '%s' to type %s.", column, target_type)
        return self._current_df

    def scale_numeric(self, column: str, method: str = "standard") -> pd.DataFrame:
        """Scale a numerical column using MinMaxScaler or StandardScaler.

        Parameters
        ----------
        column : str
            Column name.
        method : str
            Scaling method: 'standard' or 'minmax'.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame.
        """
        if column not in self._current_df.columns:
            return self._current_df

        series = self._current_df[column]
        if not pd.api.types.is_numeric_dtype(series.dtype):
            logger.warning("Scaling skipped: Column '%s' is not numeric.", column)
            return self._current_df

        if method == "standard":
            scaler: Any = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"Unknown scaling method: '{method}'. Expected 'standard' or 'minmax'."
            )

        # Fill NaNs temporarily so the scaler can fit, then restore them: a
        # missing value must stay missing rather than become the mean.
        null_mask = series.isna()
        vals = series.fillna(series.mean()).values.reshape(-1, 1)
        scaled = pd.Series(scaler.fit_transform(vals).flatten(), index=series.index)
        scaled[null_mask] = np.nan

        new_df = self._current_df.copy()
        new_df[column] = scaled
        self._commit("scale_numeric", {"column": column, "method": method}, new_df)

        logger.info("Scaled column '%s' using %s method.", column, method)
        return self._current_df

    def drop_constant_columns(self) -> pd.DataFrame:
        """Identify and drop columns containing only 1 unique value.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame.
        """
        constant_cols = [
            col for col in self._current_df.columns
            if self._current_df[col].nunique() <= 1
        ]

        if not constant_cols:
            logger.info("No constant columns detected.")
            return self._current_df

        new_df = self._current_df.drop(columns=constant_cols)
        self._commit(
            "drop_constant_columns",
            {"dropped_columns": [str(c) for c in constant_cols]},
            new_df,
        )
        logger.info("Dropped constant columns: %s", constant_cols)
        return self._current_df

    def handle_rare_categories(self, column: str, threshold: float = 0.01) -> pd.DataFrame:
        """Group low-frequency categorical values into an 'Other' category.

        Parameters
        ----------
        column : str
            Column name.
        threshold : float
            Proportion frequency threshold (default 0.01 = 1%).

        Returns
        -------
        pd.DataFrame
            The updated DataFrame.
        """
        if column not in self._current_df.columns:
            return self._current_df

        series = self._current_df[column]
        vc = series.value_counts(normalize=True)
        rare_cats = vc[vc < threshold].index.tolist()

        if not rare_cats:
            return self._current_df

        # Preserve nulls: a missing value is not a rare category, and turning
        # it into "Other" would hide it from every completeness metric.
        grouped = series.where(series.isna() | ~series.isin(rare_cats), "Other")

        new_df = self._current_df.copy()
        new_df[column] = grouped
        self._commit(
            "handle_rare_categories",
            {
                "column": column,
                "threshold": threshold,
                "grouped_categories": [_jsonable(c) for c in rare_cats],
            },
            new_df,
        )
        logger.info("Grouped %d rare categories in '%s' to 'Other'.", len(rare_cats), column)
        return self._current_df

    def clamp_outliers(self, column: str, method: str = "iqr") -> pd.DataFrame:
        """Clamp outliers in a numerical column to the percentile boundaries.

        Parameters
        ----------
        column : str
            Column name.
        method : str
            Outlier boundary strategy: 'iqr' (1.5 IQR bounds) or 'zscore' (3 std bounds).

        Returns
        -------
        pd.DataFrame
            The updated DataFrame.
        """
        if column not in self._current_df.columns:
            return self._current_df

        series = self._current_df[column]
        if not pd.api.types.is_numeric_dtype(series.dtype):
            return self._current_df

        non_null = series.dropna()
        if non_null.empty:
            return self._current_df

        if method == "iqr":
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        else:  # zscore
            mean = non_null.mean()
            std = non_null.std()
            lower_bound = mean - 3.0 * std
            upper_bound = mean + 3.0 * std

        clamped = series.clip(lower=lower_bound, upper=upper_bound)
        affected = int((series.notna() & (series != clamped)).sum())

        new_df = self._current_df.copy()
        new_df[column] = clamped
        self._commit(
            "clamp_outliers",
            {
                "column": column,
                "method": method,
                "lower_bound": _jsonable(lower_bound),
                "upper_bound": _jsonable(upper_bound),
                "values_clamped": affected,
            },
            new_df,
        )
        logger.info("Clamped outliers in '%s' to range [%.4f, %.4f].", column, lower_bound, upper_bound)
        return self._current_df

    def encode_categorical(self, column: str, method: str = "ordinal") -> pd.DataFrame:
        """Encode categorical column into numerical codes.

        Parameters
        ----------
        column : str
            Column name.
        method : str
            Encoding method: 'ordinal' or 'label'.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame.
        """
        if column not in self._current_df.columns:
            return self._current_df

        series = self._current_df[column]

        categories = series.astype("category")
        codes = categories.cat.codes
        # cat.codes marks missing values as -1, which reads as a real category
        # downstream; keep them null so completeness stays honest.
        codes = codes.where(series.notna(), pd.NA).astype("Int64")

        mapping = {
            _jsonable(cat): int(idx)
            for idx, cat in enumerate(categories.cat.categories)
        }

        new_df = self._current_df.copy()
        new_df[column] = codes
        self._commit(
            "encode_categorical",
            {"column": column, "method": method, "mapping": mapping},
            new_df,
        )
        logger.info("Encoded categorical column '%s' into numerical codes.", column)
        return self._current_df

    # ------------------------------------------------------------------
    # Whole-dataset automatic repair
    # ------------------------------------------------------------------

    def drop_missing_target_rows(self, target_column: str) -> pd.DataFrame:
        """Drop rows whose target label is missing.

        A missing label is not imputable: inventing one fabricates ground truth
        and inflates every downstream model metric. Such rows are removed
        instead, and the count is reported.
        """
        if target_column not in self._current_df.columns:
            return self._current_df

        mask = self._current_df[target_column].isna()
        dropped = int(mask.sum())
        if dropped == 0:
            return self._current_df

        new_df = self._current_df.loc[~mask].reset_index(drop=True)
        self._commit(
            "drop_missing_target_rows",
            {"column": target_column, "rows_dropped": dropped},
            new_df,
        )
        logger.info(
            "Dropped %d row(s) with a missing '%s' label (labels are never imputed).",
            dropped, target_column,
        )
        return self._current_df

    def _derive_missing(
        self,
        target_column: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Any]]:
        """Fill gaps that an exact arithmetic identity can compute.

        Returns the discovered relationships and a record of what each one
        filled. Failures are logged and swallowed: derivation is an
        optimisation over imputation, never a prerequisite for it.
        """
        from src.repair.relationships import apply_relationships, discover_relationships

        try:
            relationships = discover_relationships(self._current_df)
            if not relationships:
                return [], [], []

            filled, records = apply_relationships(self._current_df, relationships)
            if not records:
                return [r.to_dict() for r in relationships], [], relationships

            self._commit(
                "derive_from_relationships",
                {
                    "relationships": [r.expression for r in relationships],
                    "cells_filled": sum(r["cells_filled"] for r in records),
                },
                filled,
            )
            logger.info(
                "Derived %d cell(s) from %d relationship(s) before imputation.",
                sum(r["cells_filled"] for r in records), len(relationships),
            )
            return [r.to_dict() for r in relationships], records, relationships
        except Exception:  # noqa: BLE001 - imputation still covers the gaps
            # Logged with the traceback: a silent skip here once hid a total
            # failure of derivation behind a one-line warning, and the run
            # still looked successful because imputation filled every gap.
            logger.exception(
                "Relationship derivation failed; falling back to imputation."
            )
            return [], [], []

    def auto_repair(
        self,
        target_column: Optional[str] = None,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        remove_duplicate_rows: bool = True,
        drop_empty_columns: bool = True,
        derive_relationships: bool = True,
    ) -> AutoRepairResult:
        """Repair the whole dataset in one pass.

        Order matters: fully empty columns go first (they cannot be imputed
        from anything), then duplicate rows, then rows missing the target
        label, and only then per-column imputation -- so statistics are
        computed from the rows that will actually survive.

        Parameters
        ----------
        target_column : str | None
            Label column. Never imputed; rows missing it are dropped instead.
        numeric_strategy : str
            Strategy for numeric columns. Defaults to ``median``, which is not
            dragged by the outliers the auditor is designed to find.
        categorical_strategy : str
            Strategy for non-numeric columns. Defaults to ``mode``.
        remove_duplicate_rows : bool
            Drop exact duplicate rows.
        drop_empty_columns : bool
            Drop columns that are entirely null.
        derive_relationships : bool
            Look for exact arithmetic identities between numeric columns
            and use them to compute missing values before falling back to
            statistical imputation.

        Returns
        -------
        AutoRepairResult
            What was done, per column, plus rows affected and any skips.
        """
        result = AutoRepairResult(rows_before=len(self._current_df))
        log_start = len(self._repair_log)

        if drop_empty_columns:
            empty_cols = [c for c in self._current_df.columns if self._current_df[c].isna().all()]
            if empty_cols:
                new_df = self._current_df.drop(columns=empty_cols)
                self._commit(
                    "drop_empty_columns",
                    {"dropped_columns": [str(c) for c in empty_cols]},
                    new_df,
                )
                for col in empty_cols:
                    result.skipped.append(f"{col}: dropped (100% missing, nothing to impute from)")

        if remove_duplicate_rows:
            before = len(self._current_df)
            self.remove_duplicates()
            result.duplicates_removed = before - len(self._current_df)

        if target_column and target_column in self._current_df.columns:
            before = len(self._current_df)
            self.drop_missing_target_rows(target_column)
            result.target_rows_dropped = before - len(self._current_df)

        # Derive before estimating. Where a column is a function of others
        # (total = quantity x price), the missing value is computable, and a
        # median would put a plausible but wrong number in its place.
        relationship_objects = []
        gaps_before_imputation = None
        if derive_relationships:
            result.relationships, result.derived, relationship_objects = (
                self._derive_missing(target_column)
            )
            if relationship_objects:
                gaps_before_imputation = self._current_df.isna()

        for column in list(self._current_df.columns):
            if column == target_column:
                continue

            series = self._current_df[column]
            null_count = int(series.isna().sum())
            if null_count == 0:
                continue

            numeric_like = self._as_numeric(series) is not None

            # Prose and identifiers get a visible marker, never a neighbour's
            # value. The most common review is not an estimate of a missing
            # review, it is a different customer's words; the most common email
            # address is not an estimate of a missing one, it is someone else's.
            role = "non_text" if numeric_like else classify_text_column(series)
            if role in ("free_text", "identifier"):
                try:
                    self.impute_missing(
                        column, strategy="constant", fill_value=MISSING_TEXT_MARKER
                    )
                except (ValueError, TypeError) as exc:
                    result.skipped.append(f"{column}: {exc}")
                    logger.warning("Auto-repair skipped column '%s': %s", column, exc)
                    continue

                result.marked[str(column)] = {
                    "role": role,
                    "marker": MISSING_TEXT_MARKER,
                    "cells_filled": null_count,
                    "note": (
                        f"{role.replace('_', ' ')} column -- gaps marked rather than "
                        "estimated, because no other row's value stands in for a "
                        "missing one here"
                    ),
                }
                continue

            strategy = numeric_strategy if numeric_like else categorical_strategy

            try:
                self.impute_missing(column, strategy=strategy)
            except (ValueError, TypeError) as exc:
                result.skipped.append(f"{column}: {exc}")
                logger.warning("Auto-repair skipped column '%s': %s", column, exc)
                continue

            entry = self._repair_log[-1]
            result.columns_imputed[str(column)] = {
                "strategy": entry["details"].get("strategy"),
                "fill_value": entry["details"].get("fill_value"),
                "cells_filled": null_count,
                "note": entry["details"].get("note"),
            }

        # Imputation may have supplied the operands a relationship needed. Any
        # cell that was still missing at that point holds an estimate, so
        # recomputing it from the identity replaces a guess with arithmetic and
        # leaves the cleaned file internally consistent.
        if relationship_objects and gaps_before_imputation is not None:
            reconciled = self._reconcile_relationships(
                relationship_objects, gaps_before_imputation
            )
            if reconciled:
                result.derived.extend(reconciled)

        result.rows_after = len(self._current_df)
        result.actions = self._repair_log[log_start:]
        logger.info("Auto-repair complete: %s", result.summary())
        return result

    def _reconcile_relationships(
        self,
        relationships: List[Any],
        gaps_before_imputation: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Recompute estimated cells that a relationship can now determine.

        Only cells that were still missing when imputation began are touched,
        so an originally observed value is never overwritten.
        """
        records: List[Dict[str, Any]] = []
        frame = self._current_df.copy()
        changed_total = 0

        for relationship in relationships:
            column = relationship.target
            if column not in frame.columns or column not in gaps_before_imputation:
                continue

            was_missing = gaps_before_imputation[column].reindex(frame.index, fill_value=False)
            if not was_missing.any():
                continue

            computed = relationship.solve_for(frame, column)
            if computed is None:
                continue

            usable = was_missing & computed.notna() & np.isfinite(computed)
            differs = usable & ~np.isclose(
                pd.to_numeric(frame[column], errors="coerce").astype("float64"),
                computed, rtol=1e-6, atol=1e-9, equal_nan=False,
            )
            count = int(differs.sum())
            if not count:
                continue

            values = computed[differs]
            if pd.api.types.is_integer_dtype(frame[column].dtype):
                values = values.round().astype("Int64")
            frame.loc[differs, column] = values

            records.append({
                "column": column,
                "expression": relationship.expression,
                "cells_filled": count,
                "method": "reconciled",
            })
            changed_total += count

        if changed_total:
            self._commit(
                "reconcile_relationships",
                {
                    "relationships": [r.expression for r in relationships],
                    "cells_corrected": changed_total,
                },
                frame,
            )
            logger.info(
                "Replaced %d estimated cell(s) with values computed from a relationship.",
                changed_total,
            )
        return records

    # ------------------------------------------------------------------
    # Verification & export
    # ------------------------------------------------------------------

    def verify_clean(self, target_column: Optional[str] = None) -> List[str]:
        """Check the current frame against the export guarantees.

        Returns
        -------
        list[str]
            One message per violated guarantee. An empty list means the frame
            is safe to export: no nulls, no duplicate rows, no residual
            missing-value sentinels, and no all-null columns.
        """
        from src.ingestion.cleaning import MISSING_SENTINELS

        issues: List[str] = []
        df = self._current_df

        null_counts = df.isna().sum()
        offending = {str(c): int(n) for c, n in null_counts.items() if n > 0}
        if offending:
            issues.append(f"Columns still containing nulls: {offending}")

        dup_rows = int(df.duplicated().sum())
        if dup_rows:
            issues.append(f"{dup_rows} duplicate row(s) remain.")

        for col in df.columns:
            series = df[col]
            if series.dtype != object:
                continue
            as_str = series.astype(str).str.strip().str.lower()
            hits = int(as_str.isin(MISSING_SENTINELS).sum())
            if hits:
                issues.append(f"Column '{col}' still contains {hits} missing-value marker(s).")

        if target_column and target_column in df.columns:
            if int(df[target_column].isna().sum()):
                issues.append(f"Target column '{target_column}' still contains missing labels.")

        return issues

    def to_csv_bytes(self) -> bytes:
        """Serialise the repaired frame exactly as it will be written to disk.

        Written without the index and as UTF-8 with a BOM, so the file opens
        correctly in Excel and re-loads into pandas with the same shape.
        """
        return self._current_df.to_csv(index=False).encode("utf-8-sig")

    def build_manifest(
        self,
        dataset_name: str = "dataset",
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the repair manifest that accompanies an exported dataset."""
        df = self._current_df
        return {
            "dataset": dataset_name,
            "generated_at": pd.Timestamp.now().isoformat(),
            "target_column": target_column,
            "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
            "columns": [
                {
                    "name": str(c),
                    "dtype": str(df[c].dtype),
                    "nulls": int(df[c].isna().sum()),
                }
                for c in df.columns
            ],
            "repair_log": self.get_log(),
            "verification": {
                "issues": self.verify_clean(target_column=target_column),
                "passed": not self.verify_clean(target_column=target_column),
            },
        }

    def export(
        self,
        dataset_name: str = "dataset",
        output_dir: Optional[Union[str, Path]] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """Write the repaired dataset and its manifest to disk.

        Parameters
        ----------
        dataset_name : str
            Base name used for the output files.
        output_dir : str | Path | None
            Destination directory. Defaults to ``data/repaired``.
        target_column : str | None
            Target column, recorded in the manifest and verified.

        Returns
        -------
        tuple[Path, Path]
            Paths to the written CSV and JSON manifest.
        """
        out_dir = Path(output_dir) if output_dir else REPAIRED_DATA_DIR
        ensure_directory_exists(out_dir)

        stem = Path(safe_filename(dataset_name)).stem or "dataset"
        csv_name = unique_filename(f"{stem}_repaired.csv")
        csv_path = out_dir / csv_name
        manifest_path = out_dir / f"{Path(csv_name).stem}_manifest.json"

        self._current_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        manifest = self.build_manifest(dataset_name=dataset_name, target_column=target_column)
        manifest["output_file"] = str(csv_path.resolve())
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)

        issues = manifest["verification"]["issues"]
        if issues:
            logger.warning("Exported %s with %d unmet guarantee(s): %s", csv_path.name, len(issues), issues)
        else:
            logger.info("Exported clean dataset to %s (all guarantees met).", csv_path)

        return csv_path, manifest_path

    # ------------------------------------------------------------------
    # Preview Repair API
    # ------------------------------------------------------------------

    def preview_repair(self, repair_func_name: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return a preview copy of the DataFrame after applying a specific repair.

        Does not modify the internal state of the repairer.

        Parameters
        ----------
        repair_func_name : str
            Name of the repairer method to invoke.
        *args : Any
            Arguments for the method.
        **kwargs : Any
            Keyword arguments for the method.

        Returns
        -------
        pd.DataFrame
            DataFrame copy with the repair applied.
        """
        # Create a mock repairer and invoke the method
        mock_repairer = DatasetRepairer(self._current_df)
        func = getattr(mock_repairer, repair_func_name, None)
        if func is None:
            raise ValueError(f"Unknown repair method: '{repair_func_name}'")

        func(*args, **kwargs)
        return mock_repairer.df
