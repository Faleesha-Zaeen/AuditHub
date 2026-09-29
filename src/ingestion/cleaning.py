"""
AuditHub Ingestion - Structural Cleaning
=========================================

Normalisation that runs *at load time*, before any analysis, so that every
downstream module (analyzer, auditor, health, repair) sees the same honest
view of the data.

The central problem this solves: a value like ``"?"``, ``"N/A"`` or ``"-"`` is
missing data, but pandas reads it as an ordinary string. Every completeness
metric then under-reports, and mode imputation can "fill" a gap with another
missing marker. Converting sentinels to real nulls up front makes the whole
platform's notion of "missing" consistent.

Each function is pure -- it returns a new DataFrame plus a list of
:class:`CleaningAction` records describing what changed -- so the UI and the
reports can show the user exactly what was done to their file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ingestion.text_columns import casing_class, dominant_casing, is_free_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Missing-value sentinels
# ---------------------------------------------------------------------------
# Matched case-insensitively against stripped cell values.
#
# Deliberately does NOT include "0" or "-1": those are frequently legitimate
# values, and silently nulling them would corrupt real data.
#
# The set is split in two, because a token's meaning depends on who wrote it.

# Unambiguous. No dataset uses "#REF!" or "?" as an answer, so these are
# converted wherever they appear.
UNAMBIGUOUS_SENTINELS: frozenset = frozenset({
    "",
    "-",
    "--",
    "---",
    "?",
    "??",
    ".",
    "n/a",
    "n.a.",
    "n.a",
    "nan",
    "#error",
    "#value!",
    "#ref!",
    "#n/a",
    "#na",
    "#null!",
    "#div/0!",
    "<na>",
    "<null>",
})

# Ambiguous. Each of these is a real answer somewhere: "None" in an allergy
# column means the patient has no allergies, "Unknown" is an option on every
# census form, "NA" is Namibia, and Null is a surname. Each is also what an
# exporter writes when it has nothing -- a real cafe-sales export carried 1,931
# of them across its item, payment and location columns.
#
# They are only treated as missing when their capitalisation betrays them: an
# ALL-CAPS token inside a column that is otherwise Title or lower case was
# written by a machine, not chosen by a person. See
# :func:`_ambiguous_sentinel_mask`.
AMBIGUOUS_SENTINELS: frozenset = frozenset({
    "na",
    "null",
    "none",
    "nil",
    "missing",
    "unknown",
    "undefined",
    "not available",
    "not applicable",
    "not specified",
    "not provided",
    "error",
})

# Every token the cleaner knows about, for callers that want the whole list.
MISSING_SENTINELS: frozenset = UNAMBIGUOUS_SENTINELS | AMBIGUOUS_SENTINELS

# Currency symbols and grouping characters stripped before numeric coercion.
_CURRENCY_CHARS = "$€£₹¥₩₽"
_NUMERIC_STRIP_RE = re.compile(rf"[{re.escape(_CURRENCY_CHARS)},\s]")

# A value is only treated as numeric if what remains after stripping looks like
# a number: optional sign, digits, optional decimal part, optional exponent.
_NUMERIC_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

# Accounting-style negatives: "(1,234.50)" means -1234.50
_PARENS_NEGATIVE_RE = re.compile(r"^\((.*)\)$")

# Leading zeros carry meaning (zip codes, account numbers, product codes), so a
# column containing them is never coerced to a number.
_LEADING_ZERO_RE = re.compile(r"^0\d+")

# Date-ish shapes: contains a separator typical of dates or times.
_DATE_HINT_RE = re.compile(r"[-/:]|\d{4}")
_DATE_NAME_RE = re.compile(
    r"(^|_)(date|datetime|timestamp|time|day|month|year|dob|created|updated|modified)($|_)",
    re.IGNORECASE,
)

# Fraction of non-null values that must parse for a column-wide type coercion.
_COERCION_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class CleaningAction:
    """A single structural change applied to the DataFrame at load time.

    Attributes
    ----------
    step : str
        Machine-readable action name (e.g. ``"coerce_numeric"``).
    column : str | None
        Affected column, or ``None`` for whole-frame actions.
    detail : str
        Human-readable description shown in the UI and reports.
    count : int
        Number of cells, rows or columns affected.
    """

    step: str
    column: Optional[str]
    detail: str
    count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return the action as a plain dictionary."""
        return {
            "step": self.step,
            "column": self.column,
            "detail": self.detail,
            "count": self.count,
        }


@dataclass
class CleaningReport:
    """Collected :class:`CleaningAction` records from a cleaning run."""

    actions: List[CleaningAction] = field(default_factory=list)

    def add(self, step: str, column: Optional[str], detail: str, count: int = 0) -> None:
        """Record an action, ignoring no-ops (``count == 0``) for whole-frame steps."""
        self.actions.append(CleaningAction(step=step, column=column, detail=detail, count=count))

    @property
    def is_empty(self) -> bool:
        """True when nothing needed changing."""
        return not self.actions

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Return all actions as a list of dictionaries."""
        return [a.to_dict() for a in self.actions]

    def to_frame(self) -> pd.DataFrame:
        """Return all actions as a DataFrame for display."""
        if not self.actions:
            return pd.DataFrame(columns=["step", "column", "detail", "count"])
        return pd.DataFrame(self.to_dicts())

    def summary(self) -> str:
        """Return a one-line summary of the cleaning run."""
        if not self.actions:
            return "No structural cleaning was required."
        return f"{len(self.actions)} cleaning action(s) applied during load."


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------


def normalize_headers(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Strip whitespace and BOM characters from column names.

    A header read as ``" age "`` will not match ``"age"`` anywhere else in the
    platform, so this runs before any column is referenced by name.
    """
    renames: Dict[Any, str] = {}
    for col in df.columns:
        if not isinstance(col, str):
            continue
        cleaned = col.replace("﻿", "").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned != col:
            renames[col] = cleaned

    if renames:
        df = df.rename(columns=renames)
        report.add(
            "normalize_headers",
            None,
            f"Trimmed whitespace/BOM from {len(renames)} column name(s): "
            + ", ".join(f"'{k}' -> '{v}'" for k, v in list(renames.items())[:5]),
            len(renames),
        )
    return df


def deduplicate_columns(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Rename duplicate column labels so every column is addressable.

    Pandas allows two columns to share a name, but ``df[name]`` then returns a
    DataFrame instead of a Series, which breaks per-column repair.
    """
    if not df.columns.duplicated().any():
        return df

    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    renamed = 0
    for col in df.columns:
        key = str(col)
        if key in seen:
            seen[key] += 1
            new_cols.append(f"{key}_{seen[key]}")
            renamed += 1
        else:
            seen[key] = 0
            new_cols.append(key)

    df = df.copy()
    df.columns = new_cols
    report.add(
        "deduplicate_columns",
        None,
        f"Renamed {renamed} duplicate column label(s) to keep every column addressable.",
        renamed,
    )
    return df


def drop_unnamed_index(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Drop the phantom ``Unnamed: 0`` column left by a previous ``to_csv``.

    Only dropped when it actually behaves like a saved index: unique, complete
    and integer-valued. A genuine data column called ``Unnamed: 0`` survives.
    """
    candidates = [
        c for c in df.columns
        if isinstance(c, str) and re.fullmatch(r"Unnamed:\s*\d+", c.strip())
    ]
    dropped: List[str] = []
    for col in candidates:
        series = df[col]
        if series.isna().all():
            dropped.append(col)
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        looks_like_index = (
            numeric.notna().all()
            and numeric.is_unique
            and float(numeric.min()) in (0.0, 1.0)
            and (numeric % 1 == 0).all()
        )
        if looks_like_index:
            dropped.append(col)

    if dropped:
        df = df.drop(columns=dropped)
        report.add(
            "drop_unnamed_index",
            None,
            f"Dropped leftover index column(s) from a previous export: {', '.join(dropped)}.",
            len(dropped),
        )
    return df


def _ambiguous_sentinel_mask(
    series: pd.Series, as_str: pd.Series, lowered: pd.Series
) -> Tuple[pd.Series, int]:
    """Decide which ambiguous tokens in a column are placeholders.

    The test is capitalisation. ``UNKNOWN`` shouting inside a column of Title
    Case items was written by an exporter; ``Unknown`` sitting quietly among
    its neighbours was chosen by whoever designed the form, and is a real
    answer.

    Only the shouting case is converted. A lowercase ``none`` inside a Title
    Case column is *also* inconsistent, but it reads far more like a hurried
    human entry than a machine's placeholder, and the cost of being wrong is
    asymmetric: preserving a value that turns out to be junk is a nuisance,
    deleting one that turns out to be real is unrecoverable. Anything this
    misses can be forced through ``force_sentinels``.

    When the column has no convention to contrast against, or is written
    entirely in capitals anyway, nothing is converted.

    Returns
    -------
    tuple[pd.Series, int]
        The conversion mask, and the count of occurrences deliberately kept.
    """
    candidates = lowered.isin(AMBIGUOUS_SENTINELS) & series.notna()
    if not candidates.any():
        return candidates, 0

    convention = dominant_casing(series, ignore=AMBIGUOUS_SENTINELS)
    if convention is None or convention == "upper":
        # Nothing to contrast against: keep every occurrence as a real value.
        return pd.Series(False, index=series.index), int(candidates.sum())

    shouting = as_str.map(lambda v: casing_class(v) == "upper")
    mask = candidates & shouting
    return mask, int((candidates & ~shouting).sum())


def normalize_missing_sentinels(
    df: pd.DataFrame,
    report: CleaningReport,
    force_sentinels: Optional[Iterable[str]] = None,
    preserve_tokens: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Convert disguised missing markers into real ``NaN`` values.

    This is the step that makes completeness metrics honest: before it, a
    column full of ``"?"`` reports 0% missing.

    Two guards keep it from eating real words. Ambiguous tokens are judged on
    capitalisation rather than converted on sight, and free-text columns are
    exempted from that judgement entirely -- a review that happens to read
    "unknown" is a review, not a gap.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to clean.
    report : CleaningReport
        Collects both the conversions made and the values deliberately kept.
    force_sentinels : Iterable[str] | None
        Tokens to convert wherever they appear, whatever their capitalisation
        and whatever kind of column they are in. Use when the heuristic was too
        cautious for a particular file.
    preserve_tokens : Iterable[str] | None
        Tokens to never convert. Use when a value the cleaner would strip is a
        real category in this file.

    Notes
    -----
    ``preserve_tokens`` wins over ``force_sentinels``, so an explicit
    instruction to keep data is never overridden by an instruction to delete it.
    """
    forced = {t.strip().lower() for t in (force_sentinels or ())}
    preserved = {t.strip().lower() for t in (preserve_tokens or ())}
    forced -= preserved

    df = df.copy()
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue

        as_str = series.astype(str).str.strip()
        lowered = as_str.str.lower()

        mask = lowered.isin(UNAMBIGUOUS_SENTINELS) & series.notna()
        kept = 0

        if not is_free_text(series):
            ambiguous, kept = _ambiguous_sentinel_mask(series, as_str, lowered)
            mask = mask | ambiguous

        if forced:
            mask = mask | (lowered.isin(forced) & series.notna())
        if preserved:
            mask = mask & ~lowered.isin(preserved)

        hits = int(mask.sum())
        if hits:
            # np.nan, not pd.NA: scikit-learn's imputers test object arrays
            # with `X != X`, and pd.NA raises "boolean value of NA is
            # ambiguous" there. Both read as null to pandas; only one survives
            # the trip through a model pipeline.
            df.loc[mask, col] = np.nan
            report.add(
                "normalize_missing_sentinels",
                str(col),
                f"Converted {hits} disguised missing value(s) to null in '{col}'.",
                hits,
            )
        if kept:
            # Surfaced, not silent: the user can see the call that was made and
            # disagree with it.
            report.add(
                "preserve_ambiguous_value",
                str(col),
                f"Kept {kept} value(s) such as 'None'/'Unknown' in '{col}' as real "
                f"categories -- their capitalisation matches the column.",
                kept,
            )
    return df


def strip_text_whitespace(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Trim and collapse whitespace inside text cells.

    ``"Delhi "`` and ``"Delhi"`` are the same category; leaving both inflates
    cardinality and splits value counts.
    """
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue

        notna = series.notna()
        if not notna.any():
            continue

        original = series[notna].astype(str)
        cleaned = original.str.strip().str.replace(r"\s+", " ", regex=True)
        changed = int((original != cleaned).sum())
        if changed:
            df.loc[notna, col] = cleaned
            report.add(
                "strip_whitespace",
                str(col),
                f"Trimmed surrounding/repeated whitespace in {changed} cell(s) of '{col}'.",
                changed,
            )
    return df


def _parse_numeric_token(value: str) -> Optional[float]:
    """Parse a single cell into a float, handling currency, grouping and parens.

    Returns ``None`` when the token is not numeric.
    """
    token = value.strip()
    if not token:
        return None

    negative = False
    parens = _PARENS_NEGATIVE_RE.match(token)
    if parens:
        negative = True
        token = parens.group(1)

    percent = token.endswith("%")
    if percent:
        token = token[:-1]

    token = _NUMERIC_STRIP_RE.sub("", token)
    if not token or not _NUMERIC_RE.match(token):
        return None

    result = float(token)
    if percent:
        result /= 100.0
    if negative:
        result = -result
    return result


def coerce_numeric_like(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Convert text columns that hold numbers into real numeric columns.

    Handles thousands separators, currency symbols, percentages and
    accounting-style negatives. A column is only converted when at least 90% of
    its non-null values parse, so a genuinely textual column with a stray
    number in it is left alone.

    Columns whose values carry leading zeros (zip codes, account numbers) are
    never converted -- the zeros are meaningful.
    """
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue

        notna_mask = series.notna()
        non_null = series[notna_mask]
        if non_null.empty:
            continue

        as_str = non_null.astype(str).str.strip()

        # Leading zeros are significant -- do not touch this column.
        if as_str.str.match(_LEADING_ZERO_RE).any():
            continue

        parsed = as_str.map(_parse_numeric_token)
        parse_rate = float(parsed.notna().mean())
        if parse_rate < _COERCION_THRESHOLD:
            continue

        was_percent = bool(as_str.str.endswith("%").all())

        numeric = pd.Series(pd.NA, index=df.index, dtype="object")
        numeric[notna_mask] = parsed
        converted = pd.to_numeric(numeric, errors="coerce")

        # Preserve integer-ness: an int column must not become 32.5 later.
        non_null_converted = converted.dropna()
        is_integral = (
            not non_null_converted.empty
            and bool((non_null_converted % 1 == 0).all())
            and not was_percent
        )
        if is_integral:
            converted = converted.astype("Int64")

        unparsed = int((parsed.isna()).sum())
        df[col] = converted

        detail = f"Converted '{col}' from text to {'integer' if is_integral else 'numeric'}"
        if was_percent:
            detail += " (percentages scaled to fractions)"
        if unparsed:
            detail += f"; {unparsed} unparseable value(s) became null"
        report.add("coerce_numeric", str(col), detail + ".", int(notna_mask.sum()))

    return df


def coerce_datetime_like(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Convert text columns that hold dates into real datetime columns.

    Conservative by design: a column is only parsed when its name looks
    temporal *or* its values carry date-like separators, and at least 90% of
    values parse. This avoids pandas turning a column of small integers into
    dates in the 1970s.
    """
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        as_str = non_null.astype(str).str.strip()
        name_hint = bool(_DATE_NAME_RE.search(str(col)))
        value_hint = bool(as_str.str.contains(_DATE_HINT_RE, regex=True).mean() >= _COERCION_THRESHOLD)
        if not (name_hint or value_hint):
            continue

        # Pure integers reaching this point are far more likely to be codes.
        if as_str.str.fullmatch(r"\d+").mean() >= _COERCION_THRESHOLD and not name_hint:
            continue

        try:
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed", dayfirst=False)
        except (ValueError, TypeError):
            continue

        if float(parsed.notna().mean()) < _COERCION_THRESHOLD:
            continue

        converted = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)
        df[col] = converted
        report.add(
            "coerce_datetime",
            str(col),
            f"Parsed '{col}' from text into datetime values.",
            int(converted.notna().sum()),
        )

    return df


def downcast_integral_floats(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Restore integer typing to columns that pandas floated because of nulls.

    ``pd.read_csv`` has no nullable integer, so an ``age`` column containing a
    single blank comes back as ``float64``. Left alone, mean-imputing it
    produces ``32.5`` in a column of whole numbers. A float column that has
    nulls *and* whose observed values are all integral is therefore restored to
    the nullable ``Int64`` type.

    A float column with no nulls is left untouched: pandas would already have
    typed it as an integer if the file contained only whole numbers, so its
    float dtype is genuine.
    """
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_float_dtype(series.dtype):
            continue
        if not series.isna().any():
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue
        if not bool((non_null % 1 == 0).all()):
            continue
        # Values beyond Int64's range cannot be represented exactly.
        if not bool(non_null.abs().lt(2 ** 63 - 1).all()):
            continue

        df[col] = series.astype("Int64")
        report.add(
            "restore_integer_type",
            str(col),
            f"Restored '{col}' to integer type (pandas floated it only because of missing values).",
            int(series.isna().sum()),
        )
    return df


def drop_empty_rows_and_columns(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove rows and columns that are entirely null.

    Fully empty columns cannot be imputed -- there is no observed value to
    impute from -- so they are dropped rather than filled with a placeholder.
    """
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        report.add(
            "drop_empty_columns",
            None,
            f"Dropped {len(empty_cols)} fully empty column(s) (nothing to impute from): "
            + ", ".join(str(c) for c in empty_cols[:5]),
            len(empty_cols),
        )

    if df.empty:
        return df

    empty_row_mask = df.isna().all(axis=1)
    empty_rows = int(empty_row_mask.sum())
    if empty_rows:
        df = df.loc[~empty_row_mask].reset_index(drop=True)
        report.add(
            "drop_empty_rows",
            None,
            f"Dropped {empty_rows} fully empty row(s).",
            empty_rows,
        )

    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def clean_dataframe(
    df: pd.DataFrame,
    *,
    normalize_sentinels: bool = True,
    strip_whitespace: bool = True,
    coerce_numeric: bool = True,
    coerce_datetime: bool = True,
    drop_empty: bool = True,
    force_sentinels: Optional[Iterable[str]] = None,
    preserve_tokens: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Apply the full structural cleaning sequence to a freshly loaded frame.

    Order matters: headers are fixed first so columns can be addressed, then
    sentinels become nulls so they are not mistaken for data during type
    coercion, and empty columns are dropped last -- a column of nothing but
    ``"?"`` only becomes fully empty after sentinel normalisation.

    Parameters
    ----------
    df : pd.DataFrame
        Freshly loaded DataFrame.
    normalize_sentinels, strip_whitespace, coerce_numeric, coerce_datetime, drop_empty : bool
        Toggles for each stage, all enabled by default.
    force_sentinels : Iterable[str] | None
        Extra tokens to treat as missing unconditionally, overriding the
        capitalisation heuristic.
    preserve_tokens : Iterable[str] | None
        Tokens to keep as real values even when the heuristic would strip them.

    Returns
    -------
    tuple[pd.DataFrame, CleaningReport]
        The cleaned frame and a record of everything that changed.
    """
    report = CleaningReport()
    if df.empty and len(df.columns) == 0:
        return df, report

    df = normalize_headers(df, report)
    df = deduplicate_columns(df, report)
    df = drop_unnamed_index(df, report)

    if normalize_sentinels:
        df = normalize_missing_sentinels(
            df, report,
            force_sentinels=force_sentinels,
            preserve_tokens=preserve_tokens,
        )
    if strip_whitespace:
        df = strip_text_whitespace(df, report)
    if coerce_numeric:
        df = coerce_numeric_like(df, report)
    if coerce_datetime:
        df = coerce_datetime_like(df, report)
    if coerce_numeric:
        df = downcast_integral_floats(df, report)
    if drop_empty:
        df = drop_empty_rows_and_columns(df, report)

    if not report.is_empty:
        logger.info("Structural cleaning applied %d action(s).", len(report.actions))
    return df, report


__all__ = [
    "AMBIGUOUS_SENTINELS",
    "UNAMBIGUOUS_SENTINELS",
    "MISSING_SENTINELS",
    "CleaningAction",
    "CleaningReport",
    "clean_dataframe",
    "coerce_datetime_like",
    "coerce_numeric_like",
    "deduplicate_columns",
    "downcast_integral_floats",
    "drop_empty_rows_and_columns",
    "drop_unnamed_index",
    "normalize_headers",
    "normalize_missing_sentinels",
    "strip_text_whitespace",
]
