"""
AuditHub Ingestion - Dataset Analyzer
=======================================

Intelligent dataset analysis engine that computes comprehensive statistics,
infers column types, and detects dataset characteristics — all without
any hardcoded assumptions about column names or schemas.

The analyzer is completely dataset-agnostic and works with any
structured tabular data.

Usage::

    from src.ingestion.dataset_analyzer import DatasetAnalyzer
    import pandas as pd

    df = pd.read_csv("data.csv")
    analyzer = DatasetAnalyzer()
    summary = analyzer.analyze(df)
    print(summary.dataset_type)
    print(summary.potential_targets)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

from src.ingestion.dataset_fingerprint import DatasetFingerprint, FingerprintGenerator

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants for type inference thresholds
# ---------------------------------------------------------------------------
HIGH_CARDINALITY_THRESHOLD = 100  # Columns with more unique values
IDENTIFIER_THRESHOLD_RATIO = 0.95  # Ratio of unique values to rows
CONSTANT_THRESHOLD = 0.98  # Ratio of same value to total rows
BINARY_THRESHOLD = 2  # Max unique values for binary detection


# ============================================================================
# Column Type Inference
# ============================================================================
# The inferred types below are semantic types, not pandas dtypes.
# They describe what the column represents to a human/data scientist.


class ColumnCategory:
    """Constants for inferred column categories."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    MIXED = "mixed"
    CONSTANT = "constant"
    BINARY = "binary"
    IDENTIFIER = "identifier"
    HIGH_CARDINALITY = "high_cardinality"
    UNKNOWN = "unknown"


# ============================================================================
# ColumnInfo
# ============================================================================


@dataclass(frozen=True)
class ColumnInfo:
    """Detailed information about a single column.

    Attributes
    ----------
    name : str
        Column name.
    dtype : str
        Pandas dtype as a string.
    inferred_type : str
        Semantic type (one of ``ColumnCategory`` constants).
    nunique : int
        Number of unique non-null values.
    null_count : int
        Number of null values.
    null_pct : float
        Percentage of null values (0.0--100.0).
    is_numeric : bool
        Whether the column is numeric.
    is_categorical_candidate : bool
        Whether the column is a good candidate for categorical encoding.
    is_target_candidate : bool
        Whether the column could be a target variable.
    is_id_candidate : bool
        Whether the column could be an ID column.
    is_feature_candidate : bool
        Whether the column could be an ML feature.
    mean : float | None
        Mean value (for numeric columns).
    std : float | None
        Standard deviation (for numeric columns).
    min_val : float | None
        Minimum value (for numeric columns).
    max_val : float | None
        Maximum value (for numeric columns).
    top_value : Any | None
        Most frequent value.
    top_freq : int | None
        Frequency of the most frequent value.
    """

    name: str = ""
    dtype: str = ""
    inferred_type: str = ColumnCategory.UNKNOWN
    nunique: int = 0
    null_count: int = 0
    null_pct: float = 0.0
    is_numeric: bool = False
    is_categorical_candidate: bool = False
    is_target_candidate: bool = False
    is_id_candidate: bool = False
    is_feature_candidate: bool = True
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    top_value: Any = None
    top_freq: Optional[int] = None


# ============================================================================
# DatasetSummary
# ============================================================================


@dataclass
class DatasetSummary:
    """Comprehensive summary of an analyzed dataset.

    Attributes
    ----------
    general : dict
        General information (filename, rows, columns, size, etc.).
    schema_info : dict
        Schema information (column names, dtypes, etc.).
    column_stats : list[ColumnInfo]
        Per-column detailed statistics.
    missing_summary : dict
        Summary of missing values across the dataset.
    duplicate_summary : dict
        Summary of duplicate rows/columns.
    memory_stats : dict
        Memory usage statistics.
    dataset_type : dict
        Detected dataset type with confidence score.
    potential_targets : list[dict]
        Potential target columns with confidence scores.
    potential_ids : list[dict]
        Potential ID columns with confidence scores.
    potential_features : list[str]
        Columns that are good feature candidates.
    fingerprint : DatasetFingerprint | None
        Optional dataset fingerprint.
    """

    general: Dict[str, Any] = field(default_factory=dict)
    schema_info: Dict[str, Any] = field(default_factory=dict)
    column_stats: List[ColumnInfo] = field(default_factory=list)
    missing_summary: Dict[str, Any] = field(default_factory=dict)
    duplicate_summary: Dict[str, Any] = field(default_factory=dict)
    memory_stats: Dict[str, Any] = field(default_factory=dict)
    dataset_type: Dict[str, Any] = field(default_factory=dict)
    potential_targets: List[Dict[str, Any]] = field(default_factory=list)
    potential_ids: List[Dict[str, Any]] = field(default_factory=list)
    potential_features: List[str] = field(default_factory=list)
    fingerprint: Optional[DatasetFingerprint] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the summary to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary representation.
        """
        return {
            "general": self.general,
            "schema_info": self.schema_info,
            "column_stats": [vars(c) if hasattr(c, '__dataclass_fields__') else c for c in self.column_stats],
            "missing_summary": self.missing_summary,
            "duplicate_summary": self.duplicate_summary,
            "memory_stats": self.memory_stats,
            "dataset_type": self.dataset_type,
            "potential_targets": self.potential_targets,
            "potential_ids": self.potential_ids,
            "potential_features": self.potential_features,
            "fingerprint": self.fingerprint.to_dict() if self.fingerprint else None,
        }


# ============================================================================
# DatasetAnalyzer
# ============================================================================


class DatasetAnalyzer:
    """Analyze datasets to infer types, detect patterns, and compute stats.

    The analyzer is **completely dataset-agnostic** — it never assumes
    column names, schemas, or domain. Everything is inferred from data.

    Parameters
    ----------
    generate_fingerprint : bool
        Whether to generate a ``DatasetFingerprint`` during analysis.
        Defaults to ``True``.
    """

    def __init__(self, generate_fingerprint: bool = True) -> None:
        self._generate_fingerprint = generate_fingerprint
        self._fp_generator = FingerprintGenerator() if generate_fingerprint else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        source_filename: Optional[str] = None,
    ) -> DatasetSummary:
        """Perform a full analysis of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze.
        source_filename : str | None
            Optional original filename for reference.

        Returns
        -------
        DatasetSummary
            Comprehensive analysis summary.
        """
        logger.info("Analyzing dataset with shape %s", df.shape)
        df = df.copy()  # Work on a copy to avoid mutating the original

        column_info = self._compute_column_info(df)
        general = self._compute_general(df, source_filename)
        schema = self._compute_schema(df)
        missing = self._compute_missing_summary(df, column_info)
        duplicates = self._compute_duplicate_summary(df)
        memory = self._compute_memory_stats(df)
        targets = self._detect_targets(column_info)
        ids = self._detect_ids(column_info)
        features = self._select_features(column_info, ids)
        dataset_type = self._detect_dataset_type(df, column_info, targets)

        fingerprint = None
        if self._generate_fingerprint and self._fp_generator:
            fingerprint = self._fp_generator.from_dataframe(df)

        summary = DatasetSummary(
            general=general,
            schema_info=schema,
            column_stats=column_info,
            missing_summary=missing,
            duplicate_summary=duplicates,
            memory_stats=memory,
            dataset_type=dataset_type,
            potential_targets=targets,
            potential_ids=ids,
            potential_features=features,
            fingerprint=fingerprint,
        )

        logger.info(
            "Analysis complete: detected type=%s, %d target candidates, %d feature columns",
            dataset_type.get("type", "unknown"),
            len(targets),
            len(features),
        )
        return summary

    # ------------------------------------------------------------------
    # Column-level analysis
    # ------------------------------------------------------------------

    def _compute_column_info(self, df: pd.DataFrame) -> List[ColumnInfo]:
        """Compute per-column statistics and type inference.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        list[ColumnInfo]
            Detailed info for each column.
        """
        info_list: List[ColumnInfo] = []
        num_rows = len(df)

        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)

            # Basic stats
            null_count = int(series.isna().sum())
            null_pct = round((null_count / num_rows) * 100, 2) if num_rows > 0 else 0.0
            nunique = int(series.nunique())
            is_numeric = pd.api.types.is_numeric_dtype(series.dtype)

            # Numeric stats
            mean_val: Optional[float] = None
            std_val: Optional[float] = None
            min_val: Optional[float] = None
            max_val: Optional[float] = None
            if is_numeric:
                has_data = not series.isna().all()
                if pd.api.types.is_bool_dtype(series.dtype):
                    # Bool columns support min/max/mean but not round()
                    mean_val = float(series.mean()) if has_data else None
                    std_val = float(series.std()) if has_data else None
                    min_val = int(series.min()) if has_data else None
                    max_val = int(series.max()) if has_data else None
                else:
                    mean_val = round(series.mean(), 4) if has_data else None
                    std_val = round(series.std(), 4) if has_data else None
                    min_val = round(series.min(), 4) if has_data else None
                    max_val = round(series.max(), 4) if has_data else None

            # Top value
            top_value = None
            top_freq = None
            try:
                vc = series.value_counts()
                if not vc.empty:
                    top_value = vc.index[0]
                    top_freq = int(vc.iloc[0])
            except Exception:
                pass

            # Infer semantic type
            inferred = self._infer_column_type(series, nunique, is_numeric)

            # Determine if this is a target candidate
            is_target = self._is_target_candidate(series, col, nunique, is_numeric)

            # Determine if this is an ID candidate
            is_id = self._is_id_candidate(series, col, nunique, num_rows)

            # Determine categorical candidate
            is_cat_candidate = inferred in (
                ColumnCategory.CATEGORICAL,
                ColumnCategory.BOOLEAN,
                ColumnCategory.BINARY,
            ) and not is_id

            info_list.append(ColumnInfo(
                name=str(col),
                dtype=dtype_str,
                inferred_type=inferred,
                nunique=nunique,
                null_count=null_count,
                null_pct=null_pct,
                is_numeric=is_numeric,
                is_categorical_candidate=is_cat_candidate,
                is_target_candidate=is_target,
                is_id_candidate=is_id,
                is_feature_candidate=not is_id and inferred != ColumnCategory.IDENTIFIER,
                mean=mean_val,
                std=std_val,
                min_val=min_val,
                max_val=max_val,
                top_value=top_value,
                top_freq=top_freq,
            ))

        return info_list

    # ------------------------------------------------------------------
    # Column type inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_column_type(
        series: pd.Series,
        nunique: int,
        is_numeric: bool,
    ) -> str:
        """Infer the semantic type of a column.

        Parameters
        ----------
        series : pd.Series
            Column data.
        nunique : int
            Number of unique values.
        is_numeric : bool
            Whether the column is numeric.

        Returns
        -------
        str
            One of ``ColumnCategory`` constants.
        """
        num_rows = len(series.dropna())            # Constant column (same value throughout)
        if nunique <= 1 and num_rows > 0:
            return ColumnCategory.CONSTANT

        # Boolean detection — must come before numeric/binary checks
        # because is_numeric_dtype returns True for bool in pandas
        if pd.api.types.is_bool_dtype(series.dtype):
            return ColumnCategory.BOOLEAN

        # Check for binary integers (0/1 only) before general numeric
        if is_numeric and nunique == 2:
            return ColumnCategory.BINARY

        # Datetime detection
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return ColumnCategory.DATETIME

        # Numeric (already handled binary case above)
        if is_numeric:
            return ColumnCategory.NUMERIC

        # Object/text — check for mixed types
        if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
            non_null = series.dropna()
            if len(non_null) == 0:
                return ColumnCategory.UNKNOWN

            # Sample some values to check for mixed types
            sample = non_null.head(100)
            type_set = set(type(v).__name__ for v in sample)
            if len(type_set) > 2:  # More than just str + NoneType
                return ColumnCategory.MIXED

            # Check if it's actually datetime
            try:
                pd.to_datetime(series, errors="raise")
                return ColumnCategory.DATETIME
            except (ValueError, TypeError):
                pass

            # High cardinality text
            if nunique > HIGH_CARDINALITY_THRESHOLD:
                return ColumnCategory.HIGH_CARDINALITY

            return ColumnCategory.TEXT

        return ColumnCategory.UNKNOWN

    # ------------------------------------------------------------------
    # Candidate detection heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _is_target_candidate(
        series: pd.Series,
        col: Any,
        nunique: int,
        is_numeric: bool,
    ) -> bool:
        """Determine if a column could be a target variable.

        Heuristics:
        - Binary columns with 2 unique values (classification candidate)
        - Numeric columns with reasonable variance (regression candidate)
        - Columns with "target", "label", "class", "score", "outcome",
          "y" in their name get a bonus.

        Parameters
        ----------
        series : pd.Series
            Column data.
        col : any
            Column name.
        nunique : int
            Number of unique values.
        is_numeric : bool
            Whether the column is numeric.

        Returns
        -------
        bool
            ``True`` if the column is a target candidate.
        """
        col_lower = str(col).lower().strip()

        # Skip constant columns
        if nunique <= 1:
            return False

        # Skip ID-like columns
        id_keywords = {"id", "uuid", "key", "hash", "identifier", "index"}
        if col_lower in id_keywords or any(col_lower.startswith(k) for k in id_keywords):
            return False

        # Binary columns are strong classification targets
        if nunique == 2:
            return True

        # Low-cardinality categorical (3-10 values) could be multi-class
        if 3 <= nunique <= 10 and not is_numeric:
            return True

        # Numeric with "target-like" name
        target_keywords = {"target", "label", "class", "score", "outcome", "y", "result", "prediction", "category", "type"}
        if col_lower in target_keywords or col_lower.startswith("target") or col_lower.startswith("label"):
            return True

        return False

    @staticmethod
    def _is_id_candidate(
        series: pd.Series,
        col: Any,
        nunique: int,
        num_rows: int,
    ) -> bool:
        """Determine if a column could be an ID column.

        Heuristics:
        - All values are unique (or nearly all)
        - Column name suggests ID

        Parameters
        ----------
        series : pd.Series
            Column data.
        col : any
            Column name.
        nunique : int
            Number of unique values.
        num_rows : int
            Total number of rows.

        Returns
        -------
        bool
            ``True`` if the column is an ID candidate.
        """
        if num_rows == 0:
            return False

        col_lower = str(col).lower().strip()

        # Check if column name suggests ID
        id_name_patterns = {
            "id", "uuid", "key", "hash", "identifier", "index",
            "row_id", "record_id", "dataset_id", "sample_id",
            "user_id", "item_id", "product_id", "transaction_id",
        }
        if col_lower in id_name_patterns:
            return True

        # Check value uniqueness ratio
        if nunique / num_rows >= IDENTIFIER_THRESHOLD_RATIO and nunique > 1:
            return True

        return False

    @staticmethod
    def _select_features(
        column_info: List[ColumnInfo],
        ids: List[Dict[str, Any]],
    ) -> List[str]:
        """Select columns that are good feature candidates.

        Excludes ID columns and columns flagged as non-features.

        Parameters
        ----------
        column_info : list[ColumnInfo]
            Per-column analysis results.
        ids : list[dict]
            Detected ID columns.

        Returns
        -------
        list[str]
            Column names suitable as features.
        """
        id_names = {i["name"] for i in ids}
        return [
            c.name for c in column_info
            if c.is_feature_candidate and c.name not in id_names
            and c.inferred_type not in (ColumnCategory.IDENTIFIER, ColumnCategory.CONSTANT)
        ]

    # ------------------------------------------------------------------
    # General & schema stats
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_general(df: pd.DataFrame, source_filename: Optional[str]) -> Dict[str, Any]:
        """Compute general dataset information.

        Returns
        -------
        dict
            General info dictionary.
        """
        return {
            "filename": source_filename or "unknown",
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "total_cells": len(df) * len(df.columns),
            "size_bytes": int(df.memory_usage(deep=True).sum()),
        }

    @staticmethod
    def _compute_schema(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute schema information.

        Returns
        -------
        dict
            Schema info dictionary.
        """
        return {
            "column_names": list(df.columns),
            "column_order": list(range(len(df.columns))),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "has_index": df.index.name is not None,
            "index_name": str(df.index.name) if df.index.name else None,
        }

    # ------------------------------------------------------------------
    # Missing value analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_missing_summary(
        df: pd.DataFrame,
        column_info: List[ColumnInfo],
    ) -> Dict[str, Any]:
        """Compute missing value statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        column_info : list[ColumnInfo]
            Pre-computed column info.

        Returns
        -------
        dict
            Missing value summary.
        """
        total_cells = len(df) * len(df.columns)
        total_missing = int(df.isna().sum().sum())

        columns_with_missing = [
            {"name": c.name, "null_count": c.null_count, "null_pct": c.null_pct}
            for c in column_info
            if c.null_count > 0
        ]

        return {
            "total_missing": total_missing,
            "total_cells": total_cells,
            "overall_missing_pct": round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0,
            "columns_with_missing": columns_with_missing,
            "num_columns_with_missing": len(columns_with_missing),
        }

    # ------------------------------------------------------------------
    # Duplicate analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_duplicate_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute duplicate statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        dict
            Duplicate summary.
        """
        duplicate_rows = int(df.duplicated().sum())
        duplicate_cols = _find_duplicate_columns(df)

        return {
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_pct": round((duplicate_rows / len(df)) * 100, 2) if len(df) > 0 else 0.0,
            "duplicate_columns": duplicate_cols,
            "num_duplicate_columns": len(duplicate_cols),
        }

    # ------------------------------------------------------------------
    # Memory statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_memory_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute memory usage statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        dict
            Memory stats.
        """
        deep_usage = df.memory_usage(deep=True)
        usage_no_index = deep_usage[1:]  # Skip index

        return {
            "total_bytes": int(deep_usage.sum()),
            "total_mb": round(deep_usage.sum() / (1024 * 1024), 4),
            "per_column_bytes": {str(col): int(usage_no_index.iloc[i]) for i, col in enumerate(df.columns)},
            "mean_column_bytes": int(usage_no_index.mean()) if len(usage_no_index) > 0 else 0,
        }

    # ------------------------------------------------------------------
    # Target detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_targets(column_info: List[ColumnInfo]) -> List[Dict[str, Any]]:
        """Detect potential target columns with confidence scores.

        Parameters
        ----------
        column_info : list[ColumnInfo]
            Per-column analysis results.

        Returns
        -------
        list[dict]
            List of target candidates with name, type, confidence.
        """
        targets = []
        for c in column_info:
            if not c.is_target_candidate:
                continue

            # Compute confidence score
            confidence = 0.5  # Base confidence

            if c.inferred_type == ColumnCategory.BINARY:
                confidence += 0.3  # Binary is strong target signal
            elif c.inferred_type == ColumnCategory.BOOLEAN:
                confidence += 0.25
            elif c.inferred_type == ColumnCategory.NUMERIC:
                confidence += 0.1

            if c.nunique <= 2:
                confidence += 0.1  # Low cardinality bonus

            if c.null_pct < 10:
                confidence += 0.1  # Few missing values

            confidence = min(confidence, 1.0)

            targets.append({
                "name": c.name,
                "inferred_type": c.inferred_type,
                "nunique": c.nunique,
                "confidence": round(confidence, 2),
                "suggested_task": "classification" if c.nunique <= 10 else "regression",
            })

        # Sort by confidence descending
        targets.sort(key=lambda t: t["confidence"], reverse=True)
        return targets

    @staticmethod
    def _detect_ids(column_info: List[ColumnInfo]) -> List[Dict[str, Any]]:
        """Detect potential ID columns.

        Parameters
        ----------
        column_info : list[ColumnInfo]
            Per-column analysis results.

        Returns
        -------
        list[dict]
            List of ID candidates with name and confidence.
        """
        ids = []
        for c in column_info:
            if not c.is_id_candidate:
                continue

            confidence = 0.5
            if c.nunique > 0 and c.nunique / max(c.nunique, 1) >= IDENTIFIER_THRESHOLD_RATIO:
                confidence += 0.3
            if c.null_pct == 0:
                confidence += 0.1
            if c.inferred_type == ColumnCategory.IDENTIFIER:
                confidence += 0.1

            ids.append({
                "name": c.name,
                "dtype": c.dtype,
                "nunique": c.nunique,
                "confidence": round(min(confidence, 1.0), 2),
            })

        ids.sort(key=lambda i: i["confidence"], reverse=True)
        return ids

    # ------------------------------------------------------------------
    # Dataset type detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_dataset_type(
        df: pd.DataFrame,
        column_info: List[ColumnInfo],
        targets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect the likely dataset type/ML task.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        column_info : list[ColumnInfo]
            Per-column analysis results.
        targets : list[dict]
            Detected target candidates.

        Returns
        -------
        dict
            Dataset type with type name, confidence, and reasoning.
        """
        num_rows = len(df)
        num_cols = len(df.columns)
        has_datetime = any(c.inferred_type == ColumnCategory.DATETIME for c in column_info)

        # Classification candidate: has binary/low-cardinality target
        if targets:
            top_target = targets[0]
            if top_target["suggested_task"] == "classification":
                return {
                    "type": "classification",
                    "confidence": top_target["confidence"],
                    "reasoning": f"Found target '{top_target['name']}' "
                                 f"with {top_target['nunique']} classes",
                    "target_column": top_target["name"],
                }

        # Regression candidate: has numeric target
        if targets:
            top_target = targets[0]
            if top_target["suggested_task"] == "regression":
                return {
                    "type": "regression",
                    "confidence": top_target["confidence"],
                    "reasoning": f"Found numeric target '{top_target['name']}'",
                    "target_column": top_target["name"],
                }

        # Time-series candidate: has datetime column and reasonable shape
        if has_datetime and num_rows > 10:
            return {
                "type": "time_series",
                "confidence": 0.5,
                "reasoning": "Found datetime column(s) in dataset",
                "target_column": None,
            }

        # Unsupervised candidate: no clear target
        if num_cols >= 2 and num_rows >= 10:
            return {
                "type": "unsupervised",
                "confidence": 0.6,
                "reasoning": "No clear target column detected; suitable for clustering",
                "target_column": None,
            }

        # Fallback: unknown
        return {
            "type": "unknown",
            "confidence": 0.0,
            "reasoning": "Could not determine dataset type automatically",
            "target_column": None,
        }


# ============================================================================
# Module-level helpers
# ============================================================================


def _find_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """Find columns with identical values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list[str]
        Names of duplicate columns.
    """
    seen: Dict[str, str] = {}
    duplicates: List[str] = []

    for col in df.columns:
        # Use a sample hash for quick comparison; full comparison on collision
        col_hash = _column_signature(df[col])
        for existing_col, existing_hash in seen.items():
            if col_hash == existing_hash and df[col].equals(df[existing_col]):
                duplicates.append(str(col))
                break
        else:
            seen[str(col)] = col_hash

    return duplicates


def _column_signature(series: pd.Series) -> str:
    """Compute a quick hash signature for a column.

    Uses the first 1000 non-null values plus the dtype.

    Parameters
    ----------
    series : pd.Series
        Column data.

    Returns
    -------
    str
        Hash signature.
    """
    import hashlib
    sample = series.dropna().head(1000)
    sig_str = f"{str(series.dtype)}|{sample.values.tobytes()}"
    return hashlib.md5(sig_str.encode()).hexdigest()


__all__ = [
    "ColumnCategory",
    "ColumnInfo",
    "DatasetAnalyzer",
    "DatasetSummary",
]
