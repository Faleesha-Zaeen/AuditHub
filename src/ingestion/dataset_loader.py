"""
AuditHub Ingestion - Dataset Loader
=====================================

Safe, format-agnostic dataset loader for CSV, Excel (.xlsx, .xls),
and JSON (tabular) files.

Automatically detects file format from extension, validates file
integrity, and returns a ``pandas.DataFrame`` along with
``DatasetMetadata``.

Usage::

    from src.ingestion.dataset_loader import DatasetLoader

    loader = DatasetLoader()
    df, metadata = loader.load("path/to/dataset.csv")
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from src.utils.dataset_metadata import DatasetMetadata
from src.utils.logger import get_logger

from src.ingestion.exceptions import (
    CorruptedDatasetError,
    EmptyDatasetError,
    EncodingError,
    InvalidJsonError,
    MissingColumnsError,
    UnsupportedFormatError,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Supported ingestion formats (CSV, Excel, JSON only per spec)
# ---------------------------------------------------------------------------
_SUPPORTED_EXTENSIONS = frozenset({".csv", ".xlsx", ".xls", ".json"})

# ---------------------------------------------------------------------------
# Default read kwargs per format (can be overridden via load())
# ---------------------------------------------------------------------------
_DEFAULT_CSV_KWARGS: Dict[str, Any] = {
    "encoding": "utf-8",
    "low_memory": False,
}

_DEFAULT_EXCEL_KWARGS: Dict[str, Any] = {}

_DEFAULT_JSON_KWARGS: Dict[str, Any] = {}


# ============================================================================
# DatasetLoader
# ============================================================================


class DatasetLoader:
    """Load datasets from CSV, Excel, or JSON files.

    Parameters
    ----------
    **default_kwargs
        Default keyword arguments passed to all ``pd.read_*`` calls.
        Can be overridden per call in :meth:`load`.
    """

    def __init__(self, **default_kwargs: Any) -> None:
        self._default_kwargs = default_kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        file_path: Union[str, Path],
        **kwargs: Any,
    ) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Load a dataset from a file.

        Automatically detects format from the file extension.

        Parameters
        ----------
        file_path : str | Path
            Path to the dataset file (``.csv``, ``.xlsx``, ``.xls``, ``.json``).
        **kwargs
            Additional keyword arguments passed to the underlying
            ``pd.read_*`` function. Overrides defaults set in constructor.

        Returns
        -------
        tuple[pd.DataFrame, DatasetMetadata]
            Loaded DataFrame and associated metadata.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        UnsupportedFormatError
            If the file extension is not supported.
        CorruptedDatasetError
            If the file is corrupted or unreadable.
        EmptyDatasetError
            If the file has no rows (header-only or completely empty).
        MissingColumnsError
            If the file has no columns.
        EncodingError
            If the file cannot be decoded.
        InvalidJsonError
            If JSON is not valid tabular data.
        """
        path = Path(file_path)
        logger.info("Loading dataset: %s", path)

        self._validate_file(path)
        ext = path.suffix.lower()
        self._validate_extension(ext)

        merged_kwargs = {**self._default_kwargs, **kwargs}
        df = self._read_file(path, ext, merged_kwargs)
        self._validate_dataframe(df, path)

        metadata = self._build_metadata(df, path, ext)
        logger.info(
            "Loaded dataset: %s (%d rows, %d columns)",
            path.name, len(df), len(df.columns),
        )
        return df, metadata

    # ------------------------------------------------------------------
    # File validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_file(path: Path) -> None:
        """Validate that a file exists and is not a directory.

        Parameters
        ----------
        path : Path
            File path to validate.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Path is not a file: {path}")

    @staticmethod
    def _validate_extension(ext: str) -> None:
        """Validate that the file extension is supported.

        Parameters
        ----------
        ext : str
            Lowercase file extension (e.g., ``".csv"``).

        Raises
        ------
        UnsupportedFormatError
            If the extension is not in the supported set.
        """
        if ext not in _SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file format: '{ext}'. "
                f"Supported formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}",
                details={"extension": ext, "supported": list(_SUPPORTED_EXTENSIONS)},
            )

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, path: Path) -> None:
        """Validate that the loaded DataFrame meets basic requirements.

        Parameters
        ----------
        df : pd.DataFrame
            Loaded DataFrame.
        path : Path
            Original file path (for error messages).

        Raises
        ------
        EmptyDatasetError
            If the DataFrame has zero rows.
        MissingColumnsError
            If the DataFrame has zero columns.
        """
        if len(df.columns) == 0:
            raise EmptyDatasetError(
                f"Dataset has no columns: {path}",
                details={"filename": path.name},
            )
        if len(df) == 0:
            raise EmptyDatasetError(
                f"Dataset has no rows (header-only or empty): {path}",
                details={"filename": path.name},
            )

    # ------------------------------------------------------------------
    # File reading
    # ------------------------------------------------------------------

    def _read_file(
        self,
        path: Path,
        ext: str,
        kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        """Read a file into a DataFrame based on its extension.

        Parameters
        ----------
        path : Path
            File path.
        ext : str
            Lowercase extension.
        kwargs : dict
            Read keyword arguments.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.

        Raises
        ------
        CorruptedDatasetError
            If the file is corrupted.
        EncodingError
            If encoding detection fails.
        InvalidJsonError
            If JSON is invalid.
        """
        try:
            if ext == ".csv":
                return self._read_csv(path, kwargs)
            elif ext in {".xlsx", ".xls"}:
                return self._read_excel(path, kwargs)
            elif ext == ".json":
                return self._read_json(path, kwargs)
            else:
                # Should not reach here due to _validate_extension
                raise UnsupportedFormatError(f"Unsupported format: {ext}")
        except UnicodeDecodeError as exc:
            raise EncodingError(
                f"Encoding error reading: {path}",
                details={"filename": path.name, "encoding_error": str(exc)},
                cause=exc,
            )
        except pd.errors.EmptyDataError:
            raise EmptyDatasetError(
                f"Dataset file is empty: {path}",
                details={"filename": path.name},
            )
        except pd.errors.ParserError as exc:
            raise CorruptedDatasetError(
                f"Failed to parse dataset file: {path}",
                details={"filename": path.name, "error": str(exc)},
                cause=exc,
            )
        except ValueError as exc:
            raise CorruptedDatasetError(
                f"Corrupted or invalid dataset file: {path}",
                details={"filename": path.name, "error": str(exc)},
                cause=exc,
            )

    @staticmethod
    def _read_csv(path: Path, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Read a CSV file.

        Tries UTF-8 first, then falls back to common encodings.
        """
        merged = {**_DEFAULT_CSV_KWARGS, **kwargs}
        try:
            return pd.read_csv(path, **merged)
        except UnicodeDecodeError:
            # Try common fallback encodings
            for encoding in ("latin1", "iso-8859-1", "cp1252"):
                try:
                    return pd.read_csv(path, encoding=encoding, **{k: v for k, v in merged.items() if k != "encoding"})
                except UnicodeDecodeError:
                    continue
            raise

    @staticmethod
    def _read_excel(path: Path, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Read an Excel file."""
        merged = {**_DEFAULT_EXCEL_KWARGS, **kwargs}
        return pd.read_excel(path, **merged)

    @staticmethod
    def _read_json(path: Path, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Read a JSON file.

        Validates that the JSON is tabular (list of objects or records).
        """
        import json as json_module

        merged = {**_DEFAULT_JSON_KWARGS, **kwargs}

        # First validate it's parseable JSON
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw_data = json_module.load(fh)
        except (json_module.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InvalidJsonError(
                f"Invalid JSON file: {path}",
                details={"filename": path.name, "error": str(exc)},
                cause=exc,
            )

        # Validate it's tabular (list of objects at the top level, or
        # a dict where one value is a list of objects — the records format)
        is_tabular = False
        if isinstance(raw_data, list):
            if len(raw_data) > 0:
                is_tabular = isinstance(raw_data[0], dict)
            else:
                is_tabular = True  # Empty array is acceptable
        elif isinstance(raw_data, dict):
            # Check for common records patterns
            for key in ("data", "records", "results", "items", "rows"):
                if key in raw_data and isinstance(raw_data[key], list):
                    if len(raw_data[key]) == 0 or isinstance(raw_data[key][0], dict):
                        is_tabular = True
                        break

        if not is_tabular:
            raise InvalidJsonError(
                f"JSON is not tabular data: {path}. "
                f"Expected a list of objects or a dict with a records key.",
                details={"filename": path.name},
            )

        return pd.read_json(path, **merged)

    # ------------------------------------------------------------------
    # Metadata building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(
        df: pd.DataFrame,
        path: Path,
        ext: str,
    ) -> DatasetMetadata:
        """Build a ``DatasetMetadata`` instance from a loaded DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Loaded DataFrame.
        path : Path
            Path to the original file.
        ext : str
            File extension.

        Returns
        -------
        DatasetMetadata
            Populated metadata object.
        """
        size_bytes = path.stat().st_size
        checksum = _compute_sha256(path)
        dtype_map = {col: str(dtype) for col, dtype in df.dtypes.items()}

        metadata = DatasetMetadata(
            filename=path.name,
            shape=(len(df), len(df.columns)),
            column_names=list(df.columns),
            column_types=dtype_map,
            size_bytes=size_bytes,
            checksum=checksum,
            source_format=ext,
            extra_metadata={
                "file_path": str(path.resolve()),
                "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
                "has_index": df.index.name is not None,
                "index_name": str(df.index.name) if df.index.name else None,
            },
        )

        logger.debug("Built metadata for: %s (checksum: %s...)", path.name, checksum[:12])
        return metadata

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def supported_extensions() -> frozenset:
        """Return the set of supported file extensions."""
        return frozenset(_SUPPORTED_EXTENSIONS)

    # ------------------------------------------------------------------
    # Registry integration (Part 8)
    # ------------------------------------------------------------------

    @staticmethod
    def register_with_registry(
        metadata: DatasetMetadata,
        registry: Any = None,
        fingerprint: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
    ) -> str:
        """Register dataset metadata with the ``DatasetRegistry``.

        Integrates the ingestion pipeline with ``DatasetRegistry`` to
        automatically persist metadata, fingerprint, and file location
        — without storing the actual DataFrame.

        Parameters
        ----------
        metadata : DatasetMetadata
            Dataset metadata to register.
        registry : DatasetRegistry | None
            Registry instance. If ``None``, a new one is created using
            the default path.
        fingerprint : dict | None
            Optional fingerprint dictionary (e.g., from
            ``DatasetFingerprint.to_dict()``). Stored in
            ``extra_metadata``.
        file_path : str | None
            Optional source file path. Stored in ``extra_metadata``.

        Returns
        -------
        str
            The ``dataset_id`` of the registered entry.

        Examples
        --------
        >>> from src.ingestion.dataset_loader import DatasetLoader
        >>> df, meta = loader.load("data.csv")
        >>> loader.register_with_registry(meta)
        'a1b2c3d4-...'
        """
        from src.utils.dataset_registry import DatasetRegistry as DR

        reg = registry or DR()

        # Augment extra_metadata with fingerprint and file_path
        extra = dict(metadata.extra_metadata) if metadata.extra_metadata else {}
        if fingerprint:
            extra["fingerprint"] = fingerprint
        if file_path:
            extra["file_path"] = file_path

        # Build a fresh metadata with the augmented extra_metadata
        augmented_meta = DatasetMetadata(
            dataset_id=metadata.dataset_id,
            filename=metadata.filename,
            shape=metadata.shape,
            column_names=list(metadata.column_names),
            column_types=dict(metadata.column_types),
            size_bytes=metadata.size_bytes,
            checksum=metadata.checksum,
            upload_timestamp=metadata.upload_timestamp,
            source_format=metadata.source_format,
            description=metadata.description,
            tags=tuple(metadata.tags),
            extra_metadata=extra,
        )

        dataset_id = reg.register(augmented_meta)
        logger.info(
            "Registered dataset with registry: %s (id=%s)",
            metadata.filename, dataset_id,
        )
        return dataset_id


# ============================================================================
# Module-level helper
# ============================================================================


def _compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 checksum of a file.

    Parameters
    ----------
    file_path : Path
        Path to the file.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = [
    "DatasetLoader",
]
