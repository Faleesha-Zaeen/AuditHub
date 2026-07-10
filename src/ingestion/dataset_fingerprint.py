"""
AuditHub Ingestion - Dataset Fingerprint
==========================================

Generates unique fingerprints for datasets based on content hashing,
schema hashing, and shape hashing.

The fingerprint is used by DVC, MLflow, DatasetRegistry, and Reports
to uniquely identify a dataset version.

Usage::

    from src.ingestion.dataset_fingerprint import DatasetFingerprint
    import pandas as pd

    df = pd.read_csv("data.csv")
    fp = DatasetFingerprint.from_dataframe(df, "data.csv")
    print(fp.fingerprint_id)
    print(fp.schema_hash)
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# DatasetFingerprint
# ============================================================================


@dataclass(frozen=True)
class DatasetFingerprint:
    """Immutable fingerprint of a dataset's identity.

    Attributes
    ----------
    fingerprint_id : str
        Globally unique identifier for this fingerprint (UUID4).
    dataset_checksum : str
        SHA-256 checksum of the raw file content.
    schema_hash : str
        MD5 hash of the column names + column dtypes.
    column_hash : str
        MD5 hash of just the ordered column names.
    shape_hash : str
        MD5 hash of the shape tuple ``(rows, cols)``.
    file_hash : str
        SHA-256 hash of the raw file content (same as ``dataset_checksum``
        when present, else based on DataFrame content).
    content_hash : str
        SHA-256 hash of the full DataFrame content (all values serialized).
    created_at : str
        ISO-8601 timestamp when the fingerprint was generated.
    extra : dict
        Additional metadata for extensibility.
    """

    fingerprint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_checksum: str = ""
    schema_hash: str = ""
    column_hash: str = ""
    shape_hash: str = ""
    file_hash: str = ""
    content_hash: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """Check whether all core hash fields are populated."""
        return bool(
            self.schema_hash
            and self.column_hash
            and self.shape_hash
            and (self.content_hash or self.file_hash)
        )

    @property
    def summary(self) -> str:
        """Human-readable summary of the fingerprint."""
        lines = [
            f"Fingerprint: {self.fingerprint_id}",
            f"  Created:     {self.created_at}",
            f"  Schema:      {self.schema_hash[:12]}...",
            f"  Columns:     {self.column_hash[:12]}...",
            f"  Shape:       {self.shape_hash[:12]}...",
            f"  Content:     {self.content_hash[:16] if self.content_hash else 'N/A'}...",
            f"  File:        {self.file_hash[:16] if self.file_hash else 'N/A'}...",
            f"  Complete:    {self.is_complete}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Dictionary representation.
        """
        return {
            "fingerprint_id": self.fingerprint_id,
            "dataset_checksum": self.dataset_checksum,
            "schema_hash": self.schema_hash,
            "column_hash": self.column_hash,
            "shape_hash": self.shape_hash,
            "file_hash": self.file_hash,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "is_complete": self.is_complete,
            "extra": self.extra,
        }


# ============================================================================
# FingerprintGenerator
# ============================================================================


class FingerprintGenerator:
    """Generates :class:`DatasetFingerprint` instances from DataFrames and files.

    Usage::

        generator = FingerprintGenerator()
        fp = generator.from_dataframe(df)
        fp_from_file = generator.from_file("data.csv", df)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_dataframe(
        self,
        df: pd.DataFrame,
        file_path: Optional[Union[str, Path]] = None,
    ) -> DatasetFingerprint:
        """Generate a fingerprint from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to fingerprint.
        file_path : str | Path | None
            Optional path to the source file. If provided, the file
            checksum is also computed.

        Returns
        -------
        DatasetFingerprint
            Complete fingerprint for the dataset.
        """
        logger.debug("Generating fingerprint from DataFrame with shape %s", df.shape)

        schema_hash = self._hash_schema(df)
        column_hash = self._hash_columns(df)
        shape_hash = self._hash_shape(df)
        content_hash = self._hash_content(df)

        extra: Dict[str, Any] = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
        }

        if file_path is not None:
            file_checksum = self._hash_file(file_path)
            extra["file_path"] = str(Path(file_path).resolve())
        else:
            file_checksum = content_hash  # Use content hash as file hash

        fp = DatasetFingerprint(
            dataset_checksum=file_checksum,
            schema_hash=schema_hash,
            column_hash=column_hash,
            shape_hash=shape_hash,
            file_hash=file_checksum,
            content_hash=content_hash,
            extra=extra,
        )

        logger.debug(
            "Generated fingerprint %s (schema=%s..., shape=%s...)",
            fp.fingerprint_id[:8],
            schema_hash[:8],
            shape_hash[:8],
        )
        return fp

    def from_file(
        self,
        file_path: Union[str, Path],
        df: Optional[pd.DataFrame] = None,
    ) -> DatasetFingerprint:
        """Generate a fingerprint from a file.

        Parameters
        ----------
        file_path : str | Path
            Path to the dataset file.
        df : pd.DataFrame | None
            Optional pre-loaded DataFrame. If not provided, a file-only
            fingerprint (no content hash) is generated.

        Returns
        -------
        DatasetFingerprint
            Fingerprint with file checksum.
        """
        path = Path(file_path)
        file_checksum = self._hash_file(path)

        if df is not None:
            schema_hash = self._hash_schema(df)
            column_hash = self._hash_columns(df)
            shape_hash = self._hash_shape(df)
            content_hash = self._hash_content(df)
        else:
            schema_hash = ""
            column_hash = ""
            shape_hash = ""
            content_hash = ""

        fp = DatasetFingerprint(
            dataset_checksum=file_checksum,
            schema_hash=schema_hash,
            column_hash=column_hash,
            shape_hash=shape_hash,
            file_hash=file_checksum,
            content_hash=content_hash,
            extra={"file_path": str(path.resolve()), "file_size": path.stat().st_size},
        )

        logger.debug("Generated file fingerprint for: %s", path.name)
        return fp

    # ------------------------------------------------------------------
    # Hashing internals
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_schema(df: pd.DataFrame) -> str:
        """Hash column names + dtypes.

        Returns
        -------
        str
            MD5 hex digest.
        """
        schema_str = json.dumps(
            {col: str(dtype) for col, dtype in df.dtypes.items()},
            sort_keys=True,
        )
        return hashlib.md5(schema_str.encode()).hexdigest()

    @staticmethod
    def _hash_columns(df: pd.DataFrame) -> str:
        """Hash just the ordered column names.

        Returns
        -------
        str
            MD5 hex digest.
        """
        cols_str = json.dumps(list(df.columns))
        return hashlib.md5(cols_str.encode()).hexdigest()

    @staticmethod
    def _hash_shape(df: pd.DataFrame) -> str:
        """Hash the shape tuple.

        Returns
        -------
        str
            MD5 hex digest of ``"(rows, cols)"``.
        """
        shape_str = f"({len(df)},{len(df.columns)})"
        return hashlib.md5(shape_str.encode()).hexdigest()

    @staticmethod
    def _hash_content(df: pd.DataFrame) -> str:
        """Hash the full DataFrame content.

        Uses SHA-256 on the CSV-serialized representation of all values.

        Returns
        -------
        str
            SHA-256 hex digest.
        """
        content = df.to_csv(index=False, sep=",").encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _hash_file(file_path: Union[str, Path]) -> str:
        """Compute a SHA-256 checksum of a file.

        Parameters
        ----------
        file_path : str | Path
            Path to the file.

        Returns
        -------
        str
            SHA-256 hex digest.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found for hashing: {path}")

        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


__all__ = [
    "DatasetFingerprint",
    "FingerprintGenerator",
]
