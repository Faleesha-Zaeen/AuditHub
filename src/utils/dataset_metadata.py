"""
AuditHub - Dataset Metadata
=============================

Strongly typed dataclass for storing dataset metadata.
Stores information *about* a dataset — not the dataset itself.
The actual DataFrame is never stored in this object.

Usage::

    from src.utils.dataset_metadata import DatasetMetadata

    meta = DatasetMetadata(
        filename="dataset.csv",
        shape=(1000, 10),
        column_names=["id", "feat_1", "target"],
        column_types={"id": "int64", "feat_1": "float64", "target": "int64"},
        size_bytes=204800,
        checksum="abc123def456",
        dataset_id="uuid-here",
    )
    print(meta.summary)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# DatasetMetadata
# ============================================================================


@dataclass(frozen=True)
class DatasetMetadata:
    """Immutable metadata about a dataset.

    Attributes
    ----------
    dataset_id : str
        Unique identifier for the dataset (UUID4).
    filename : str
        Original filename of the dataset.
    shape : tuple[int, int]
        (rows, columns) of the dataset.
    column_names : list[str]
        Ordered list of column names.
    column_types : dict[str, str]
        Mapping of column name -> pandas dtype string.
    size_bytes : int
        File size in bytes on disk.
    checksum : str
        SHA-256 checksum of the file.
    upload_timestamp : str
        ISO-8601 formatted upload timestamp (UTC).
    source_format : str
        Original file format/extension (e.g., ``".csv"``).
    description : str | None
        Optional user-provided description.
    tags : tuple[str, ...]
        Optional tags for categorization.
    extra_metadata : dict
        Additional metadata key-value pairs for extensibility.
    """

    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    shape: Tuple[int, int] = (0, 0)
    column_names: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    size_bytes: int = 0
    checksum: str = ""
    upload_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    source_format: str = ""
    description: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_rows(self) -> int:
        """Number of rows in the dataset."""
        return self.shape[0]

    @property
    def num_columns(self) -> int:
        """Number of columns in the dataset."""
        return self.shape[1]

    @property
    def size_kb(self) -> float:
        """File size in kilobytes."""
        return self.size_bytes / 1024

    @property
    def size_mb(self) -> float:
        """File size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def summary(self) -> str:
        """Human-readable summary of the dataset.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            f"Dataset: {self.filename}",
            f"  ID:       {self.dataset_id}",
            f"  Shape:    {self.num_rows} rows x {self.num_columns} columns",
            f"  Size:     {self.size_mb:.2f} MB ({self.size_bytes} bytes)",
            f"  Format:   {self.source_format}",
            f"  Checksum: {self.checksum[:16]}...",
            f"  Upload:   {self.upload_timestamp}",
        ]
        if self.description:
            lines.append(f"  Desc:     {self.description}")
        if self.tags:
            lines.append(f"  Tags:     {', '.join(self.tags)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Create a DatasetMetadata instance from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing metadata fields.

        Returns
        -------
        DatasetMetadata
            New metadata instance.
        """
        # Convert shape from list to tuple if needed
        shape_data = data.get("shape", (0, 0))
        if isinstance(shape_data, list):
            shape_data = tuple(shape_data)

        # Convert tags from list to tuple if needed
        tags_data = data.get("tags", ())
        if isinstance(tags_data, list):
            tags_data = tuple(tags_data)

        return cls(
            dataset_id=data.get("dataset_id", str(uuid.uuid4())),
            filename=data.get("filename", ""),
            shape=shape_data,
            column_names=data.get("column_names", []),
            column_types=data.get("column_types", {}),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
            upload_timestamp=data.get(
                "upload_timestamp",
                datetime.now(timezone.utc).isoformat(),
            ),
            source_format=data.get("source_format", ""),
            description=data.get("description"),
            tags=tags_data,
            extra_metadata=data.get("extra_metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary representation.
        """
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "shape": list(self.shape),
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "column_names": self.column_names,
            "column_types": self.column_types,
            "size_bytes": self.size_bytes,
            "size_kb": self.size_kb,
            "size_mb": self.size_mb,
            "checksum": self.checksum,
            "upload_timestamp": self.upload_timestamp,
            "source_format": self.source_format,
            "description": self.description,
            "tags": list(self.tags),
            "extra_metadata": self.extra_metadata,
        }


__all__ = [
    "DatasetMetadata",
]
