"""
AuditHub - Dataset Registry
=============================

SQLite-backed registry for maintaining metadata about uploaded datasets.
Stores only metadata — never the actual DataFrame.

Uses SQLAlchemy ORM for database access. Tables are created on first
use via the existing database module.

Usage::

    from src.utils.dataset_registry import DatasetRegistry
    from src.utils.dataset_metadata import DatasetMetadata

    registry = DatasetRegistry()
    meta = DatasetMetadata(filename="data.csv", ...)
    registry.register(meta)
    retrieved = registry.get(meta.dataset_id)
    all_datasets = registry.list_all()
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import Session as SASession
from sqlalchemy.orm import declarative_base, sessionmaker

from src.utils.dataset_metadata import DatasetMetadata
from src.utils.exceptions import DatabaseException
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SQLAlchemy ORM Model
# ---------------------------------------------------------------------------

_Base = declarative_base()


class _DatasetRecord(_Base):
    """Internal ORM model for the dataset_registry table."""

    __tablename__ = "dataset_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(36), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    num_rows = Column(Integer, default=0)
    num_columns = Column(Integer, default=0)
    size_bytes = Column(Integer, default=0)
    checksum = Column(String(128), default="")
    source_format = Column(String(16), default="")
    description = Column(Text, nullable=True)
    tags = Column(Text, default="")  # Comma-separated
    column_names = Column(Text, default="")  # JSON-encoded list
    column_types = Column(Text, default="")  # JSON-encoded dict
    extra_metadata = Column(Text, default="{}")  # JSON-encoded dict
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<_DatasetRecord(dataset_id='{self.dataset_id}', filename='{self.filename}')>"


# ============================================================================
# DatasetRegistry
# ============================================================================


class DatasetRegistry:
    """SQLite-backed registry for dataset metadata.

    Parameters
    ----------
    db_path : str | None
        Path to the SQLite database file. If ``None``, uses the default
        path ``data/audithub.db`` relative to the working directory.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or "data/audithub.db"
        self._engine = create_engine(
            f"sqlite:///{self._db_path}",
            connect_args={"check_same_thread": False},
        )
        # Create table if it doesn't exist
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.debug("DatasetRegistry initialized with db: %s", self._db_path)

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def register(self, metadata: DatasetMetadata) -> str:
        """Register a new dataset's metadata.

        Parameters
        ----------
        metadata : DatasetMetadata
            Dataset metadata to persist.

        Returns
        -------
        str
            The dataset ID of the registered entry.

        Raises
        ------
        DatabaseException
            If registration fails (e.g., duplicate ID).
        """
        import json

        session = self._Session()
        try:
            record = _DatasetRecord(
                dataset_id=metadata.dataset_id,
                filename=metadata.filename,
                num_rows=metadata.num_rows,
                num_columns=metadata.num_columns,
                size_bytes=metadata.size_bytes,
                checksum=metadata.checksum,
                source_format=metadata.source_format,
                description=metadata.description,
                tags=",".join(metadata.tags) if metadata.tags else "",
                column_names=json.dumps(metadata.column_names),
                column_types=json.dumps(metadata.column_types),
                extra_metadata=json.dumps(metadata.extra_metadata),
            )
            session.add(record)
            session.commit()
            logger.info("Registered dataset: %s (%s)", metadata.filename, metadata.dataset_id)
            return metadata.dataset_id
        except Exception as exc:
            session.rollback()
            raise DatabaseException(
                f"Failed to register dataset: {metadata.filename}",
                details={"dataset_id": metadata.dataset_id},
                cause=exc,
            )
        finally:
            session.close()

    def get(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Retrieve metadata by dataset ID.

        Parameters
        ----------
        dataset_id : str
            Unique dataset identifier.

        Returns
        -------
        DatasetMetadata | None
            The metadata if found, ``None`` otherwise.
        """
        import json

        session = self._Session()
        try:
            record = (
                session.query(_DatasetRecord)
                .filter(_DatasetRecord.dataset_id == dataset_id)
                .first()
            )
            if record is None:
                return None

            return DatasetMetadata(
                dataset_id=record.dataset_id,
                filename=record.filename,
                shape=(record.num_rows or 0, record.num_columns or 0),
                size_bytes=record.size_bytes or 0,
                checksum=record.checksum or "",
                source_format=record.source_format or "",
                description=record.description,
                tags=tuple(t.strip() for t in record.tags.split(",") if t.strip()) if record.tags else (),
                column_names=json.loads(record.column_names) if record.column_names else [],
                column_types=json.loads(record.column_types) if record.column_types else {},
                extra_metadata=json.loads(record.extra_metadata) if record.extra_metadata else {},
                upload_timestamp=record.created_at.isoformat() if record.created_at else "",
            )
        finally:
            session.close()

    def get_by_filename(self, filename: str) -> List[DatasetMetadata]:
        """Retrieve all datasets matching a filename.

        Parameters
        ----------
        filename : str
            Exact filename to search for.

        Returns
        -------
        list[DatasetMetadata]
            List of matching metadata records.
        """
        import json

        session = self._Session()
        try:
            records = (
                session.query(_DatasetRecord)
                .filter(_DatasetRecord.filename == filename)
                .all()
            )
            results = []
            for record in records:
                results.append(DatasetMetadata(
                    dataset_id=record.dataset_id,
                    filename=record.filename,
                    shape=(record.num_rows or 0, record.num_columns or 0),
                    size_bytes=record.size_bytes or 0,
                    checksum=record.checksum or "",
                    source_format=record.source_format or "",
                    description=record.description,
                    tags=tuple(t.strip() for t in record.tags.split(",") if t.strip()) if record.tags else (),
                    column_names=json.loads(record.column_names) if record.column_names else [],
                    column_types=json.loads(record.column_types) if record.column_types else {},
                    extra_metadata=json.loads(record.extra_metadata) if record.extra_metadata else {},
                    upload_timestamp=record.created_at.isoformat() if record.created_at else "",
                ))
            return results
        finally:
            session.close()

    def list_all(self) -> List[DatasetMetadata]:
        """List all registered datasets.

        Returns
        -------
        list[DatasetMetadata]
            List of all metadata records, sorted by creation time (newest first).
        """
        import json

        session = self._Session()
        try:
            records = (
                session.query(_DatasetRecord)
                .order_by(_DatasetRecord.created_at.desc())
                .all()
            )
            results = []
            for record in records:
                results.append(DatasetMetadata(
                    dataset_id=record.dataset_id,
                    filename=record.filename,
                    shape=(record.num_rows or 0, record.num_columns or 0),
                    size_bytes=record.size_bytes or 0,
                    checksum=record.checksum or "",
                    source_format=record.source_format or "",
                    description=record.description,
                    tags=tuple(t.strip() for t in record.tags.split(",") if t.strip()) if record.tags else (),
                    column_names=json.loads(record.column_names) if record.column_names else [],
                    column_types=json.loads(record.column_types) if record.column_types else {},
                    extra_metadata=json.loads(record.extra_metadata) if record.extra_metadata else {},
                    upload_timestamp=record.created_at.isoformat() if record.created_at else "",
                ))
            return results
        finally:
            session.close()

    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset's metadata by ID.

        Parameters
        ----------
        dataset_id : str
            Unique dataset identifier.

        Returns
        -------
        bool
            ``True`` if deleted, ``False`` if not found.
        """
        session = self._Session()
        try:
            record = (
                session.query(_DatasetRecord)
                .filter(_DatasetRecord.dataset_id == dataset_id)
                .first()
            )
            if record is None:
                return False

            session.delete(record)
            session.commit()
            logger.info("Deleted dataset registry entry: %s", dataset_id)
            return True
        except Exception as exc:
            session.rollback()
            raise DatabaseException(
                f"Failed to delete dataset: {dataset_id}",
                cause=exc,
            )
        finally:
            session.close()

    def count(self) -> int:
        """Return the total number of registered datasets.

        Returns
        -------
        int
            Count of metadata records.
        """
        session = self._Session()
        try:
            return session.query(_DatasetRecord).count()
        finally:
            session.close()

    def update_metadata(
        self,
        dataset_id: str,
        updates: Dict[str, Any],
    ) -> Optional[DatasetMetadata]:
        """Update metadata fields for an existing dataset.

        Parameters
        ----------
        dataset_id : str
            Unique dataset identifier.
        updates : dict
            Fields to update. Supported keys include ``description``,
            ``tags``, ``extra_metadata``.

        Returns
        -------
        DatasetMetadata | None
            Updated metadata, or ``None`` if not found.
        """
        import json

        session = self._Session()
        try:
            record = (
                session.query(_DatasetRecord)
                .filter(_DatasetRecord.dataset_id == dataset_id)
                .first()
            )
            if record is None:
                return None

            if "description" in updates:
                record.description = updates["description"]
            if "tags" in updates:
                tags = updates["tags"]
                if isinstance(tags, (list, tuple)):
                    record.tags = ",".join(tags)
                else:
                    record.tags = str(tags)
            if "extra_metadata" in updates:
                current_extra = json.loads(record.extra_metadata) if record.extra_metadata else {}
                current_extra.update(updates["extra_metadata"])
                record.extra_metadata = json.dumps(current_extra)

            session.commit()
            logger.info("Updated metadata for dataset: %s", dataset_id)

            # Refresh from DB and return
            return self.get(dataset_id)
        except Exception as exc:
            session.rollback()
            raise DatabaseException(
                f"Failed to update metadata for dataset: {dataset_id}",
                cause=exc,
            )
        finally:
            session.close()

    def close(self) -> None:
        """Dispose of the database engine and release resources."""
        self._engine.dispose()
        logger.debug("DatasetRegistry engine disposed")


__all__ = [
    "DatasetRegistry",
]
