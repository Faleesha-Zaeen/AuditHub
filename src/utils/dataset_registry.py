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
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import Session as SASession
from sqlalchemy.orm import declarative_base, sessionmaker

from src.utils.constants import DATABASE_PATH
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
    # Versioning: datasets sharing a version_group are successive versions of
    # the same logical dataset, numbered from 1.
    version = Column(Integer, default=1, index=True)
    version_group = Column(String(512), default="", index=True)
    parent_dataset_id = Column(String(36), default="")
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
        # Anchored to the project root (not the process CWD) and its parent
        # directory created first, so the registry also works when launched
        # from another working directory or on a fresh deployment where
        # data/ is not under version control.
        default_path = DATABASE_PATH
        default_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_path or default_path))
        self._engine = create_engine(
            f"sqlite:///{self._db_path}",
            connect_args={"check_same_thread": False},
        )
        # Create table if it doesn't exist
        _Base.metadata.create_all(self._engine)
        self._migrate()
        self._Session = sessionmaker(bind=self._engine)
        logger.debug("DatasetRegistry initialized with db: %s", self._db_path)

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created.

        ``create_all`` only creates missing *tables*, so a registry written by
        an earlier version keeps its original columns. Rather than requiring a
        migration tool for three nullable columns, they are added in place when
        absent. Existing rows become version 1 of a group named after their
        filename, which is the correct reading of a pre-versioning registry.
        """
        added: List[str] = []
        with self._engine.begin() as conn:
            existing = {
                row[1] for row in conn.exec_driver_sql(
                    "PRAGMA table_info(dataset_registry)"
                ).fetchall()
            }
            for name, ddl in (
                ("version", "INTEGER DEFAULT 1"),
                ("version_group", "VARCHAR(512) DEFAULT ''"),
                ("parent_dataset_id", "VARCHAR(36) DEFAULT ''"),
            ):
                if name not in existing:
                    conn.exec_driver_sql(
                        f"ALTER TABLE dataset_registry ADD COLUMN {name} {ddl}"
                    )
                    added.append(name)

            if added:
                # Backfill: pre-existing rows are version 1, grouped by filename.
                conn.exec_driver_sql(
                    "UPDATE dataset_registry SET version = 1 WHERE version IS NULL"
                )
                conn.exec_driver_sql(
                    "UPDATE dataset_registry SET version_group = filename "
                    "WHERE version_group IS NULL OR version_group = ''"
                )
                logger.info(
                    "Migrated dataset_registry: added column(s) %s", ", ".join(added)
                )

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    @staticmethod
    def default_version_group(filename: str) -> str:
        """Derive a version group from a filename.

        Successive exports are usually named ``customers_v1.csv``,
        ``customers_2024_06.csv`` and so on, so the stem is stripped of a
        trailing version or date suffix to keep them in one group.
        """
        import re

        stem = str(filename).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        stem = stem.rsplit(".", 1)[0]
        stem = re.sub(r"[_-](v?\d+(\.\d+)*|\d{4}([_-]?\d{2}){0,2})$", "", stem, flags=re.IGNORECASE)
        return stem or str(filename)

    def register(
        self,
        metadata: DatasetMetadata,
        version_group: Optional[str] = None,
        parent_dataset_id: Optional[str] = None,
    ) -> str:
        """Register a new dataset's metadata.

        Parameters
        ----------
        metadata : DatasetMetadata
            Dataset metadata to persist.
        version_group : str | None
            Logical dataset this file is a version of. Derived from the
            filename when omitted. The version number is assigned
            automatically as one past the highest existing version in the
            group, so repeated uploads of the same logical dataset become
            V1, V2, V3 without the caller tracking anything.
        parent_dataset_id : str | None
            Explicit predecessor. Defaults to the latest version in the group.

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
            group = version_group or self.default_version_group(metadata.filename)
            previous = (
                session.query(_DatasetRecord)
                .filter(_DatasetRecord.version_group == group)
                .order_by(_DatasetRecord.version.desc())
                .first()
            )
            next_version = (previous.version or 0) + 1 if previous else 1
            parent = parent_dataset_id or (previous.dataset_id if previous else "")

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
                version=next_version,
                version_group=group,
                parent_dataset_id=parent,
            )
            session.add(record)
            session.commit()
            logger.info(
                "Registered dataset: %s (%s) as %s v%d",
                metadata.filename, metadata.dataset_id, group, next_version,
            )
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
            return self._to_metadata(record)
        finally:
            session.close()

    def _to_metadata(self, record: "_DatasetRecord") -> DatasetMetadata:
        """Convert an ORM row into a :class:`DatasetMetadata`."""
        import json

        extra = json.loads(record.extra_metadata) if record.extra_metadata else {}
        # Surface the versioning fields without changing the frozen dataclass.
        extra.setdefault("version", record.version or 1)
        extra.setdefault("version_group", record.version_group or "")
        extra.setdefault("parent_dataset_id", record.parent_dataset_id or "")

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
            extra_metadata=extra,
            upload_timestamp=record.created_at.isoformat() if record.created_at else "",
        )

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    def list_version_groups(self) -> List[Dict[str, Any]]:
        """List every logical dataset that has at least one registered version.

        Returns
        -------
        list[dict]
            One entry per group with its name, version count and latest version.
        """
        session = self._Session()
        try:
            records = (
                session.query(_DatasetRecord)
                .order_by(_DatasetRecord.version_group, _DatasetRecord.version)
                .all()
            )
            groups: Dict[str, Dict[str, Any]] = {}
            for record in records:
                group = record.version_group or record.filename
                entry = groups.setdefault(
                    group, {"version_group": group, "versions": 0, "latest_version": 0,
                            "latest_dataset_id": "", "filenames": []}
                )
                entry["versions"] += 1
                if (record.version or 1) >= entry["latest_version"]:
                    entry["latest_version"] = record.version or 1
                    entry["latest_dataset_id"] = record.dataset_id
                if record.filename not in entry["filenames"]:
                    entry["filenames"].append(record.filename)
            return sorted(groups.values(), key=lambda g: g["version_group"])
        finally:
            session.close()

    def list_versions(self, version_group: str) -> List[DatasetMetadata]:
        """Return every registered version of a logical dataset, oldest first."""
        session = self._Session()
        try:
            records = (
                session.query(_DatasetRecord)
                .filter(_DatasetRecord.version_group == version_group)
                .order_by(_DatasetRecord.version)
                .all()
            )
            return [self._to_metadata(r) for r in records]
        finally:
            session.close()

    def get_version(self, version_group: str, version: int) -> Optional[DatasetMetadata]:
        """Return a specific version of a logical dataset, if registered."""
        session = self._Session()
        try:
            record = (
                session.query(_DatasetRecord)
                .filter(
                    _DatasetRecord.version_group == version_group,
                    _DatasetRecord.version == version,
                )
                .first()
            )
            return self._to_metadata(record) if record else None
        finally:
            session.close()

    def latest_version(self, version_group: str) -> Optional[DatasetMetadata]:
        """Return the most recent version of a logical dataset."""
        session = self._Session()
        try:
            record = (
                session.query(_DatasetRecord)
                .filter(_DatasetRecord.version_group == version_group)
                .order_by(_DatasetRecord.version.desc())
                .first()
            )
            return self._to_metadata(record) if record else None
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
                results.append(self._to_metadata(record))
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
                results.append(self._to_metadata(record))
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
