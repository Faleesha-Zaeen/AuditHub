"""
AuditHub - Database Module
===========================

SQLAlchemy-based database layer for the AuditHub platform.
Uses SQLite as the primary storage backend.

This module provides:
- Database engine and session management
- Declarative base for ORM models
- Connection lifecycle utilities

**No tables are created yet.** Table definitions will be added
as feature modules are implemented.

Usage::

    from src.database import get_session, init_db

    # Initialize the database (create tables)
    init_db()

    # Get a session for transactional access
    with get_session() as session:
        ...  # perform queries
"""

from pathlib import Path
from typing import Any, Dict, Generator, Optional

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.constants import DATABASE_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SQLAlchemy Declarative Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Declarative base class for all AuditHub ORM models."""

    pass


# ---------------------------------------------------------------------------
# Engine & Session Factory
# ---------------------------------------------------------------------------

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def get_engine(db_path: Optional[Path] = None) -> Engine:
    """Create or return the global SQLAlchemy engine.

    Parameters
    ----------
    db_path : Path | None
        Path to the SQLite database file. Defaults to
        ``data/audithub.db`` relative to the project root.

    Returns
    -------
    Engine
        Configured SQLAlchemy engine.
    """
    global _engine

    if _engine is not None:
        return _engine

    path = db_path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    database_url = f"sqlite:///{path.resolve()}"

    _engine = create_engine(
        database_url,
        echo=False,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )

    # Enable WAL mode for better concurrency
    @event.listens_for(_engine, "connect")
    def _set_wal_mode(dbapi_connection: Any, connection_record: Any) -> None:  # type: ignore[misc]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    logger.info("SQLAlchemy engine created: %s", database_url)
    return _engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """Create or return the global session factory.

    Parameters
    ----------
    engine : Engine | None
        SQLAlchemy engine. If ``None``, uses the global engine from
        :func:`get_engine`.

    Returns
    -------
    sessionmaker
        Configured session factory.
    """
    global _SessionFactory

    if _SessionFactory is not None:
        return _SessionFactory

    _SessionFactory = sessionmaker(
        bind=engine or get_engine(),
        autocommit=False,
        autoflush=False,
    )
    return _SessionFactory


# ---------------------------------------------------------------------------
# Session Lifecycle
# ---------------------------------------------------------------------------


def get_session() -> Generator[Session, None, None]:
    """Yield a database session as a context manager.

    Use this as a dependency or context manager to safely manage
    database sessions.

    Yields
    ------
    Session
        Active SQLAlchemy session.

    Examples
    --------
    >>> with get_session() as session:
    ...     result = session.execute(...)
    """
    factory = get_session_factory()
    session = factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_session() -> Session:
    """Create a new database session directly.

    Callers are responsible for closing the session and handling
    commits/rollbacks.

    Returns
    -------
    Session
        A new SQLAlchemy session.

    Examples
    --------
    >>> session = create_session()
    >>> try:
    ...     result = session.execute(...)
    ...     session.commit()
    ... except:
    ...     session.rollback()
    ... finally:
    ...     session.close()
    """
    factory = get_session_factory()
    return factory()


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------


def init_db(engine: Optional[Engine] = None) -> None:
    """Initialize the database by creating all defined tables.

    This function is safe to call multiple times — SQLAlchemy's
    ``create_all`` checks for existing tables.

    Parameters
    ----------
    engine : Engine | None
        SQLAlchemy engine. Uses the global engine if not provided.

    Examples
    --------
    >>> from src.database import init_db
    >>> init_db()
    """
    eng = engine or get_engine()
    Base.metadata.create_all(eng)
    logger.info("Database initialized: all tables created/verified")


def drop_all_tables(engine: Optional[Engine] = None) -> None:
    """Drop all tables from the database.

    .. warning::
        This is destructive. All data will be lost.

    Parameters
    ----------
    engine : Engine | None
        SQLAlchemy engine. Uses the global engine if not provided.
    """
    eng = engine or get_engine()
    Base.metadata.drop_all(eng)
    logger.warning("All database tables dropped")


def get_table_names(engine: Optional[Engine] = None) -> Dict[str, Any]:
    """Return metadata about all known tables.

    Parameters
    ----------
    engine : Engine | None
        SQLAlchemy engine. Uses the global engine if not provided.

    Returns
    -------
    dict
        Dictionary of table metadata.
    """
    eng = engine or get_engine()
    metadata = Base.metadata
    return {name: table.columns.keys() for name, table in metadata.tables.items()}


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    "Base",
    "create_session",
    "drop_all_tables",
    "get_engine",
    "get_session",
    "get_session_factory",
    "get_table_names",
    "init_db",
]
