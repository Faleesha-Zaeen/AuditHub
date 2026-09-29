"""
AuditHub Lineage Module
========================

Dataset version comparison and pipeline lineage tracking, both backed by the
existing ``data/audithub.db`` registry database.
"""

from src.lineage.comparator import (
    ChangeType,
    ColumnChange,
    VersionComparator,
    VersionComparison,
)
from src.lineage.tracker import LineageEvent, LineageTrace, LineageTracker

__all__ = [
    "ChangeType",
    "ColumnChange",
    "LineageEvent",
    "LineageTrace",
    "LineageTracker",
    "VersionComparator",
    "VersionComparison",
]
