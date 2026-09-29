"""
AuditHub Drift Detection Module
================================

Compares a current dataset against a reference (training) dataset and reports
which columns have moved, by how much, and why.
"""

from src.drift.detector import (
    ColumnDriftResult,
    DriftDetector,
    DriftReport,
    DriftStatus,
)

__all__ = [
    "ColumnDriftResult",
    "DriftDetector",
    "DriftReport",
    "DriftStatus",
]
