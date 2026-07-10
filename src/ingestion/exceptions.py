"""
AuditHub Ingestion - Custom Exceptions
=======================================

Specialized exceptions for dataset ingestion operations.
All inherit from the base ``IngestionException`` in ``src.utils.exceptions``.
"""

from src.utils.exceptions import IngestionException


class UnsupportedFormatError(IngestionException):
    """Raised when a dataset file format is not supported.

    Supported formats: CSV, Excel (.xlsx, .xls), JSON (tabular).
    """


class CorruptedDatasetError(IngestionException):
    """Raised when a dataset file appears to be corrupted or unreadable."""


class EmptyDatasetError(IngestionException):
    """Raised when a dataset has no rows after loading."""


class EncodingError(IngestionException):
    """Raised when a dataset file cannot be decoded with the detected encoding."""


class InvalidJsonError(IngestionException):
    """Raised when a JSON file cannot be parsed or is not tabular JSON.
    
    Tabular JSON must be either a list of objects or a records-oriented structure.
    """


class MissingColumnsError(IngestionException):
    """Raised when a dataset has no columns after loading."""


__all__ = [
    "UnsupportedFormatError",
    "CorruptedDatasetError",
    "EmptyDatasetError",
    "EncodingError",
    "InvalidJsonError",
    "MissingColumnsError",
]
