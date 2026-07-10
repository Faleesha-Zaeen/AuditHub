"""
AuditHub - Dataset Ingestion & Intelligent Dataset Analysis Engine
====================================================================

Production-grade ingestion module for loading, analyzing, and
fingerprinting any structured tabular dataset.

The engine is **completely dataset-agnostic** — no assumptions are made
about column names, schemas, data types, or domain.

Capabilities
------------
- Load CSV, Excel (.xlsx, .xls), and JSON (tabular) files
- Auto-detect file format from extension
- Validate file integrity (corruption detection, encoding fallbacks)
- Compute comprehensive dataset statistics
- Infer column types (numeric, categorical, boolean, datetime, text, etc.)
- Detect potential target columns, ID columns, and feature columns
- Detect dataset type (classification, regression, time-series, unsupervised)
- Generate unique dataset fingerprints (for DVC, MLflow, Registry)
- Integrate with ``DatasetRegistry`` for metadata persistence

Architecture
------------
dataset_loader.py      DatasetLoader — safe file loading + metadata
dataset_analyzer.py    DatasetAnalyzer — stats, type inference, detection
dataset_fingerprint.py DatasetFingerprint — content/schema/shape hashing
exceptions.py          Custom exceptions for ingestion errors
"""

from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.dataset_analyzer import (
    ColumnCategory,
    ColumnInfo,
    DatasetAnalyzer,
    DatasetSummary,
)
from src.ingestion.dataset_fingerprint import DatasetFingerprint, FingerprintGenerator
from src.ingestion.exceptions import (
    CorruptedDatasetError,
    EmptyDatasetError,
    EncodingError,
    InvalidJsonError,
    MissingColumnsError,
    UnsupportedFormatError,
)

__all__ = [
    # Loader
    "DatasetLoader",
    # Analyzer
    "ColumnCategory",
    "ColumnInfo",
    "DatasetAnalyzer",
    "DatasetSummary",
    # Fingerprint
    "DatasetFingerprint",
    "FingerprintGenerator",
    # Exceptions
    "CorruptedDatasetError",
    "EmptyDatasetError",
    "EncodingError",
    "InvalidJsonError",
    "MissingColumnsError",
    "UnsupportedFormatError",
]
