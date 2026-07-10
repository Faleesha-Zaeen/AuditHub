"""
AuditHub - Global Constants
============================

Central repository for all directory paths, filenames, supported file
types, and other global constants used across the AuditHub platform.

Never hardcode paths or magic values in business logic modules; import
from here instead.
"""

from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

# ============================================================================
# Project Root
# ============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

# ============================================================================
# Directory Paths
# ============================================================================

APP_DIR: Path = PROJECT_ROOT / "app"
PAGES_DIR: Path = APP_DIR / "pages"
COMPONENTS_DIR: Path = APP_DIR / "components"
ASSETS_DIR: Path = APP_DIR / "assets"

CONFIG_DIR: Path = PROJECT_ROOT / "configs"

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
VALIDATED_DATA_DIR: Path = DATA_DIR / "validated"
REPAIRED_DATA_DIR: Path = DATA_DIR / "repaired"
MUTATED_DATA_DIR: Path = DATA_DIR / "mutated"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REPORTS_DATA_DIR: Path = DATA_DIR / "reports"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

SOURCE_DIR: Path = PROJECT_ROOT / "src"
INGESTION_DIR: Path = SOURCE_DIR / "ingestion"
VALIDATION_DIR: Path = SOURCE_DIR / "validation"
PROFILING_DIR: Path = SOURCE_DIR / "profiling"
QUALITY_DIR: Path = SOURCE_DIR / "quality"
REPAIR_DIR: Path = SOURCE_DIR / "repair"
HEALTH_DIR: Path = SOURCE_DIR / "health"
MUTATION_LAB_DIR: Path = SOURCE_DIR / "mutation_lab"
ROBUSTNESS_DIR: Path = SOURCE_DIR / "robustness"
TRAINING_DIR: Path = SOURCE_DIR / "training"
EVALUATION_DIR: Path = SOURCE_DIR / "evaluation"
REPORTING_DIR: Path = SOURCE_DIR / "reporting"
DATABASE_DIR: Path = SOURCE_DIR / "database"
UTILS_DIR: Path = SOURCE_DIR / "utils"
PIPELINE_DIR: Path = SOURCE_DIR / "pipeline"

# ============================================================================
# File & Configuration Paths
# ============================================================================

CONFIG_FILE: Path = CONFIG_DIR / "config.yaml"
PARAMS_FILE: Path = CONFIG_DIR / "params.yaml"
SCHEMA_FILE: Path = CONFIG_DIR / "schema.yaml"
LOGGING_CONFIG_FILE: Path = CONFIG_DIR / "logging.yaml"

DATABASE_PATH: Path = DATA_DIR / "audithub.db"

LOGS_FILE: Path = LOGS_DIR / "audithub.log"
ERROR_LOGS_FILE: Path = LOGS_DIR / "audithub_error.log"

# ============================================================================
# Supported File Types
# ============================================================================

SUPPORTED_DATA_EXTENSIONS: FrozenSet[str] = frozenset({
    ".csv",
    ".tsv",
    ".json",
    ".parquet",
    ".xlsx",
    ".xls",
    ".feather",
    ".pickle",
    ".pkl",
})

SUPPORTED_IMAGE_EXTENSIONS: FrozenSet[str] = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".webp",
})

SUPPORTED_REPORT_EXTENSIONS: FrozenSet[str] = frozenset({
    ".html",
    ".pdf",
    ".json",
    ".md",
})

SUPPORTED_MODEL_EXTENSIONS: FrozenSet[str] = frozenset({
    ".pkl",
    ".joblib",
    ".onnx",
    ".pt",
    ".pth",
})

# ============================================================================
# Data Types
# ============================================================================

# Column type mapping for pandas dtype inference
COLUMN_TYPE_MAPPING: Dict[str, List[str]] = {
    "integer": [
        "int64", "int32", "int16", "int8",
        "uint64", "uint32", "uint16", "uint8",
    ],
    "float": [
        "float64", "float32", "float16",
    ],
    "categorical": [
        "object", "category", "bool",
    ],
    "datetime": [
        "datetime64[ns]", "datetime64[ns, UTC]",
    ],
}

NUMERIC_DTYPES: Tuple[str, ...] = (
    *COLUMN_TYPE_MAPPING["integer"],
    *COLUMN_TYPE_MAPPING["float"],
)

CATEGORICAL_DTYPES: Tuple[str, ...] = tuple(COLUMN_TYPE_MAPPING["categorical"])

DATETIME_DTYPES: Tuple[str, ...] = tuple(COLUMN_TYPE_MAPPING["datetime"])

# ============================================================================
# Dataset Constraints
# ============================================================================

MIN_DATASET_ROWS: int = 10
MAX_DATASET_ROWS: int = 10_000_000
MIN_DATASET_COLUMNS: int = 1
MAX_DATASET_COLUMNS: int = 1000
MAX_FILE_SIZE_MB: int = 500
DEFAULT_CHUNK_SIZE: int = 10_000

# ============================================================================
# MLflow Constants
# ============================================================================

DEFAULT_MLFLOW_EXPERIMENT: str = "audithub_default"
DEFAULT_MLFLOW_TRACKING_URI: str = str(MLRUNS_DIR)

# ============================================================================
# Health Score Weights
# ============================================================================

HEALTH_SCORE_WEIGHTS: Dict[str, float] = {
    "completeness": 0.25,
    "uniqueness": 0.15,
    "consistency": 0.20,
    "accuracy": 0.25,
    "timeliness": 0.15,
}

HEALTH_GRADE_RANGES: Dict[str, Tuple[float, float]] = {
    "A": (90.0, 100.0),
    "B": (75.0, 89.0),
    "C": (60.0, 74.0),
    "D": (40.0, 59.0),
    "F": (0.0, 39.0),
}

# ============================================================================
# Pipeline Constants
# ============================================================================

DEFAULT_RANDOM_SEED: int = 42
DEFAULT_TEST_SPLIT: float = 0.2
DEFAULT_VALIDATION_SPLIT: float = 0.1
DEFAULT_CV_FOLDS: int = 5

# ============================================================================
# Environment Variables
# ============================================================================

ENV_AUDITHUB_ENVIRONMENT: str = "AUDITHUB_ENVIRONMENT"
ENV_AUDITHUB_DEBUG: str = "AUDITHUB_DEBUG"
ENV_AUDITHUB_CONFIG_PATH: str = "AUDITHUB_CONFIG_PATH"
ENV_AUDITHUB_DB_PATH: str = "AUDITHUB_DB_PATH"
ENV_MLFLOW_TRACKING_URI: str = "MLFLOW_TRACKING_URI"

# ============================================================================
# Application Metadata
# ============================================================================

APP_NAME: str = "AuditHub"
APP_VERSION: str = "0.1.0"
APP_DESCRIPTION: str = "End-to-end Dataset Quality, Repair, Robustness and MLOps Platform"
APP_AUTHOR: str = "AuditHub Team"
