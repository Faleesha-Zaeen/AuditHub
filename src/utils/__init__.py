"""
AuditHub - Utilities Package
=============================

Shared utilities for the AuditHub platform.

Modules
-------
logger
    Centralized logging system with file rotation and console output.
constants
    Global constants, paths, and configuration defaults.
exceptions
    Custom exception hierarchy for structured error handling.
helpers
    Reusable helper functions for file I/O, YAML, and timestamps.
config_manager
    Centralized configuration system with caching and env var overrides.
app_settings
    Strongly typed application settings dataclass.
path_manager
    Path resolution with automatic directory creation.
file_manager
    DataFrame, JSON, and pickle file I/O with backup support.
dataset_metadata
    Strongly typed dataset metadata dataclass.
dataset_registry
    SQLite-backed registry for dataset metadata persistence.
decorators
    Reusable decorators: timer, log_execution, retry, validate_input.
"""

from src.utils.logger import get_logger, set_log_level
from src.utils.constants import (
    APP_NAME,
    APP_VERSION,
    CONFIG_DIR,
    DATA_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
    SUPPORTED_DATA_EXTENSIONS,
)
from src.utils.exceptions import (
    AuditHubException,
    ConfigurationException,
    DatabaseException,
    DatasetException,
    HealthException,
    IngestionException,
    ModelException,
    MutationException,
    PipelineException,
    ProfilingException,
    QualityException,
    RepairException,
    ReportException,
    RobustnessException,
    ValidationException,
)
from src.utils.helpers import (
    create_directory,
    ensure_directory_exists,
    generate_timestamp,
    generate_uuid,
    list_files,
    load_yaml,
    safe_filename,
    save_yaml,
)
from src.utils.config_manager import ConfigManager
from src.utils.app_settings import AppSettings
from src.utils.path_manager import PathManager
from src.utils.file_manager import FileManager
from src.utils.dataset_metadata import DatasetMetadata
from src.utils.dataset_registry import DatasetRegistry
from src.utils.decorators import (
    timer as decorators_timer,
    log_execution,
    retry,
    validate_input,
)

# Re-export timer with preferred namespace
timer = decorators_timer

__all__ = [
    # Logger
    "get_logger",
    "set_log_level",
    # Constants
    "APP_NAME",
    "APP_VERSION",
    "CONFIG_DIR",
    "DATA_DIR",
    "LOGS_DIR",
    "PROJECT_ROOT",
    "SUPPORTED_DATA_EXTENSIONS",
    # Exceptions
    "AuditHubException",
    "ConfigurationException",
    "DatabaseException",
    "DatasetException",
    "HealthException",
    "IngestionException",
    "ModelException",
    "MutationException",
    "PipelineException",
    "ProfilingException",
    "QualityException",
    "RepairException",
    "ReportException",
    "RobustnessException",
    "ValidationException",
    # Helpers
    "create_directory",
    "ensure_directory_exists",
    "generate_timestamp",
    "generate_uuid",
    "list_files",
    "load_yaml",
    "safe_filename",
    "save_yaml",
    # Config Manager
    "ConfigManager",
    # App Settings
    "AppSettings",
    # Path Manager
    "PathManager",
    # File Manager
    "FileManager",
    # Dataset Metadata
    "DatasetMetadata",
    # Dataset Registry
    "DatasetRegistry",
    # Decorators
    "timer",
    "log_execution",
    "retry",
    "validate_input",
]
