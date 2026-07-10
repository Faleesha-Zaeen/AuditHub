"""
AuditHub - Test Fixtures
=========================

Shared pytest fixtures for the AuditHub test suite.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pandas as pd
import pytest
import yaml

from src.utils.config_manager import ConfigManager
from src.utils.path_manager import PathManager


# ---------------------------------------------------------------------------
# Fixtures: Temporary directories
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project_dir() -> Generator[Path, None, None]:
    """Create a temporary project directory with configs structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create minimal config structure
        configs = tmp_path / "configs"
        configs.mkdir(parents=True, exist_ok=True)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "logs").mkdir(exist_ok=True)
        yield tmp_path


@pytest.fixture
def configs_dir(tmp_project_dir: Path) -> Path:
    """Return the configs directory in the temp project."""
    cfg_dir = tmp_project_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir


# ---------------------------------------------------------------------------
# Fixtures: Minimal config files
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Return a minimal valid config.yaml content."""
    return {
        "app": {
            "name": "AuditHub",
            "version": "0.1.0",
            "environment": "test",
            "debug": True,
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8501,
            "api_port": 8000,
            "reload": False,
        },
        "database": {
            "engine": "sqlite",
            "path": "data/test.db",
            "echo_sql": False,
        },
        "logging": {
            "level": "INFO",
            "format": "%(message)s",
            "file": "logs/test.log",
            "rotation": "midnight",
            "retention": "7 days",
        },
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "test_experiment",
            "artifact_location": "artifacts/mlflow",
        },
        "dvc": {
            "remote": "local",
            "cache_dir": "data/dvc_cache",
        },
        "streamlit": {
            "page_title": "Test",
            "page_icon": "📊",
            "layout": "wide",
        },
        "great_expectations": {
            "data_context_path": "configs/great_expectations",
            "expectation_suite_name": "test_suite",
        },
        "evidently": {
            "report_path": "reports/evidently",
        },
        "profiling": {
            "minimal_mode": False,
            "explorative": True,
            "sensitive": False,
            "pool_size": 0,
        },
    }


@pytest.fixture
def minimal_params() -> Dict[str, Any]:
    """Return a minimal valid params.yaml content."""
    return {
        "data": {
            "ingestion": {
                "chunk_size": 1000,
                "encoding": "utf-8",
                "max_file_size_mb": 100,
                "supported_formats": ["csv", "json"],
                "missing_threshold": 0.5,
            },
            "validation": {
                "dataset_min_rows": 1,
                "dataset_max_rows": 1000,
                "dataset_min_columns": 1,
                "dataset_max_columns": 100,
                "duplicate_threshold": 0.0,
            },
            "profiling": {
                "max_categories": 50,
                "correlation_threshold": 0.8,
                "sample_size": 1000,
            },
            "quality": {
                "health_score_weights": {
                    "completeness": 0.25,
                    "uniqueness": 0.15,
                    "consistency": 0.20,
                    "accuracy": 0.25,
                    "timeliness": 0.15,
                },
            },
            "repair": {
                "max_iterations": 10,
                "imputation_strategy": "mean",
                "outlier_method": "iqr",
                "outlier_threshold": 3.0,
            },
            "mutation": {
                "noise_level": 0.05,
                "corruption_fraction": 0.1,
                "missing_fraction": 0.1,
                "max_mutations": 100,
            },
            "robustness": {
                "test_split_ratio": 0.2,
                "validation_split_ratio": 0.1,
                "random_seed": 42,
                "perturbation_levels": [0.01, 0.05, 0.1],
            },
        },
        "training": {
            "model_type": "auto",
            "test_size": 0.2,
            "validation_size": 0.1,
            "random_state": 42,
            "cross_validation_folds": 5,
            "hyperparameter_tuning": False,
            "tuning_trials": 10,
            "early_stopping_rounds": 10,
        },
        "evaluation": {
            "metrics": {
                "classification": ["accuracy", "precision"],
                "regression": ["mse", "rmse"],
            },
            "report_format": "html",
        },
    }


@pytest.fixture
def minimal_schema() -> Dict[str, Any]:
    """Return a minimal valid schema.yaml content."""
    return {
        "dataset": {
            "expected_columns": [
                {"name": "id", "type": "auto", "nullable": False, "unique": True},
            ],
            "constraints": {
                "min_rows": 1,
                "max_rows": 1000,
                "max_missing_fraction": 0.5,
                "max_duplicate_fraction": 0.0,
                "allowed_file_types": [".csv", ".json"],
            },
            "column_types": {
                "integer": ["int64"],
                "float": ["float64"],
                "categorical": ["object"],
                "datetime": ["datetime64[ns]"],
            },
        },
        "profiling_schema": {
            "report_sections": ["overview", "variables"],
        },
        "health_schema": {
            "grade_ranges": {
                "A": {"min": 90, "max": 100},
                "B": {"min": 75, "max": 89},
            },
        },
    }


@pytest.fixture
def minimal_logging_config() -> Dict[str, Any]:
    """Return a minimal valid logging.yaml content."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
        },
    }


@pytest.fixture
def config_manager(tmp_project_dir: Path, configs_dir: Path,
                   minimal_config: Dict[str, Any],
                   minimal_params: Dict[str, Any],
                   minimal_schema: Dict[str, Any],
                   minimal_logging_config: Dict[str, Any]) -> Generator[ConfigManager, None, None]:
    """Create a ConfigManager with minimal config files written to disk."""
    # Reset singleton
    ConfigManager.reset_instance()

    # Write config files
    with open(configs_dir / "config.yaml", "w") as f:
        yaml.dump(minimal_config, f)
    with open(configs_dir / "params.yaml", "w") as f:
        yaml.dump(minimal_params, f)
    with open(configs_dir / "schema.yaml", "w") as f:
        yaml.dump(minimal_schema, f)
    with open(configs_dir / "logging.yaml", "w") as f:
        yaml.dump(minimal_logging_config, f)

    cm = ConfigManager(config_dir=configs_dir)
    yield cm

    # Cleanup singleton
    ConfigManager.reset_instance()


# ---------------------------------------------------------------------------
# Fixtures: DataFrames for testing
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a small DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "value": [10.5, 20.3, 30.7],
    })


@pytest.fixture
def sample_dataframe_path(tmp_project_dir: Path, sample_dataframe: pd.DataFrame) -> Path:
    """Save a sample DataFrame to a CSV file and return the path."""
    file_path = tmp_project_dir / "data" / "sample.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sample_dataframe.to_csv(file_path, index=False)
    return file_path
