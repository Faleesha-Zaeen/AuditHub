"""
AuditHub - Application Settings
================================

Strongly typed, validated application settings dataclass.
Aggregates configuration values from ``ConfigManager`` into a single
typed object.

No module should access raw dictionaries from ConfigManager directly;
they should use ``AppSettings`` instead.

Usage::

    from src.utils.app_settings import AppSettings

    settings = AppSettings.load()
    print(settings.app_name)
    print(settings.db_path)
    print(settings.mlflow_tracking_uri)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.utils.config_manager import ConfigManager
from src.utils.constants import DEFAULT_MLFLOW_TRACKING_URI
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# AppSettings
# ============================================================================


@dataclass(frozen=True)
class AppSettings:
    """Strongly typed application-wide settings.

    This is the single source of truth for all configuration values.
    Instantiate via :meth:`AppSettings.load()`.

    Attributes
    ----------
    app_name : str
        Application name (default: ``"AuditHub"``).
    app_version : str
        Application version (default: ``"0.1.0"``).
    environment : str
        Runtime environment (``"development"``, ``"staging"``, ``"production"``).
    debug : bool
        Enable debug mode.
    host : str
        Server bind host (default: ``"0.0.0.0"``).
    port : int
        Streamlit server port (default: ``8501``).
    api_port : int
        FastAPI backend port (default: ``8000``).
    db_engine : str
        Database engine type (default: ``"sqlite"``).
    db_path : Path
        Path to the database file.
    db_echo_sql : bool
        Whether to echo SQL queries.
    log_level : str
        Logging level (default: ``"INFO"``).
    log_file : Path
        Path to the log file.
    log_rotation : str
        Log rotation schedule (default: ``"midnight"``).
    log_retention : str
        Log retention period (default: ``"30 days"``).
    mlflow_tracking_uri : str
        MLflow tracking URI.
    mlflow_experiment : str
        Default MLflow experiment name.
    mlflow_artifact_location : str
        Default MLflow artifact location.
    dvc_remote : str
        Default DVC remote name.
    dvc_cache_dir : Path
        DVC cache directory path.
    streamlit_page_title : str
        Streamlit page title.
    streamlit_page_icon : str
        Streamlit page icon.
    streamlit_layout : str
        Streamlit layout (``"wide"`` or ``"centered"``).
    profiling_minimal : bool
        Enable minimal profiling mode.
    profiling_explorative : bool
        Enable explorative profiling.
    profiling_sensitive : bool
        Enable sensitive data detection.
    profiling_pool_size : int
        Profiling pool size for parallelization.
    great_expectations_context_path : str
        Great Expectations data context path.
    great_expectations_suite : str
        Default expectation suite name.
    evidently_report_path : Path
        Evidently AI report output path.
    """

    # Application
    app_name: str = "AuditHub"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8501
    api_port: int = 8000

    # Database
    db_engine: str = "sqlite"
    db_path: Path = field(default_factory=lambda: Path("data/audithub.db"))
    db_echo_sql: bool = False

    # Logging
    log_level: str = "INFO"
    log_file: Path = field(default_factory=lambda: Path("logs/audithub.log"))
    log_rotation: str = "midnight"
    log_retention: str = "30 days"

    # MLflow
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI
    mlflow_experiment: str = "audithub_default"
    mlflow_artifact_location: str = "artifacts/mlflow"

    # DVC
    dvc_remote: str = "local"
    dvc_cache_dir: Path = field(default_factory=lambda: Path("data/dvc_cache"))

    # Streamlit
    streamlit_page_title: str = "AuditHub - Dataset Quality & MLOps Platform"
    streamlit_page_icon: str = "📊"
    streamlit_layout: str = "wide"

    # Profiling
    profiling_minimal: bool = False
    profiling_explorative: bool = True
    profiling_sensitive: bool = False
    profiling_pool_size: int = 0

    # Great Expectations
    great_expectations_context_path: str = "configs/great_expectations"
    great_expectations_suite: str = "default_suite"

    # Evidently
    evidently_report_path: Path = field(default_factory=lambda: Path("reports/evidently"))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, config_dir: Optional[Path] = None) -> "AppSettings":
        """Load application settings from configuration.

        Reads all config YAML files via ``ConfigManager`` and populates
        a strongly typed ``AppSettings`` instance.

        Parameters
        ----------
        config_dir : Path | None
            Optional custom config directory.

        Returns
        -------
        AppSettings
            Fully populated settings instance.
        """
        cm = ConfigManager(config_dir=config_dir)
        config = cm.get_config()
        logging_cfg = cm.get_logging()

        # Helper to safely traverse nested dicts
        _sentinel = object()

        def _get(d: dict, *keys: str, default: any = None) -> any:  # type: ignore[no-untyped-def]
            current = d
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key, _sentinel)
                    if current is _sentinel:
                        return default
                else:
                    return default
            return current

        settings = cls(
            # Application
            app_name=_get(config, "app", "name", default="AuditHub"),
            app_version=_get(config, "app", "version", default="0.1.0"),
            environment=_get(config, "app", "environment", default="development"),
            debug=_get(config, "app", "debug", default=True),
            # Server
            host=_get(config, "server", "host", default="0.0.0.0"),
            port=_get(config, "server", "port", default=8501),
            api_port=_get(config, "server", "api_port", default=8000),
            # Database
            db_engine=_get(config, "database", "engine", default="sqlite"),
            db_path=Path(_get(config, "database", "path", default="data/audithub.db")),
            db_echo_sql=_get(config, "database", "echo_sql", default=False),
            # Logging
            log_level=_get(logging_cfg, "handlers", "console", "level", default="INFO"),
            log_file=Path(_get(config, "logging", "file", default="logs/audithub.log")),
            log_rotation=_get(config, "logging", "rotation", default="midnight"),
            log_retention=_get(config, "logging", "retention", default="30 days"),
            # MLflow
            mlflow_tracking_uri=_get(
                config, "mlflow", "tracking_uri",
                default=DEFAULT_MLFLOW_TRACKING_URI,
            ),
            mlflow_experiment=_get(config, "mlflow", "experiment_name", default="audithub_default"),
            mlflow_artifact_location=_get(config, "mlflow", "artifact_location", default="artifacts/mlflow"),
            # DVC
            dvc_remote=_get(config, "dvc", "remote", default="local"),
            dvc_cache_dir=Path(_get(config, "dvc", "cache_dir", default="data/dvc_cache")),
            # Streamlit
            streamlit_page_title=_get(config, "streamlit", "page_title", default="AuditHub - Dataset Quality & MLOps Platform"),
            streamlit_page_icon=_get(config, "streamlit", "page_icon", default="📊"),
            streamlit_layout=_get(config, "streamlit", "layout", default="wide"),
            # Profiling
            profiling_minimal=_get(config, "profiling", "minimal_mode", default=False),
            profiling_explorative=_get(config, "profiling", "explorative", default=True),
            profiling_sensitive=_get(config, "profiling", "sensitive", default=False),
            profiling_pool_size=_get(config, "profiling", "pool_size", default=0),
            # Great Expectations
            great_expectations_context_path=_get(config, "great_expectations", "data_context_path", default="configs/great_expectations"),
            great_expectations_suite=_get(config, "great_expectations", "expectation_suite_name", default="default_suite"),
            # Evidently
            evidently_report_path=Path(_get(config, "evidently", "report_path", default="reports/evidently")),
        )

        logger.debug("AppSettings loaded: %s", settings.app_name)
        return settings

    def to_dict(self) -> dict:
        """Convert settings to a flat dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation of all settings.
        """
        result = {}
        for field_name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(self, field_name)
            if isinstance(value, Path):
                value = str(value)
            result[field_name] = value
        return result


__all__ = [
    "AppSettings",
]
