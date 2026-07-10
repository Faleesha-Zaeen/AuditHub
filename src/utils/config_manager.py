"""
AuditHub - Configuration Manager
=================================

Singleton-based centralized configuration system for AuditHub.
Loads, caches, validates, and provides access to all YAML configuration
files. Supports environment variable overrides.

Every future module shall obtain configuration through this manager.
No module shall hardcode paths, filenames, thresholds, or any other
configurable value.

Usage::

    from src.utils.config_manager import ConfigManager

    cm = ConfigManager()
    config = cm.get_config()
    params = cm.get_params()
    schema = cm.get_schema()
    logging_cfg = cm.get_logging()
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from src.utils.exceptions import ConfigurationException
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Required keys that must exist in each config file
# ---------------------------------------------------------------------------
_REQUIRED_CONFIG_KEYS: Tuple[str, ...] = (
    "app.name",
    "app.version",
    "app.environment",
    "server.port",
    "database.engine",
)

_REQUIRED_PARAMS_KEYS: Tuple[str, ...] = (
    "data.ingestion",
    "data.validation",
    "training.test_size",
)

_REQUIRED_SCHEMA_KEYS: Tuple[str, ...] = (
    "dataset.constraints",
    "dataset.column_types",
)

_REQUIRED_LOGGING_KEYS: Tuple[str, ...] = (
    "version",
    "handlers",
)

# ---------------------------------------------------------------------------
# Environment variable mapping: (env_var_name, dot_separated_config_path)
# ---------------------------------------------------------------------------
_ENV_OVERRIDE_MAP: Dict[str, str] = {
    "APP_ENV": "app.environment",
    "DATABASE_URL": "database.path",
    "MLFLOW_TRACKING_URI": "mlflow.tracking_uri",
    "DVC_REMOTE": "dvc.remote",
    "LOG_LEVEL": "logging.level",
    "STREAMLIT_SERVER_PORT": "server.port",
}

# ---------------------------------------------------------------------------
# Default config file paths (relative to project root)
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path("configs")


# ============================================================================
# ConfigManager
# ============================================================================


class ConfigManager:
    """Singleton configuration manager for AuditHub.

    Loads YAML configuration files from ``configs/``, caches them
    in memory, validates required keys, and applies environment variable
    overrides.

    Parameters
    ----------
    config_dir : str | Path | None
        Directory containing config files. Defaults to ``configs/``
        relative to the working directory.
    """

    _instance: Optional["ConfigManager"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._config_dir: Path = Path(config_dir) if config_dir else _CONFIG_DIR
        self._config: Optional[Dict[str, Any]] = None
        self._params: Optional[Dict[str, Any]] = None
        self._schema: Optional[Dict[str, Any]] = None
        self._logging_config: Optional[Dict[str, Any]] = None
        self._initialized: bool = True

        logger.debug("ConfigManager initialized with config_dir=%s", self._config_dir)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Load and return the main application configuration.

        Returns
        -------
        dict
            Application configuration with env var overrides applied.

        Raises
        ------
        ConfigurationException
            If the file is missing, invalid, or missing required keys.
        """
        if self._config is None:
            raw = self._load_and_validate(
                filename="config.yaml",
                required_keys=_REQUIRED_CONFIG_KEYS,
            )
            self._config = self._apply_env_overrides(raw)
        return copy.deepcopy(self._config)

    def get_params(self) -> Dict[str, Any]:
        """Load and return the tunable parameters configuration.

        Returns
        -------
        dict
            Parameters dictionary.

        Raises
        ------
        ConfigurationException
            If the file is missing, invalid, or missing required keys.
        """
        if self._params is None:
            self._params = self._load_and_validate(
                filename="params.yaml",
                required_keys=_REQUIRED_PARAMS_KEYS,
            )
        return dict(self._params)

    def get_schema(self) -> Dict[str, Any]:
        """Load and return the schema definitions.

        Returns
        -------
        dict
            Schema dictionary.

        Raises
        ------
        ConfigurationException
            If the file is missing, invalid, or missing required keys.
        """
        if self._schema is None:
            self._schema = self._load_and_validate(
                filename="schema.yaml",
                required_keys=_REQUIRED_SCHEMA_KEYS,
            )
        return dict(self._schema)

    def get_logging(self) -> Dict[str, Any]:
        """Load and return the logging configuration.

        Returns
        -------
        dict
            Logging configuration dictionary (compatible with
            ``logging.config.dictConfig``).

        Raises
        ------
        ConfigurationException
            If the file is missing, invalid, or missing required keys.
        """
        if self._logging_config is None:
            self._logging_config = self._load_and_validate(
                filename="logging.yaml",
                required_keys=_REQUIRED_LOGGING_KEYS,
            )
        return dict(self._logging_config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: str) -> Path:
        """Resolve the full path to a config file.

        Parameters
        ----------
        filename : str
            Config file name (e.g., ``"config.yaml"``).

        Returns
        -------
        Path
            Resolved path.

        Raises
        ------
        ConfigurationException
            If the config directory doesn't exist.
        """
        config_dir = self._config_dir
        if not config_dir.exists():
            raise ConfigurationException(
                f"Configuration directory not found: {config_dir.resolve()}",
            )
        return config_dir / filename

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML file and return its content.

        Parameters
        ----------
        file_path : Path
            Path to the YAML file.

        Returns
        -------
        dict
            Parsed YAML content.

        Raises
        ------
        ConfigurationException
            If the file doesn't exist or contains invalid YAML.
        """
        if not file_path.exists():
            raise ConfigurationException(
                f"Configuration file not found: {file_path.resolve()}",
            )

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigurationException(
                f"Invalid YAML in configuration file: {file_path.resolve()}",
                details={"yaml_error": str(exc)},
            ) from exc

        if not isinstance(data, dict):
            raise ConfigurationException(
                f"Configuration file must contain a top-level mapping: {file_path.resolve()}",
            )

        return data

    def _validate_required_keys(
        self,
        data: Dict[str, Any],
        required_keys: Tuple[str, ...],
        file_path: Path,
    ) -> None:
        """Validate that all required dot-separated keys exist in the data.

        Parameters
        ----------
        data : dict
            Parsed configuration data.
        required_keys : tuple[str, ...]
            Dot-separated key paths that must exist.
        file_path : Path
            Path to the config file (for error messages).

        Raises
        ------
        ConfigurationException
            If any required key is missing.
        """
        missing: list[str] = []
        invalid: list[str] = []

        for key_path in required_keys:
            parts = key_path.split(".")
            current: Any = data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    missing.append(key_path)
                    break
            else:
                # Key exists; check value is not None
                if current is None:
                    invalid.append(key_path)

        if missing or invalid:
            messages = []
            if missing:
                messages.append(f"Missing keys: {', '.join(missing)}")
            if invalid:
                messages.append(f"Invalid (null) keys: {', '.join(invalid)}")
            raise ConfigurationException(
                f"Validation failed for {file_path.resolve()}: {'; '.join(messages)}",
                details={
                    "missing_keys": missing,
                    "invalid_keys": invalid,
                    "file": str(file_path),
                },
            )

    def _load_and_validate(
        self,
        filename: str,
        required_keys: Tuple[str, ...],
    ) -> Dict[str, Any]:
        """Load a YAML file and validate its required keys.

        Parameters
        ----------
        filename : str
            Config file name.
        required_keys : tuple[str, ...]
            Required dot-separated key paths.

        Returns
        -------
        dict
            Validated configuration data.
        """
        file_path = self._resolve_path(filename)
        logger.debug("Loading config file: %s", file_path)
        data = self._load_yaml_file(file_path)
        self._validate_required_keys(data, required_keys, file_path)
        logger.info("Loaded and validated config: %s", filename)
        return data

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override YAML config values with environment variables.

        Parameters
        ----------
        config : dict
            Base config dictionary.

        Returns
        -------
        dict
            Config with env var overrides applied.
        """
        result = copy.deepcopy(config)

        for env_var, config_path in _ENV_OVERRIDE_MAP.items():
            env_value = os.environ.get(env_var)
            if env_value is None:
                continue

            parts = config_path.split(".")
            current = result

            # Navigate to the parent of the target key
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value
            key = parts[-1]
            old_value = current.get(key)
            current[key] = self._coerce_type(env_value, old_value)

            logger.debug(
                "Env override: %s -> config.%s = %s (was %s)",
                env_var, config_path, current[key], old_value,
            )

        return result

    @staticmethod
    def _coerce_type(value: str, existing: Any) -> Any:
        """Coerce an environment variable string to match the existing value's type.

        Parameters
        ----------
        value : str
            Environment variable value (always a string).
        existing : any
            Existing config value to infer the target type from.

        Returns
        -------
        any
            Coerced value.
        """
        if existing is None:
            return value

        target_type = type(existing)

        if target_type is bool:
            return value.lower() in ("true", "1", "yes", "on")
        if target_type is int:
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        if target_type is float:
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if target_type is list:
            return [item.strip() for item in value.split(",") if item.strip()]

        return value

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Clear all caches so configs are reloaded on next access."""
        self._config = None
        self._params = None
        self._schema = None
        self._logging_config = None
        logger.info("ConfigManager caches cleared")

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def config_dir(self) -> Path:
        """Return the resolved config directory path."""
        return self._config_dir.resolve()


__all__ = [
    "ConfigManager",
]
