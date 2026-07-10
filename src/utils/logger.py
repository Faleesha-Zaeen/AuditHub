"""
AuditHub - Centralized Logging System
======================================

Provides a unified logging interface for all AuditHub modules.
Supports both file-based (with rotation) and console logging.
Reads configuration from ``configs/logging.yaml`` or falls back to
sensible defaults.

Typical usage::

    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("An error occurred", exc_info=True)
"""

import logging
import logging.config
import logging.handlers
import os
from pathlib import Path
from typing import Optional, Union

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_LOGGING_CONFIG_PATH = Path("configs/logging.yaml")
_DEFAULT_LOG_DIR = Path("logs")
_DEFAULT_LOG_LEVEL = logging.INFO

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_loggers_configured: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_logger(
    name: str,
    level: Optional[Union[int, str]] = None,
) -> logging.Logger:
    """Return a configured logger for the given *name*.

    Parameters
    ----------
    name : str
        Logger name — conventionally ``__name__`` from the calling module.
    level : int | str | None
        Optional override for the logger level. If *None* the level
        defined in the logging configuration is used.

    Returns
    -------
    logging.Logger
        A ready-to-use logger instance.

    Examples
    --------
    >>> logger = get_logger("audithub.ingestion")
    >>> logger.info("Hello from ingestion")
    """
    _ensure_logging_configured()

    logger = logging.getLogger(name)

    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), _DEFAULT_LOG_LEVEL)
        logger.setLevel(level)

    return logger


def set_log_level(level: Union[str, int]) -> None:
    """Globally change the log level for all AuditHub loggers.

    Parameters
    ----------
    level : str | int
        Either a string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, ``"CRITICAL"``) or a ``logging`` level constant.

    Examples
    --------
    >>> set_log_level("DEBUG")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), _DEFAULT_LOG_LEVEL)

    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("audithub"):
            logging.getLogger(logger_name).setLevel(level)


def add_file_handler(
    logger_name: str,
    file_path: Union[str, Path],
    level: Union[str, int] = logging.DEBUG,
    fmt: Optional[str] = None,
) -> logging.Handler:
    """Add a file handler to a specific logger at runtime.

    Parameters
    ----------
    logger_name : str
        Name of the logger to attach the handler to.
    file_path : str | Path
        Path to the log file.
    level : str | int
        Log level for this handler.
    fmt : str | None
        Optional custom format string.

    Returns
    -------
    logging.Handler
        The newly created handler.
    """
    logger = logging.getLogger(logger_name)

    handler = logging.FileHandler(str(file_path))
    handler.setLevel(level if isinstance(level, int) else getattr(logging, level.upper(), _DEFAULT_LOG_LEVEL))

    formatter = logging.Formatter(
        fmt or "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return handler


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _ensure_logging_configured() -> None:
    """Configure logging once on first use."""
    global _loggers_configured

    if _loggers_configured:
        return

    config_path = _resolve_config_path()
    if config_path and config_path.exists():
        _configure_from_yaml(config_path)
    else:
        _configure_default()

    _loggers_configured = True


def _resolve_config_path() -> Optional[Path]:
    """Try to find the logging configuration file."""
    # Check environment variable first
    env_path = os.getenv("AUDITHUB_LOGGING_CONFIG")
    if env_path:
        return Path(env_path)

    # Check default path
    if _DEFAULT_LOGGING_CONFIG_PATH.exists():
        return _DEFAULT_LOGGING_CONFIG_PATH

    return None


def _configure_from_yaml(config_path: Path) -> None:
    """Load logging configuration from a YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        # Ensure log directory exists
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.config.dictConfig(config)
    except Exception as exc:
        # Fall back to default if YAML config fails
        _configure_default()
        logging.getLogger(__name__).warning(
            "Failed to load logging config from '%s': %s. Using defaults.",
            config_path,
            exc,
        )


def _configure_default() -> None:
    """Configure logging with sensible defaults."""
    _DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(_DEFAULT_LOG_DIR / "audithub.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Error file handler
    error_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(_DEFAULT_LOG_DIR / "audithub_error.log"),
        when="midnight",
        interval=1,
        backupCount=90,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
