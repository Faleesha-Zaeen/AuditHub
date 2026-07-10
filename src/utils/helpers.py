"""
AuditHub - Helper Utilities
============================

Reusable utility functions used across the AuditHub platform.
These helpers have **no business logic** — they provide common
file I/O, path, and data-manipulation operations.

All functions include type hints, docstrings, and logging.
"""

import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Directory & File Operations
# ============================================================================

def create_directory(path: Union[str, Path], exist_ok: bool = True) -> Path:
    """Create a directory and all parent directories if they don't exist.

    Parameters
    ----------
    path : str | Path
        Path to the directory to create.
    exist_ok : bool
        If ``True``, no error is raised if the target directory already
        exists. Defaults to ``True``.

    Returns
    -------
    Path
        The resolved ``Path`` object of the created directory.

    Examples
    --------
    >>> create_directory("data/raw")
    WindowsPath('.../data/raw')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=exist_ok)
    logger.debug("Ensured directory exists: %s", path)
    return path.resolve()


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """Alias for :func:`create_directory`."""
    return create_directory(path, exist_ok=True)


def list_files(
    directory: Union[str, Path],
    pattern: Optional[str] = None,
    extensions: Optional[List[str]] = None,
) -> List[Path]:
    """List files in a directory, optionally filtered by pattern or extension.

    Parameters
    ----------
    directory : str | Path
        Directory to search.
    pattern : str | None
        Glob pattern to filter files (e.g., ``"*.csv"``).
    extensions : list[str] | None
        List of extensions to filter (e.g., ``[".csv", ".json"]``).

    Returns
    -------
    list[Path]
        Sorted list of matching file paths.
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning("Directory does not exist: %s", directory)
        return []

    if pattern:
        files = sorted(directory.glob(pattern))
    elif extensions:
        files = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
    else:
        files = sorted(p for p in directory.iterdir() if p.is_file())

    logger.debug("Found %d files in %s", len(files), directory)
    return files


def delete_directory(path: Union[str, Path]) -> bool:
    """Recursively delete a directory.

    Parameters
    ----------
    path : str | Path
        Path to the directory to delete.

    Returns
    -------
    bool
        ``True`` if deletion was successful, ``False`` otherwise.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Directory does not exist, cannot delete: %s", path)
        return False

    try:
        shutil.rmtree(path)
        logger.info("Deleted directory: %s", path)
        return True
    except OSError as exc:
        logger.error("Failed to delete directory '%s': %s", path, exc)
        return False


def get_file_size(file_path: Union[str, Path]) -> int:
    """Return the size of a file in bytes.

    Parameters
    ----------
    file_path : str | Path
        Path to the file.

    Returns
    -------
    int
        File size in bytes. Returns ``0`` if the file doesn't exist.
    """
    path = Path(file_path)
    return path.stat().st_size if path.exists() else 0


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Return the size of a file in megabytes.

    Parameters
    ----------
    file_path : str | Path
        Path to the file.

    Returns
    -------
    float
        File size in MB.
    """
    return get_file_size(file_path) / (1024 * 1024)


# ============================================================================
# YAML Operations
# ============================================================================

def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file.

    Parameters
    ----------
    file_path : str | Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.

    Examples
    --------
    >>> config = load_yaml("configs/config.yaml")
    >>> config["app"]["name"]
    'AuditHub'
    """
    path = Path(file_path)
    logger.debug("Loading YAML from: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    logger.debug("Loaded YAML (%d top-level keys)", len(data or {}))
    return data or {}


def save_yaml(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    sort_keys: bool = False,
) -> None:
    """Save a dictionary to a YAML file.

    Parameters
    ----------
    data : dict
        Data to serialize.
    file_path : str | Path
        Destination path.
    sort_keys : bool
        Whether to sort dictionary keys. Defaults to ``False``.

    Examples
    --------
    >>> save_yaml({"name": "AuditHub"}, "output.yaml")
    """
    path = Path(file_path)
    create_directory(path.parent)

    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=sort_keys)

    logger.debug("Saved YAML to: %s", path)


# ============================================================================
# Timestamp & Identifier Utilities
# ============================================================================

def generate_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Generate a formatted timestamp string.

    Parameters
    ----------
    format_str : str
        ``strftime`` format string. Defaults to ``"%Y%m%d_%H%M%S"``.

    Returns
    -------
    str
        Formatted timestamp string.

    Examples
    --------
    >>> generate_timestamp()
    '20260710_143022'
    """
    return datetime.now(timezone.utc).strftime(format_str)


def generate_uuid() -> str:
    """Generate a UUID4 string.

    Returns
    -------
    str
        A random UUID string (e.g. ``"a1b2c3d4-..."``).
    """
    return str(uuid.uuid4())


def generate_run_id() -> str:
    """Generate a human-readable run identifier.

    Returns
    -------
    str
        Identifier combining a timestamp and short UUID.

    Examples
    --------
    >>> generate_run_id()
    'run_20260710_143022_a1b2c3d4'
    """
    short_id = uuid.uuid4().hex[:8]
    return f"run_{generate_timestamp()}_{short_id}"


# ============================================================================
# Filename & Path Utilities
# ============================================================================

def safe_filename(filename: str, replacement: str = "_") -> str:
    """Sanitize a filename by removing or replacing unsafe characters.

    Removes characters that are illegal or unsafe in filenames across
    common operating systems.

    Parameters
    ----------
    filename : str
        Raw filename to sanitize.
    replacement : str
        Character to replace unsafe characters with. Defaults to ``"_"``.

    Returns
    -------
    str
        Sanitized filename.

    Examples
    --------
    >>> safe_filename("my/file:name?.csv")
    'my_file_name_.csv'
    >>> safe_filename("../data/raw/dataset.csv")
    '.._data_raw_dataset.csv'
    """
    # Characters unsafe across Windows, Linux, macOS
    unsafe_chars = '<>:"/\\|?*'

    result = "".join(
        replacement if c in unsafe_chars else c
        for c in filename
    )

    # Remove leading/trailing dots and spaces
    result = result.strip(". ")

    logger.debug("Sanitized filename: '%s' -> '%s'", filename, result)
    return result


def unique_filename(filename: Union[str, Path]) -> str:
    """Append a timestamp to a filename to make it unique.

    Parameters
    ----------
    filename : str | Path
        Original filename.

    Returns
    -------
    str
        Unique filename with timestamp inserted before the extension.

    Examples
    --------
    >>> unique_filename("report.html")
    'report_20260710_143022.html'
    """
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    return f"{stem}_{generate_timestamp()}{suffix}"


def ensure_extension(filename: str, extension: str) -> str:
    """Ensure a filename has the expected extension.

    Parameters
    ----------
    filename : str
        Filename to check.
    extension : str
        Desired extension (with or without leading dot).

    Returns
    -------
    str
        Filename with extension appended if it wasn't already present.

    Examples
    --------
    >>> ensure_extension("report", ".csv")
    'report.csv'
    >>> ensure_extension("data.csv", ".csv")
    'data.csv'
    """
    ext = extension if extension.startswith(".") else f".{extension}"
    return filename if filename.endswith(ext) else f"{filename}{ext}"


# ============================================================================
# Time Utilities
# ============================================================================

def timer() -> Callable[[], float]:
    """Create a simple stopwatch timer.

    Returns
    -------
    callable
        A function that returns the elapsed time in seconds since the
        timer was created.

    Examples
    --------
    >>> stopwatch = timer()
    >>> ...  # do some work
    >>> elapsed = stopwatch()
    >>> print(f"Elapsed: {elapsed:.3f}s")
    """
    start = time.perf_counter()

    def _elapsed() -> float:
        return time.perf_counter() - start

    return _elapsed


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string like ``"1h 23m 45s"`` or ``"45.2s"``.

    Examples
    --------
    >>> format_duration(3661)
    '1h 1m 1s'
    >>> format_duration(45.2)
    '45.2s'
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not hours and not minutes:
        parts.append(f"{seconds:.1f}s")
    else:
        parts.append(f"{secs}s")

    return " ".join(parts)


# ============================================================================
# Miscellaneous
# ============================================================================

def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of equal size.

    Parameters
    ----------
    items : list
        The list to split.
    chunk_size : int
        Maximum size of each chunk.

    Returns
    -------
    list[list]
        List of chunks.

    Examples
    --------
    >>> chunk_list([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4], [5]]
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dictionaries. Values from *override* take precedence.

    Parameters
    ----------
    base : dict
        Base dictionary.
    override : dict
        Dictionary whose values take precedence.

    Returns
    -------
    dict
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get an environment variable with a fallback default.

    Parameters
    ----------
    key : str
        Environment variable name.
    default : str | None
        Default value if the variable is not set.

    Returns
    -------
    str | None
        Environment variable value or default.
    """
    return os.environ.get(key, default)
