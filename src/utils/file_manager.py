"""
AuditHub - File Manager
========================

Reusable file I/O operations for DataFrames, JSON, and pickle files.
Supports multiple formats (CSV, Excel, JSON, Parquet) with automatic
directory creation and backup support.

Usage::

    from src.utils.file_manager import FileManager
    import pandas as pd

    fm = FileManager()
    df = pd.DataFrame({"a": [1, 2, 3]})

    # Save and load DataFrame
    fm.save_dataframe(df, "data/raw/my_data.csv")
    df_loaded = fm.load_dataframe("data/raw/my_data.csv")

    # JSON
    fm.save_json({"key": "value"}, "data/output.json")
    data = fm.load_json("data/output.json")

    # Pickle
    fm.save_pickle({"obj": "value"}, "data/obj.pkl")
    obj = fm.load_pickle("data/obj.pkl")
"""

import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.utils.exceptions import DatasetException
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Supported formats for DataFrame I/O
# ---------------------------------------------------------------------------
_SUPPORTED_DATAFRAME_FORMATS = frozenset({".csv", ".tsv", ".json", ".parquet", ".xlsx", ".xls", ".feather", ".pkl", ".pickle"})


# ============================================================================
# FileManager
# ============================================================================


class FileManager:
    """Reusable file I/O manager.

    Provides static methods for saving and loading DataFrames, JSON,
    pickle files, and backups.

    Parameters
    ----------
    base_dir : str | Path | None
        Optional base directory. If provided, all relative paths are
        resolved relative to this directory.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None) -> None:
        self._base_dir: Optional[Path] = Path(base_dir).resolve() if base_dir else None
        logger.debug("FileManager initialized with base_dir=%s", self._base_dir)

    # ------------------------------------------------------------------
    # DataFrame I/O
    # ------------------------------------------------------------------

    def save_dataframe(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        index: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Save a pandas DataFrame to disk.

        Automatically determines format from the file extension.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save.
        file_path : str | Path
            Destination path (extension determines format).
        index : bool
            Whether to write row indices (default: ``False``).
        **kwargs
            Additional keyword arguments passed to the writer function.

        Returns
        -------
        Path
            Resolved path where the file was saved.

        Raises
        ------
        DatasetException
            If the file format is unsupported or saving fails.
        """
        path = self._resolve(file_path)
        self._ensure_parent(path)
        ext = path.suffix.lower()

        if ext not in _SUPPORTED_DATAFRAME_FORMATS:
            raise DatasetException(
                f"Unsupported DataFrame format: '{ext}'. "
                f"Supported: {sorted(_SUPPORTED_DATAFRAME_FORMATS)}",
            )

        try:
            if ext == ".csv":
                df.to_csv(path, index=index, **kwargs)
            elif ext == ".tsv":
                df.to_csv(path, sep="\t", index=index, **kwargs)
            elif ext == ".json":
                df.to_json(path, orient="records", index=index, **kwargs)
            elif ext in (".xlsx", ".xls"):
                df.to_excel(path, index=index, **kwargs)
            elif ext == ".parquet":
                df.to_parquet(path, index=index, **kwargs)
            elif ext in (".pkl", ".pickle"):
                df.to_pickle(path, **kwargs)
            elif ext == ".feather":
                df.to_feather(path, **kwargs)

            logger.info("Saved DataFrame (%d rows, %d cols) to: %s", len(df), len(df.columns), path)
            return path
        except Exception as exc:
            raise DatasetException(
                f"Failed to save DataFrame to: {path}",
                details={"format": ext, "error": str(exc)},
                cause=exc,
            )

    def load_dataframe(
        self,
        file_path: Union[str, Path],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load a pandas DataFrame from disk.

        Automatically determines format from the file extension.

        Parameters
        ----------
        file_path : str | Path
            Source file path.
        **kwargs
            Additional keyword arguments passed to the reader function.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.

        Raises
        ------
        DatasetException
            If the file format is unsupported, the file doesn't exist,
            or loading fails.
        """
        path = self._resolve(file_path)

        if not path.exists():
            raise DatasetException(f"File not found: {path}")

        ext = path.suffix.lower()

        if ext not in _SUPPORTED_DATAFRAME_FORMATS:
            raise DatasetException(
                f"Unsupported DataFrame format: '{ext}'. "
                f"Supported: {sorted(_SUPPORTED_DATAFRAME_FORMATS)}",
            )

        try:
            if ext == ".csv":
                df = pd.read_csv(path, **kwargs)
            elif ext == ".tsv":
                df = pd.read_csv(path, sep="\t", **kwargs)
            elif ext == ".json":
                df = pd.read_json(path, **kwargs)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(path, **kwargs)
            elif ext == ".parquet":
                df = pd.read_parquet(path, **kwargs)
            elif ext in (".pkl", ".pickle"):
                df = pd.read_pickle(path)  # type: ignore[arg-type]
            elif ext == ".feather":
                df = pd.read_feather(path)  # type: ignore[arg-type]
            else:
                raise DatasetException(f"Unsupported format: {ext}")

            logger.info("Loaded DataFrame (%d rows, %d cols) from: %s", len(df), len(df.columns), path)
            return df
        except Exception as exc:
            raise DatasetException(
                f"Failed to load DataFrame from: {path}",
                details={"format": ext, "error": str(exc)},
                cause=exc,
            )

    # ------------------------------------------------------------------
    # JSON I/O
    # ------------------------------------------------------------------

    def save_json(
        self,
        data: Any,
        file_path: Union[str, Path],
        indent: int = 2,
        **kwargs: Any,
    ) -> Path:
        """Save data as JSON.

        Parameters
        ----------
        data : any
            Data to serialize (must be JSON-serializable).
        file_path : str | Path
            Destination path.
        indent : int
            JSON indentation level (default: ``2``).
        **kwargs
            Additional keyword arguments passed to ``json.dump``.

        Returns
        -------
        Path
            Resolved path where the file was saved.
        """
        path = self._resolve(file_path)
        self._ensure_parent(path)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, **kwargs)

        logger.info("Saved JSON to: %s", path)
        return path

    def load_json(
        self,
        file_path: Union[str, Path],
        **kwargs: Any,
    ) -> Any:
        """Load data from a JSON file.

        Parameters
        ----------
        file_path : str | Path
            Source file path.
        **kwargs
            Additional keyword arguments passed to ``json.load``.

        Returns
        -------
        any
            Deserialized data.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = self._resolve(file_path)

        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh, **kwargs)

        logger.debug("Loaded JSON from: %s", path)
        return data

    # ------------------------------------------------------------------
    # Pickle I/O
    # ------------------------------------------------------------------

    def save_pickle(
        self,
        obj: Any,
        file_path: Union[str, Path],
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        """Serialize an object using pickle.

        Parameters
        ----------
        obj : any
            Object to pickle.
        file_path : str | Path
            Destination path.
        protocol : int
            Pickle protocol version (default: highest available).

        Returns
        -------
        Path
            Resolved path where the file was saved.
        """
        path = self._resolve(file_path)
        self._ensure_parent(path)

        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=protocol)

        logger.info("Saved pickle to: %s", path)
        return path

    def load_pickle(self, file_path: Union[str, Path]) -> Any:
        """Deserialize a pickle file.

        Parameters
        ----------
        file_path : str | Path
            Source file path.

        Returns
        -------
        any
            Deserialized object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = self._resolve(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")

        with open(path, "rb") as fh:
            obj = pickle.load(fh)

        logger.debug("Loaded pickle from: %s", path)
        return obj

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def file_exists(self, file_path: Union[str, Path]) -> bool:
        """Check if a file exists.

        Parameters
        ----------
        file_path : str | Path
            Path to check.

        Returns
        -------
        bool
            ``True`` if the file exists and is a file.
        """
        return self._resolve(file_path).is_file()

    def safe_delete(self, file_path: Union[str, Path]) -> bool:
        """Safely delete a file if it exists.

        Parameters
        ----------
        file_path : str | Path
            Path to the file to delete.

        Returns
        -------
        bool
            ``True`` if deletion was successful, ``False`` if file
            didn't exist or deletion failed.
        """
        path = self._resolve(file_path)

        if not path.exists():
            logger.warning("File not found, cannot delete: %s", path)
            return False

        try:
            path.unlink()
            logger.info("Deleted file: %s", path)
            return True
        except OSError as exc:
            logger.error("Failed to delete file '%s': %s", path, exc)
            return False

    def create_backup(self, file_path: Union[str, Path]) -> Optional[Path]:
        """Create a timestamped backup of a file.

        Parameters
        ----------
        file_path : str | Path
            Path to the file to back up.

        Returns
        -------
        Path | None
            Path to the backup file, or ``None`` if the source file
            doesn't exist.
        """
        path = self._resolve(file_path)

        if not path.exists():
            logger.warning("File not found, cannot create backup: %s", path)
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

        try:
            shutil.copy2(path, backup_path)
            logger.info("Created backup: %s -> %s", path, backup_path)
            return backup_path
        except OSError as exc:
            logger.error("Failed to create backup of '%s': %s", path, exc)
            return None

    def get_size(self, file_path: Union[str, Path]) -> int:
        """Get the size of a file in bytes.

        Parameters
        ----------
        file_path : str | Path
            Path to the file.

        Returns
        -------
        int
            File size in bytes, or ``0`` if the file doesn't exist.
        """
        path = self._resolve(file_path)
        return path.stat().st_size if path.exists() else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: Union[str, Path]) -> Path:
        """Resolve a path relative to the base directory if set.

        Parameters
        ----------
        path : str | Path
            Input path.

        Returns
        -------
        Path
            Resolved absolute path.
        """
        p = Path(path)
        if not p.is_absolute() and self._base_dir:
            p = self._base_dir / p
        return p.resolve()

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        """Ensure the parent directory of a file exists.

        Parameters
        ----------
        path : Path
            File path whose parent should be created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"FileManager(base_dir={self._base_dir})"


__all__ = [
    "FileManager",
]
