"""
AuditHub - Path Manager
========================

Resolves project paths for all data stages, artifact directories,
logs, reports, and models. Automatically creates missing directories
on first access.

Usage::

    from src.utils.path_manager import PathManager

    pm = PathManager()
    raw_path = pm.raw_data_dir("my_dataset.csv")
    report_path = pm.reports_dir("report.html")
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# PathManager
# ============================================================================


class PathManager:
    """Resolve and manage project directory paths.

    Automatically creates directories when they are first accessed.
    All paths are relative to the project root.

    Parameters
    ----------
    project_root : Path | None
        Override the project root path. If ``None``, auto-detects by
        walking up from the source file location.
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self._root: Path = (
            project_root.resolve()
            if project_root
            else self._detect_project_root()
        )
        self._created_dirs: set = set()
        logger.debug("PathManager initialized with root: %s", self._root)

    # ------------------------------------------------------------------
    # Project root
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Return the resolved project root path."""
        return self._root

    # ------------------------------------------------------------------
    # Data directories
    # ------------------------------------------------------------------

    @property
    def raw_data_dir(self) -> Path:
        """Path to raw/uploaded datasets."""
        return self._ensure(self._root / "data" / "raw")

    @property
    def validated_data_dir(self) -> Path:
        """Path to validated datasets."""
        return self._ensure(self._root / "data" / "validated")

    @property
    def repaired_data_dir(self) -> Path:
        """Path to repaired datasets."""
        return self._ensure(self._root / "data" / "repaired")

    @property
    def mutated_data_dir(self) -> Path:
        """Path to mutated datasets."""
        return self._ensure(self._root / "data" / "mutated")

    @property
    def processed_data_dir(self) -> Path:
        """Path to processed/feature-engineered datasets."""
        return self._ensure(self._root / "data" / "processed")

    @property
    def reports_data_dir(self) -> Path:
        """Path to data-related reports."""
        return self._ensure(self._root / "data" / "reports")

    # ------------------------------------------------------------------
    # Artifact & model directories
    # ------------------------------------------------------------------

    @property
    def artifacts_dir(self) -> Path:
        """Path to ML artifacts."""
        return self._ensure(self._root / "artifacts")

    @property
    def models_dir(self) -> Path:
        """Path to trained models."""
        return self._ensure(self._root / "models")

    @property
    def reports_dir(self) -> Path:
        """Path to generated reports."""
        return self._ensure(self._root / "reports")

    @property
    def logs_dir(self) -> Path:
        """Path to log files."""
        return self._ensure(self._root / "logs")

    @property
    def database_dir(self) -> Path:
        """Path to the database file's parent directory."""
        return self._ensure(self._root / "data")

    # ------------------------------------------------------------------
    # Config directory
    # ------------------------------------------------------------------

    @property
    def config_dir(self) -> Path:
        """Path to configuration files."""
        return self._ensure(self._root / "configs")

    # ------------------------------------------------------------------
    # Context-specific paths
    # ------------------------------------------------------------------

    def database_path(self, db_name: str = "audithub.db") -> Path:
        """Return the full path to a SQLite database file.

        Parameters
        ----------
        db_name : str
            Database filename (default: ``"audithub.db"``).

        Returns
        -------
        Path
            Full path to the database file.
        """
        return self.database_dir / db_name

    def raw_data_file(self, filename: str) -> Path:
        """Return the full path to a file in the raw data directory.

        Parameters
        ----------
        filename : str
            Name of the data file.

        Returns
        -------
        Path
            Full path to the file.
        """
        return self.raw_data_dir / filename

    def validated_data_file(self, filename: str) -> Path:
        """Return the full path to a file in the validated data directory.

        Parameters
        ----------
        filename : str
            Name of the data file.

        Returns
        -------
        Path
            Full path to the file.
        """
        return self.validated_data_dir / filename

    def repaired_data_file(self, filename: str) -> Path:
        """Return the full path to a file in the repaired data directory.

        Parameters
        ----------
        filename : str
            Name of the data file.

        Returns
        -------
        Path
            Full path to the file.
        """
        return self.repaired_data_dir / filename

    def model_file(self, filename: str) -> Path:
        """Return the full path to a model file.

        Parameters
        ----------
        filename : str
            Name of the model file.

        Returns
        -------
        Path
            Full path to the model file.
        """
        return self.models_dir / filename

    def report_file(self, filename: str) -> Path:
        """Return the full path to a report file.

        Parameters
        ----------
        filename : str
            Name of the report file.

        Returns
        -------
        Path
            Full path to the report file.
        """
        return self.reports_dir / filename

    def log_file(self, filename: str = "audithub.log") -> Path:
        """Return the full path to a log file.

        Parameters
        ----------
        filename : str
            Name of the log file (default: ``"audithub.log"``).

        Returns
        -------
        Path
            Full path to the log file.
        """
        return self.logs_dir / filename

    def artifact_file(self, filename: str) -> Path:
        """Return the full path to an artifact file.

        Parameters
        ----------
        filename : str
            Name of the artifact file.

        Returns
        -------
        Path
            Full path to the artifact file.
        """
        return self.artifacts_dir / filename

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_raw_files(self, pattern: str = "*") -> List[Path]:
        """List files in the raw data directory.

        Parameters
        ----------
        pattern : str
            Glob pattern to filter by (default: ``"*"`` for all files).

        Returns
        -------
        list[Path]
            Sorted list of matching file paths.
        """
        return sorted(self.raw_data_dir.glob(pattern))

    # ------------------------------------------------------------------
    # Checksum utility
    # ------------------------------------------------------------------

    @staticmethod
    def compute_checksum(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
        """Compute a file checksum.

        Parameters
        ----------
        file_path : str | Path
            Path to the file.
        algorithm : str
            Hash algorithm (``"sha256"``, ``"md5"``, etc.).

        Returns
        -------
        str
            Hex digest of the file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found for checksum: {path}")

        hasher = hashlib.new(algorithm)
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure(self, path: Path) -> Path:
        """Ensure a directory exists and return it.

        Parameters
        ----------
        path : Path
            Directory path to ensure exists.

        Returns
        -------
        Path
            The (now-existing) directory path.
        """
        if path not in self._created_dirs:
            path.mkdir(parents=True, exist_ok=True)
            self._created_dirs.add(path)
            logger.debug("Ensured directory: %s", path)
        return path

    @staticmethod
    def _detect_project_root() -> Path:
        """Auto-detect the project root by walking up the directory tree.

        Looks for a ``configs`` directory as a marker of the project root.

        Returns
        -------
        Path
            Detected project root path.
        """
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "configs").exists() or (parent / ".git").exists():
                return parent
        # Fallback to cwd
        return Path.cwd().resolve()

    def __repr__(self) -> str:
        return f"PathManager(root={self._root})"


__all__ = [
    "PathManager",
]
