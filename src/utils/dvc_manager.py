"""
AuditHub DVC - DVC Integration Manager
========================================

Manages data and model versioning programmatically using DVC.
Falls back to simulated metadata versioning if DVC/SCM is not initialized.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.utils.logger import get_logger
from src.utils.helpers import get_file_size, generate_timestamp

logger = get_logger(__name__)


class DVCManager:
    """Helper manager to track, version, and manage dataset snapshots via DVC."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        """Initialize the DVCManager.

        Parameters
        ----------
        root_dir : Path | None
            Root directory of the project. Defaults to PROJECT_ROOT.
        """
        from src.utils.constants import PROJECT_ROOT
        self.root_dir = root_dir or PROJECT_ROOT
        self._dvc_repo = None
        self._is_enabled = False

        self._initialize_dvc()

    def _initialize_dvc(self) -> None:
        """Attempt to initialize or hook into DVC repository."""
        try:
            from dvc.repo import Repo
            from dvc.exceptions import NotDvcRepoError

            # Verify if .dvc exists
            dvc_path = self.root_dir / ".dvc"
            if not dvc_path.exists():
                logger.info("DVC is not initialized. Initializing dynamic DVC repo without SCM.")
                self._dvc_repo = Repo.init(root_dir=str(self.root_dir), no_scm=True)
            else:
                self._dvc_repo = Repo(root_dir=str(self.root_dir))
            
            self._is_enabled = True
            logger.info("DVC integration successfully enabled.")
        except Exception as exc:
            logger.warning(
                "DVC python package or repo initialization failed: %s. "
                "AuditHub will fallback to simulated file versioning.", exc
            )
            self._is_enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if DVC versioning is active."""
        return self._is_enabled

    def track_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Track a dataset or model file in DVC.

        Parameters
        ----------
        file_path : str | Path
            The file to version.

        Returns
        -------
        dict
            DVC versioning metadata (version, checksum, file size).
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found to track: {path}")

        logger.info("Tracking file via DVC: %s", path.name)

        if self._is_enabled and self._dvc_repo:
            try:
                # Add to DVC repo
                self._dvc_repo.add(str(path))
                logger.info("Successfully added %s to DVC.", path.name)

                # Read checksum from the generated .dvc file
                dvc_meta_path = path.with_suffix(path.suffix + ".dvc")
                checksum = "unknown"
                if dvc_meta_path.exists():
                    import yaml
                    with open(dvc_meta_path, "r") as f:
                        meta = yaml.safe_load(f)
                        outs = meta.get("outs", [])
                        if outs:
                            checksum = outs[0].get("md5", "unknown")

                return {
                    "status": "success",
                    "engine": "dvc",
                    "file": path.name,
                    "checksum": checksum,
                    "version_label": f"v_{generate_timestamp()}",
                    "size_bytes": get_file_size(path),
                }
            except Exception as exc:
                logger.error("DVC add command failed: %s. Falling back to simulation.", exc)

        # Fallback simulated versioning
        import hashlib
        h = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    h.update(chunk)
            checksum = h.hexdigest()
        except Exception:
            checksum = f"sim-{generate_timestamp()}"

        return {
            "status": "success",
            "engine": "simulated",
            "file": path.name,
            "checksum": checksum,
            "version_label": f"v_{generate_timestamp()}",
            "size_bytes": get_file_size(path),
        }
