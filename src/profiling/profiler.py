"""
AuditHub Profiling - Dataset Profiler
======================================

Generates comprehensive data profiling reports using ydata-profiling.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from ydata_profiling import ProfileReport

from src.utils.app_settings import AppSettings
from src.utils.constants import REPORTS_DIR
from src.utils.logger import get_logger
from src.utils.helpers import ensure_directory_exists, unique_filename

logger = get_logger(__name__)


class DatasetProfiler:
    """Wraps ydata-profiling to generate rich html profiles and summary stats.

    Configuration is loaded dynamically from AppSettings.
    """

    def __init__(self) -> None:
        """Initialize the DatasetProfiler with configured settings."""
        settings = AppSettings.load()
        self.minimal = settings.profiling_minimal
        self.explorative = settings.profiling_explorative
        self.sensitive = settings.profiling_sensitive
        self.pool_size = settings.profiling_pool_size

        logger.debug(
            "DatasetProfiler initialized with minimal=%s, explorative=%s, sensitive=%s, pool_size=%d",
            self.minimal, self.explorative, self.sensitive, self.pool_size
        )

    def profile(
        self,
        df: pd.DataFrame,
        dataset_name: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate a profile report and save it as HTML.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to profile.
        dataset_name : str | None
            Name of the dataset (for file naming and titles).
        output_dir : Path | None
            Custom directory to save the HTML report. Defaults to REPORTS_DIR.

        Returns
        -------
        dict
            Dictionary containing aggregated profiling statistics.
        """
        name = dataset_name or "dataset"
        logger.info("Profiling dataset: %s (shape: %s)", name, df.shape)

        # 1. Create ydata-profiling report
        report = ProfileReport(
            df,
            title=f"AuditHub Profile Report - {name}",
            minimal=self.minimal,
            explorative=self.explorative,
            sensitive=self.sensitive,
            pool_size=self.pool_size,
        )

        # 2. Save HTML report
        out_dir = output_dir or REPORTS_DIR
        ensure_directory_exists(out_dir)
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        filename = unique_filename(f"{safe_name}_profile.html")
        report_path = out_dir / filename
        
        logger.info("Saving profiling report to %s", report_path)
        report.to_file(report_path)

        # 3. Extract and simplify summary description
        logger.debug("Extracting profile description dict.")
        desc = report.get_description()

        # Simplify statistics to return a JSON-serializable dictionary
        if isinstance(desc, dict):
            table_stats = desc.get("table", {})
            variables = desc.get("variables", {})
        else:
            table_stats = getattr(desc, "table", {})
            variables = getattr(desc, "variables", {})

        # Convert table_stats to dict if it is an object
        if not isinstance(table_stats, dict):
            try:
                table_stats = dict(table_stats)
            except Exception:
                table_stats = {
                    "n": getattr(table_stats, "n", 0),
                    "n_var": getattr(table_stats, "n_var", 0),
                    "memory_size": getattr(table_stats, "memory_size", 0),
                    "n_duplicates": getattr(table_stats, "n_duplicates", 0),
                }

        simplified_vars = {}
        if not isinstance(variables, dict):
            try:
                variables = dict(variables)
            except Exception:
                pass

        for col, meta in variables.items():
            if isinstance(meta, dict):
                simplified_vars[str(col)] = {
                    "type": str(meta.get("type", "unknown")),
                    "n_distinct": int(meta.get("n_distinct", 0)),
                    "p_distinct": float(meta.get("p_distinct", 0.0)),
                    "n_missing": int(meta.get("n_missing", 0)),
                    "p_missing": float(meta.get("p_missing", 0.0)),
                    "is_unique": bool(meta.get("is_unique", False)),
                    "memory_size": int(meta.get("memory_size", 0)),
                }
            else:
                simplified_vars[str(col)] = {
                    "type": str(getattr(meta, "type", "unknown")),
                    "n_distinct": int(getattr(meta, "n_distinct", 0)),
                    "p_distinct": float(getattr(meta, "p_distinct", 0.0)),
                    "n_missing": int(getattr(meta, "n_missing", 0)),
                    "p_missing": float(getattr(meta, "p_missing", 0.0)),
                    "is_unique": bool(getattr(meta, "is_unique", False)),
                    "memory_size": int(getattr(meta, "memory_size", 0)),
                }

        summary = {
            "dataset_name": name,
            "report_path": str(report_path.resolve()),
            "n_records": int(table_stats.get("n", 0)),
            "n_features": int(table_stats.get("n_var", 0)),
            "memory_size": int(table_stats.get("memory_size", 0)),
            "record_size_avg": float(table_stats.get("record_size_avg", 0.0)),
            "p_missing": float(table_stats.get("p_missing", 0.0)),
            "n_duplicates": int(table_stats.get("n_duplicates", 0)),
            "p_duplicates": float(table_stats.get("p_duplicates", 0.0)),
            "variables": simplified_vars,
        }

        logger.info("Profiling complete for dataset: %s", name)
        return summary
