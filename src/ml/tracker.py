"""
AuditHub ML - MLflow Tracker
=============================

Provides standardized logging and tracking of model parameters, metrics, plots, and models to MLflow.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import mlflow
import mlflow.sklearn
# Used when filtering metrics down to loggable scalars. Its absence raised
# NameError inside the try block, so every run was created but left empty.
import numpy as np

from src.utils.app_settings import AppSettings
from src.utils.constants import (
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_MLFLOW_TRACKING_URI,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """Manages MLflow session tracking for AuditHub model training and evaluations."""

    def __init__(self) -> None:
        """Initialize the MLflowTracker and configure tracking settings.

        Experiment tracking is optional infrastructure: an unreadable or
        misconfigured config file must degrade to the built-in defaults rather
        than take training down with it.
        """
        try:
            settings = AppSettings.load()
            self.tracking_uri = settings.mlflow_tracking_uri
            self.experiment_name = settings.mlflow_experiment
        except Exception as exc:
            logger.warning(
                "Could not read MLflow settings (%s); using defaults.", exc
            )
            self.tracking_uri = DEFAULT_MLFLOW_TRACKING_URI
            self.experiment_name = DEFAULT_MLFLOW_EXPERIMENT

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow configured: tracking_uri=%s, experiment=%s", self.tracking_uri, self.experiment_name)
        except Exception as exc:
            logger.warning("Failed to configure MLflow tracking: %s. Runs will not be tracked.", exc)

    def log_run(
        self,
        model: Any,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        artifacts: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Log a complete training run to MLflow.

        Parameters
        ----------
        model : Any
            The trained scikit-learn model/pipeline.
        model_name : str
            A name for the model (e.g. 'RandomForestClassifier').
        params : dict
            Hyperparameters or data parameters to log.
        metrics : dict
            Evaluation metrics to log.
        artifacts : dict[str, str] | None
            Dictionary mapping artifact names to local file paths (e.g., plot image paths).

        Returns
        -------
        str | None
            The MLflow run ID if successful, or None.
        """
        try:
            with mlflow.start_run(run_name=model_name) as run:
                run_id = run.info.run_id
                logger.info("Logging to MLflow. Run ID: %s", run_id)

                # 1. Log parameters
                mlflow.log_params(params)

                # 2. Log metrics
                # Convert any non-numeric metrics or dicts (like confusion matrix)
                scalar_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.number))}
                mlflow.log_metrics(scalar_metrics)

                # 3. Log artifacts (plots, matrices)
                if artifacts:
                    for name, path in artifacts.items():
                        if Path(path).exists():
                            mlflow.log_artifact(path, artifact_path="plots")

                # 4. Log the model
                # MLflow 3.x serialises sklearn models with skops, which
                # refuses anything it does not recognise -- including the
                # pipeline's own datetime-encoding step. These pipelines are
                # produced and consumed by this application, so pickle is the
                # appropriate format; skops' trust model is aimed at models
                # received from third parties.
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                )
                
                logger.info("Successfully logged model and metrics to MLflow.")
                return run_id
        except Exception as exc:
            logger.error("Failed to log run to MLflow: %s", exc)
            return None
