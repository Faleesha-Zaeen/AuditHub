"""
AuditHub ML - Evaluator
========================

Computes model performance metrics and generates visualization plots (confusion matrix, feature importance).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.utils.constants import REPORTS_DIR
from src.utils.logger import get_logger
from src.utils.helpers import ensure_directory_exists, unique_filename

logger = get_logger(__name__)


class MLEvaluator:
    """Computes evaluation metrics and generates plots for classification and regression tasks."""

    def __init__(self) -> None:
        """Initialize the MLEvaluator."""
        ensure_directory_exists(REPORTS_DIR)

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Compute metrics and save plots.

        Parameters
        ----------
        model : Any
            Trained model.
        X_test : pd.DataFrame
            Features set.
        y_test : pd.Series
            True target values.
        task_type : str
            'classification' or 'regression'.
        feature_names : list[str] | None
            Names of the features.

        Returns
        -------
        tuple[dict, dict]
            Metrics dictionary and a dictionary mapping plot names to local file paths.
        """
        logger.info("Evaluating model on test set. Task type: %s", task_type)
        preds = model.predict(X_test)
        metrics: Dict[str, Any] = {}
        plots: Dict[str, str] = {}

        if task_type == "classification":
            # 1. Compute classification metrics
            metrics["accuracy"] = float(accuracy_score(y_test, preds))
            # Binary or multiclass metrics handling
            unique_classes = np.unique(y_test)
            is_binary = len(unique_classes) == 2

            if is_binary:
                # sklearn's binary metrics default to pos_label=1, which raises
                # outright on string labels such as 'yes'/'no'. Classes come
                # back sorted, so the second one is the positive class -- that
                # keeps the familiar behaviour for 0/1 and False/True targets.
                pos_label = unique_classes[1]
                metrics["positive_label"] = str(pos_label)
                metrics["precision"] = float(
                    precision_score(y_test, preds, pos_label=pos_label, zero_division=0)
                )
                metrics["recall"] = float(
                    recall_score(y_test, preds, pos_label=pos_label, zero_division=0)
                )
                metrics["f1_score"] = float(
                    f1_score(y_test, preds, pos_label=pos_label, zero_division=0)
                )
                try:
                    # predict_proba columns follow model.classes_, which is
                    # sorted the same way, so column 1 is pos_label's score.
                    probs = model.predict_proba(X_test)[:, 1]
                    # roc_auc_score has no pos_label argument, so express the
                    # truth as a 0/1 indicator for the positive class.
                    y_binary = (np.asarray(y_test) == pos_label).astype(int)
                    metrics["roc_auc"] = float(roc_auc_score(y_binary, probs))
                except Exception as exc:
                    logger.warning("Could not compute ROC AUC: %s", exc)
                    metrics["roc_auc"] = 0.5
            else:
                metrics["precision"] = float(precision_score(y_test, preds, average="weighted", zero_division=0))
                metrics["recall"] = float(recall_score(y_test, preds, average="weighted", zero_division=0))
                metrics["f1_score"] = float(f1_score(y_test, preds, average="weighted", zero_division=0))
                metrics["roc_auc"] = 0.5  # Default for multi-class simple evaluation

            # 2. Confusion Matrix Plot
            try:
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=unique_classes, yticklabels=unique_classes)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                fig.tight_layout()
                
                plot_filename = unique_filename("confusion_matrix.png")
                plot_path = REPORTS_DIR / plot_filename
                fig.savefig(plot_path)
                plt.close(fig)
                plots["confusion_matrix"] = str(plot_path.resolve())
            except Exception as exc:
                logger.warning("Failed to generate confusion matrix plot: %s", exc)

        else:
            # 1. Compute regression metrics
            metrics["mae"] = float(mean_absolute_error(y_test, preds))
            mse = mean_squared_error(y_test, preds)
            metrics["rmse"] = float(np.sqrt(mse))
            metrics["r2_score"] = float(r2_score(y_test, preds))
            
            # 2. Residuals Plot
            try:
                residuals = y_test - preds
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(preds, residuals, alpha=0.5)
                ax.axhline(0, color="r", linestyle="--")
                ax.set_title("Residuals vs Predicted")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residuals")
                fig.tight_layout()
                
                plot_filename = unique_filename("residuals_plot.png")
                plot_path = REPORTS_DIR / plot_filename
                fig.savefig(plot_path)
                plt.close(fig)
                plots["residuals_plot"] = str(plot_path.resolve())
            except Exception as exc:
                logger.warning("Failed to generate residuals plot: %s", exc)

        # 3. Feature Importance Plot (If model supports it)
        feats = feature_names or (X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else [])
        importances = None
        
        # Check if pipeline or direct model
        base_model = model
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            base_model = model.named_steps["model"]

        if hasattr(base_model, "feature_importances_"):
            importances = base_model.feature_importances_
        elif hasattr(base_model, "coef_"):
            importances = np.abs(base_model.coef_)
            if len(importances.shape) > 1:
                importances = importances[0]  # Take first class coefs for multi-class

        if importances is not None and len(importances) == len(feats) and len(feats) > 0:
            try:
                indices = np.argsort(importances)[::-1]
                top_indices = indices[:10]  # Top 10 features
                
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.barh(np.array(feats)[top_indices][::-1], importances[top_indices][::-1], color="teal")
                ax.set_title("Feature Importances (Top 10)")
                ax.set_xlabel("Importance Score")
                fig.tight_layout()

                plot_filename = unique_filename("feature_importances.png")
                plot_path = REPORTS_DIR / plot_filename
                fig.savefig(plot_path)
                plt.close(fig)
                plots["feature_importance"] = str(plot_path.resolve())
                
                # Store numerical importances in metrics
                metrics["feature_importances"] = {feats[i]: float(importances[i]) for i in indices}
            except Exception as exc:
                logger.warning("Failed to generate feature importance plot: %s", exc)

        return metrics, plots
