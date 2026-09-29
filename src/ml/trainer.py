"""
AuditHub ML - Trainer
======================

Automatically infers task types, trains multiple model candidates, preprocesses features,
selects the best model, and logs everything to MLflow.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.ml.evaluator import MLEvaluator
from src.ml.tracker import MLflowTracker
from src.utils.app_settings import AppSettings
from src.utils.constants import MODELS_DIR
from src.utils.logger import get_logger
from src.utils.helpers import ensure_directory_exists, unique_filename

logger = get_logger(__name__)

# Nanoseconds per day, used to express datetimes as a plain numeric feature.
_NS_PER_DAY = 86_400_000_000_000


def encode_datetime_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Convert any datetime column into days since the epoch.

    Dates carry real ordinal signal (recency, seasonality), so they are far
    more useful as a number than as one dummy column per distinct day.

    This lives inside the fitted pipeline rather than being applied to the
    training frame beforehand. Doing it beforehand produced a model that only
    worked on pre-converted input: ``model.predict(raw_df)`` raised
    ``TypeError: Cannot cast DatetimeArray to dtype float64``, which broke
    prediction, explainability and robustness alike. Defined at module level so
    joblib can pickle the pipeline.
    """
    X = X.copy()
    for column in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[column].dtype):
            X[column] = X[column].astype("int64").where(X[column].notna()) / _NS_PER_DAY
    return X


class MLTrainingEngine:
    """Manages automated ML training runs, comparison, and experiment tracking."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the MLTrainingEngine.

        Parameters
        ----------
        seed : int
            Random seed for model split and algorithms.
        """
        self.seed = seed
        self.evaluator = MLEvaluator()
        self.tracker = MLflowTracker()
        ensure_directory_exists(MODELS_DIR)

    def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_name: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any], str]:
        """Auto-infer task, pre-process, train multiple models, select best, log to MLflow, and return.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset for training.
        target_column : str
            Name of target column.
        dataset_name : str | None
            Name of dataset for run logging.

        Returns
        -------
        tuple[Any, dict, str]
            Best trained pipeline, its metrics, and local save path.
        """
        logger.info("Starting Auto-ML training on target '%s'", target_column)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        # 1. Infer task type
        target_series = df[target_column].dropna()
        nunique_target = target_series.nunique()
        is_classification = (
            pd.api.types.is_string_dtype(target_series.dtype) or
            pd.api.types.is_bool_dtype(target_series.dtype) or
            (pd.api.types.is_integer_dtype(target_series.dtype) and nunique_target <= 10)
        )
        task_type = "classification" if is_classification else "regression"
        logger.info("Inferred ML task type: %s", task_type)

        # Drop rows with null target values
        df_clean = df.dropna(subset=[target_column]).copy()
        
        # 2. Split features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        # Categorical columns / Numeric columns identification.
        # A column of numbers stored as text (one stray "?" in the source file
        # is enough) would otherwise be one-hot encoded into hundreds of dummy
        # columns instead of being treated as the number it is.
        numeric_features: List[str] = []
        categorical_features: List[str] = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col].dtype):
                numeric_features.append(col)
                continue

            # Datetimes are converted by the pipeline's first step, so they are
            # numeric by the time the ColumnTransformer sees them. They must be
            # intercepted before pd.to_numeric, which raises on a DatetimeArray.
            if pd.api.types.is_datetime64_any_dtype(X[col].dtype):
                numeric_features.append(col)
                logger.info("Encoding datetime column '%s' as days since epoch.", col)
                continue

            coerced = pd.to_numeric(X[col], errors="coerce")
            observed = int(X[col].notna().sum())
            if observed and coerced.notna().sum() / observed >= 0.9:
                X[col] = coerced
                numeric_features.append(col)
                logger.info("Treating text column '%s' as numeric for training.", col)
            else:
                categorical_features.append(col)

        # Nullable extension dtypes (Int64) are not accepted by every sklearn
        # transformer; convert them to plain float now that gaps are imputed
        # downstream by SimpleImputer anyway.
        for col in numeric_features:
            if isinstance(X[col].dtype, pd.api.extensions.ExtensionDtype):
                X[col] = X[col].astype("float64")

        # 3. Train-Test Split (80% train, 20% test).
        # Stratify classification targets so rare classes appear in both
        # splits; without it an imbalanced dataset can produce a test set
        # missing a class entirely and a meaningless F1.
        stratify = None
        if task_type == "classification":
            class_counts = y.value_counts()
            if len(class_counts) > 1 and int(class_counts.min()) >= 2:
                stratify = y
            else:
                logger.warning(
                    "Not stratifying: the rarest class has %d member(s).",
                    int(class_counts.min()) if len(class_counts) else 0,
                )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=stratify
        )

        # 4. Define candidate models
        candidates: List[Tuple[str, Any]] = []
        if task_type == "classification":
            candidates = [
                ("LogisticRegression", LogisticRegression(max_iter=500, random_state=self.seed)),
                ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=self.seed)),
                ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=100, random_state=self.seed))
            ]
            selection_metric = "f1_score"
            higher_is_better = True
        else:
            candidates = [
                ("RidgeRegression", Ridge(random_state=self.seed)),
                ("RandomForestRegressor", RandomForestRegressor(n_estimators=100, random_state=self.seed)),
                ("GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=100, random_state=self.seed))
            ]
            selection_metric = "rmse"
            higher_is_better = False

        best_score = -float("inf") if higher_is_better else float("inf")
        best_pipeline = None
        best_metrics: Dict[str, Any] = {}
        best_model_name = ""
        best_plots: Dict[str, str] = {}

        # 5. Train and evaluate each candidate
        for model_name, model in candidates:
            logger.info("Training candidate: %s", model_name)

            # Preprocessing pipelines
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ])

            # Complete pipeline. The datetime step comes first so the fitted
            # model accepts the same raw frame the user supplied.
            pipeline = Pipeline(steps=[
                ("datetime_encoder", FunctionTransformer(
                    encode_datetime_columns, feature_names_out="one-to-one"
                )),
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            # Fit model
            pipeline.fit(X_train, y_train)

            # Evaluate
            metrics, plots = self.evaluator.evaluate(
                pipeline, X_test, y_test, task_type=task_type, feature_names=X.columns.tolist()
            )

            score = metrics.get(selection_metric, 0.0)
            logger.info("Candidate %s score (%s): %.4f", model_name, selection_metric, score)

            # Selection logic
            is_better = (score > best_score) if higher_is_better else (score < best_score)
            if is_better or best_pipeline is None:
                best_score = score
                best_pipeline = pipeline
                best_metrics = metrics
                best_model_name = model_name
                best_plots = plots

        # 6. Save the best model
        name_str = dataset_name or "dataset"
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name_str)
        model_filename = unique_filename(f"{safe_name}_{best_model_name.lower()}.joblib")
        model_path = MODELS_DIR / model_filename
        
        logger.info("Saving best model (%s) to %s", best_model_name, model_path)
        joblib.dump(best_pipeline, model_path)

        # 7. Log to MLflow
        run_params = {
            "dataset_name": name_str,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "task_type": task_type,
            "best_algorithm": best_model_name,
            "random_seed": self.seed,
        }
        
        self.tracker.log_run(
            model=best_pipeline,
            model_name=best_model_name,
            params=run_params,
            metrics=best_metrics,
            artifacts=best_plots
        )

        return best_pipeline, best_metrics, str(model_path.resolve())
