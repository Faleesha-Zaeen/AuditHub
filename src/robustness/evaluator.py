"""
AuditHub Robustness - Robustness Evaluation Engine
===================================================

Evaluates machine learning model decay under increasing dataset corruption.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.mutation_lab.mutator import DatasetMutator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RobustnessEvaluator:
    """Trains a baseline model and evaluates its degradation on mutated datasets."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the RobustnessEvaluator.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        """
        self.seed = seed
        self.mutator = DatasetMutator(seed=seed)

    def evaluate_robustness(
        self,
        df: pd.DataFrame,
        target_column: str,
        mutation_type: str,
        levels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Evaluate performance degradation of a model under various mutation levels.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to evaluate.
        target_column : str
            Target column name.
        mutation_type : str
            Type of mutation to apply: 'missing_values', 'gaussian_noise', 'outliers', 'duplicates'.
        levels : list[float] | None
            The mutation levels to evaluate. Defaults to [0.0, 0.05, 0.10, 0.25, 0.50].

        Returns
        -------
        dict
            Evaluation results containing metric decay, baseline, and elasticity score.
        """
        logger.info("Evaluating robustness to '%s' on target '%s'", mutation_type, target_column)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        # Default levels
        if levels is None:
            levels = [0.0, 0.05, 0.10, 0.25, 0.50]
        levels = sorted(list(set(levels)))
        if 0.0 not in levels:
            levels.insert(0, 0.0)

        # 1. Determine task type (Classification vs Regression)
        target_series = df[target_column].dropna()
        nunique_target = target_series.nunique()
        is_classification = (
            pd.api.types.is_string_dtype(target_series.dtype) or
            pd.api.types.is_bool_dtype(target_series.dtype) or
            (pd.api.types.is_integer_dtype(target_series.dtype) and nunique_target <= 10)
        )
        task_type = "classification" if is_classification else "regression"
        logger.info("Inferred ML task type: %s", task_type)

        # 2. Simple tabular preprocessing
        # Identify features and target
        features = [c for c in df.columns if c != target_column]
        df_clean = df.copy()

        # Handle NaNs in target
        df_clean = df_clean.dropna(subset=[target_column])

        # Fill missing values and encode categoricals for model training.
        # Numeric columns are cast to float first: a nullable Int64 column
        # rejects a fractional mean outright, and the models want floats anyway.
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col].dtype):
                as_float = df_clean[col].astype("float64")
                mean_val = as_float.mean()
                df_clean[col] = as_float.fillna(mean_val if not pd.isna(mean_val) else 0.0)
            else:
                df_clean[col] = df_clean[col].astype("object").fillna("missing")
                df_clean[col] = df_clean[col].astype("category").cat.codes

        X = df_clean[features]
        y = df_clean[target_column]

        # 3. Train-Test Split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        # 4. Instantiate and train baseline model
        if is_classification:
            model = RandomForestClassifier(n_estimators=50, random_state=self.seed)
            metric_name = "accuracy"
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=self.seed)
            metric_name = "rmse"

        model.fit(X_train, y_train)

        # 5. Iteratively evaluate on mutated test sets
        performances = []
        for level in levels:
            if level == 0.0:
                X_test_mut = X_test
            else:
                # Apply mutation to features only
                if mutation_type == "missing_values":
                    X_test_mut = self.mutator.inject_missing_values(X_test, fraction=level)
                elif mutation_type == "gaussian_noise":
                    X_test_mut = self.mutator.inject_gaussian_noise(X_test, noise_level=level)
                elif mutation_type == "outliers":
                    X_test_mut = self.mutator.inject_outliers(X_test, fraction=level)
                elif mutation_type == "duplicates":
                    # Duplicating rows in test set doesn't change model performance drastically,
                    # but we can mutate it or use other mutation types.
                    # Let's apply random corruption to features instead
                    X_test_mut = self.mutator.inject_random_corruption(X_test, fraction=level)
                else:
                    X_test_mut = X_test

            # Impute missing values in test set before predicting so model doesn't crash
            X_test_eval = X_test_mut.copy()
            for col in X_test_eval.columns:
                if X_test_eval[col].isna().any():
                    mean_val = X_train[col].mean()
                    X_test_eval[col] = X_test_eval[col].fillna(mean_val if not pd.isna(mean_val) else 0)

            # Evaluate model
            preds = model.predict(X_test_eval)

            if is_classification:
                perf = float(accuracy_score(y_test, preds))
            else:
                perf = float(np.sqrt(mean_squared_error(y_test, preds)))
            
            performances.append(perf)

        # 6. Compute Elasticity Score
        # Elasticity is the area under the normalized performance curve.
        # For accuracy, normalized_perf = perf / baseline_perf
        # For rmse, normalized_perf = baseline_perf / max(perf, baseline_perf) (smaller is better, so if rmse increases, normalized_perf decreases)
        baseline_perf = performances[0]
        normalized_perfs = []
        for perf in performances:
            if is_classification:
                norm_val = perf / baseline_perf if baseline_perf > 0 else 0.0
            else:
                # RMSE: if error increases, score decays
                norm_val = baseline_perf / perf if perf > 0 else 0.0
            normalized_perfs.append(min(1.0, norm_val))

        # Integrate curve area (Trapezoidal rule)
        auc = 0.0
        for i in range(len(levels) - 1):
            h = levels[i+1] - levels[i]
            y_avg = (normalized_perfs[i] + normalized_perfs[i+1]) / 2.0
            auc += h * y_avg
        
        # Divide by max levels range to bound between 0 and 1
        max_range = levels[-1] - levels[0]
        elasticity_score = auc / max_range if max_range > 0 else 1.0

        results = {
            "task_type": task_type,
            "mutation_type": mutation_type,
            "metric_name": metric_name,
            "baseline_performance": round(baseline_perf, 4),
            "levels": levels,
            "performances": [round(p, 4) for p in performances],
            "normalized_performances": [round(p, 4) for p in normalized_perfs],
            "elasticity_score": round(elasticity_score, 4),
        }

        logger.info("Robustness evaluation complete. Elasticity: %.4f", elasticity_score)
        return results
