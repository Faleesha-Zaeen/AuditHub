"""
AuditHub ML - Model Explainability
===================================

Explains what a trained model learned, and why it made a particular
prediction.

Everything here operates on the **fitted pipeline**, not the bare estimator,
so explanations reflect the exact preprocessing used during training --
imputation, scaling and one-hot encoding included. Two consequences follow and
are handled explicitly:

* Global importance uses permutation importance over the raw input frame. It
  is model-agnostic, and because the whole pipeline is permuted it measures the
  original column rather than an encoded fragment of it.
* SHAP runs against the *transformed* matrix, so its output is per encoded
  feature (``cat__city_Delhi``). Those are summed back into the original column
  (``city``) before being shown -- a user asked "why was this flagged" wants to
  read ``city``, not fourteen dummy columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Rows sampled as the SHAP background distribution. Kernel SHAP is O(background
# x features) per explained row, so this is capped to keep it interactive.
_MAX_BACKGROUND = 100

# Permutation importance repeats. Enough to be stable without being slow.
_DEFAULT_REPEATS = 10


@dataclass
class FeatureImportance:
    """One feature's global contribution to the model."""

    feature: str
    importance: float
    std: float = 0.0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return the entry as a JSON-serialisable dictionary."""
        return {
            "feature": self.feature,
            "importance": self.importance,
            "std": self.std,
            "rank": self.rank,
        }


@dataclass
class FeatureContribution:
    """One feature's push on a single prediction."""

    feature: str
    value: Any
    contribution: float

    @property
    def direction(self) -> str:
        """``increases`` or ``decreases``, relative to the explained outcome."""
        return "increases" if self.contribution >= 0 else "decreases"

    def to_dict(self) -> Dict[str, Any]:
        """Return the contribution as a JSON-serialisable dictionary."""
        return {
            "feature": self.feature,
            "value": _jsonable(self.value),
            "contribution": self.contribution,
            "direction": self.direction,
        }


@dataclass
class LocalExplanation:
    """Why the model predicted what it did for one row."""

    row_index: int
    prediction: Any
    confidence: Optional[float] = None
    base_value: Optional[float] = None
    contributions: List[FeatureContribution] = field(default_factory=list)
    task_type: str = "classification"

    def pushing_toward(self) -> List[FeatureContribution]:
        """Features arguing for the predicted outcome, strongest first."""
        return sorted(
            [c for c in self.contributions if c.contribution > 0],
            key=lambda c: -c.contribution,
        )

    def pushing_against(self) -> List[FeatureContribution]:
        """Features arguing against the predicted outcome, strongest first."""
        return sorted(
            [c for c in self.contributions if c.contribution < 0],
            key=lambda c: c.contribution,
        )

    def narrative(self, limit: int = 3) -> str:
        """Render the explanation as a short readable paragraph."""
        lines = [f"Prediction: {self.prediction}"]
        if self.confidence is not None:
            lines[0] += f" (confidence {self.confidence:.1%})"

        toward = self.pushing_toward()[:limit]
        against = self.pushing_against()[:limit]
        if toward:
            lines.append("Top contributors:")
            for rank, contribution in enumerate(toward, start=1):
                lines.append(
                    f"  {rank}. {contribution.feature} = {_short(contribution.value)}"
                    f" -> pushed toward this outcome"
                )
        if against:
            lines.append("Pushing against:")
            for rank, contribution in enumerate(against, start=1):
                lines.append(
                    f"  {rank}. {contribution.feature} = {_short(contribution.value)}"
                    f" -> pushed away from this outcome"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return the explanation as a JSON-serialisable dictionary."""
        return {
            "row_index": self.row_index,
            "prediction": _jsonable(self.prediction),
            "confidence": self.confidence,
            "base_value": self.base_value,
            "task_type": self.task_type,
            "contributions": [c.to_dict() for c in self.contributions],
            "narrative": self.narrative(),
        }

    def to_frame(self) -> pd.DataFrame:
        """Return the contributions as a DataFrame for charting."""
        if not self.contributions:
            return pd.DataFrame(columns=["feature", "value", "contribution", "direction"])
        return pd.DataFrame([c.to_dict() for c in self.contributions])


@dataclass
class ExplanationReport:
    """Global and local explanations for one trained model."""

    model_name: str = ""
    task_type: str = "classification"
    method: str = ""
    global_importance: List[FeatureImportance] = field(default_factory=list)
    local_explanations: List[LocalExplanation] = field(default_factory=list)
    target_column: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def top_features(self, limit: int = 10) -> List[FeatureImportance]:
        """Return the most important features."""
        return self.global_importance[:limit]

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        if not self.global_importance:
            return "No global importance could be computed."
        top = ", ".join(f.feature for f in self.global_importance[:3])
        return (
            f"{self.model_name or 'Model'} relies most on: {top}. "
            f"{len(self.local_explanations)} individual prediction(s) explained."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "method": self.method,
            "target_column": self.target_column,
            "global_importance": [f.to_dict() for f in self.global_importance],
            "local_explanations": [e.to_dict() for e in self.local_explanations],
            "summary": self.summary(),
            "timestamp": self.timestamp,
        }

    def importance_frame(self) -> pd.DataFrame:
        """Return global importance as a DataFrame for charting."""
        if not self.global_importance:
            return pd.DataFrame(columns=["feature", "importance", "std", "rank"])
        return pd.DataFrame([f.to_dict() for f in self.global_importance])


class ModelExplainer:
    """Explains a fitted scikit-learn pipeline.

    Parameters
    ----------
    pipeline : Any
        The fitted pipeline produced by :class:`~src.ml.trainer.MLTrainingEngine`
        (preprocessor + estimator). A bare estimator also works when no
        preprocessing was applied.
    task_type : str
        ``"classification"`` or ``"regression"``.
    seed : int
        Random seed for permutation importance and SHAP sampling.
    """

    def __init__(
        self,
        pipeline: Any,
        task_type: str = "classification",
        seed: int = 42,
    ) -> None:
        self.pipeline = pipeline
        self.task_type = task_type
        self.seed = seed
        self._shap_explainer: Any = None
        self._shap_failed = False

    # ------------------------------------------------------------------
    # Pipeline introspection
    # ------------------------------------------------------------------

    @property
    def preprocessor(self) -> Any:
        """Return the fitted preprocessor, or ``None`` for a bare estimator."""
        steps = getattr(self.pipeline, "named_steps", {})
        return steps.get("preprocessor")

    @property
    def estimator(self) -> Any:
        """Return the final estimator inside the pipeline."""
        steps = getattr(self.pipeline, "named_steps", {})
        return steps.get("model", self.pipeline)

    def encoded_feature_names(self) -> List[str]:
        """Return the feature names the estimator actually sees."""
        pre = self.preprocessor
        if pre is None:
            return []
        try:
            return [str(n) for n in pre.get_feature_names_out()]
        except Exception as exc:  # pragma: no cover - depends on sklearn version
            logger.warning("Could not read encoded feature names: %s", exc)
            return []

    def encoded_to_original(self, original_columns: List[str]) -> Dict[str, str]:
        """Map each encoded feature back to the column it came from.

        ``num__age`` maps to ``age``; ``cat__city_Delhi`` maps to ``city``. The
        longest matching original column wins, so a dataset containing both
        ``city`` and ``city_code`` is resolved correctly rather than by
        whichever name happens to be checked first.
        """
        mapping: Dict[str, str] = {}
        # Longest first so 'city_code' is preferred over 'city' when both match.
        candidates = sorted((str(c) for c in original_columns), key=len, reverse=True)

        for encoded in self.encoded_feature_names():
            stripped = re.sub(r"^(num|cat|remainder)__", "", encoded)
            match = next(
                (c for c in candidates if stripped == c or stripped.startswith(f"{c}_")),
                None,
            )
            mapping[encoded] = match if match is not None else stripped
        return mapping

    # ------------------------------------------------------------------
    # Global explanation
    # ------------------------------------------------------------------

    def global_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = _DEFAULT_REPEATS,
        max_rows: int = 2000,
    ) -> Tuple[List[FeatureImportance], str]:
        """Rank features by how much the model's score depends on them.

        Permutation importance is measured over the full pipeline, so the score
        is attributed to the original column even when it expands into many
        encoded features.

        Returns
        -------
        tuple[list[FeatureImportance], str]
            Ranked importances and the method actually used.
        """
        frame = X
        labels = y
        if len(frame) > max_rows:
            # Permutation importance is O(rows x features x repeats); sampling
            # keeps it interactive without materially changing the ranking.
            frame = frame.sample(max_rows, random_state=self.seed)
            labels = y.loc[frame.index]

        try:
            result = permutation_importance(
                self.pipeline, frame, labels,
                n_repeats=n_repeats, random_state=self.seed, n_jobs=1,
            )
            entries = [
                FeatureImportance(
                    feature=str(col),
                    importance=float(result.importances_mean[i]),
                    std=float(result.importances_std[i]),
                )
                for i, col in enumerate(frame.columns)
            ]
            method = "permutation_importance"
        except Exception as exc:
            logger.warning("Permutation importance failed (%s); using model weights.", exc)
            entries, method = self._native_importance(list(frame.columns))

        entries.sort(key=lambda f: -abs(f.importance))
        for rank, entry in enumerate(entries, start=1):
            entry.rank = rank
        return entries, method

    def _native_importance(
        self, original_columns: List[str]
    ) -> Tuple[List[FeatureImportance], str]:
        """Fall back to the estimator's own weights, aggregated per column."""
        estimator = self.estimator
        if hasattr(estimator, "feature_importances_"):
            weights = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            weights = np.abs(np.asarray(estimator.coef_, dtype=float))
            if weights.ndim > 1:
                weights = weights.mean(axis=0)
        else:
            return [], "unavailable"

        encoded = self.encoded_feature_names()
        if not encoded or len(encoded) != len(weights):
            return (
                [FeatureImportance(feature=str(c), importance=float(w))
                 for c, w in zip(original_columns, weights)],
                "model_weights",
            )

        mapping = self.encoded_to_original(original_columns)
        totals: Dict[str, float] = {}
        for name, weight in zip(encoded, weights):
            column = mapping.get(name, name)
            totals[column] = totals.get(column, 0.0) + float(weight)

        return (
            [FeatureImportance(feature=k, importance=v) for k, v in totals.items()],
            "model_weights",
        )

    # ------------------------------------------------------------------
    # Local explanation
    # ------------------------------------------------------------------

    def _build_shap_explainer(self, background: pd.DataFrame) -> Any:
        """Create a SHAP explainer suited to the final estimator.

        Tree and linear models get their exact, fast explainers; anything else
        falls back to the model-agnostic sampling explainer.
        """
        if self._shap_failed:
            return None
        if self._shap_explainer is not None:
            return self._shap_explainer

        try:
            import shap
        except ImportError:
            logger.info("shap is not installed; local explanations will use occlusion.")
            self._shap_failed = True
            return None

        try:
            estimator = self.estimator
            transformed = self._transform(background)

            if hasattr(estimator, "estimators_") or hasattr(estimator, "tree_"):
                self._shap_explainer = shap.TreeExplainer(estimator)
            elif hasattr(estimator, "coef_"):
                self._shap_explainer = shap.LinearExplainer(estimator, transformed)
            else:
                sample = shap.sample(transformed, min(_MAX_BACKGROUND, len(transformed)),
                                     random_state=self.seed)
                predict = (
                    estimator.predict_proba
                    if self.task_type == "classification" and hasattr(estimator, "predict_proba")
                    else estimator.predict
                )
                self._shap_explainer = shap.KernelExplainer(predict, sample)
            return self._shap_explainer
        except Exception as exc:
            logger.warning("Could not build a SHAP explainer (%s); using occlusion.", exc)
            self._shap_failed = True
            return None

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply every fitted step before the estimator to raw input.

        Transforming through the preprocessor alone would skip any earlier step
        -- the datetime encoder, for instance -- and hand a DatetimeArray to a
        transformer expecting floats.
        """
        steps = getattr(self.pipeline, "named_steps", None)
        if steps and len(steps) > 1:
            try:
                return np.asarray(self.pipeline[:-1].transform(X))
            except Exception as exc:  # pragma: no cover - falls back below
                logger.debug("Full-pipeline transform failed (%s); using preprocessor.", exc)

        pre = self.preprocessor
        if pre is None:
            return np.asarray(X, dtype=float)
        return np.asarray(pre.transform(X))

    def _shap_row_values(
        self,
        explainer: Any,
        row_transformed: np.ndarray,
        class_index: Optional[int],
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Return per-encoded-feature SHAP values plus the base value.

        SHAP's output shape varies by explainer and task -- a list per class, a
        3-D array, or a single flat vector -- so it is normalised here rather
        than at every call site.

        The flat-vector case needs care. A binary ``LinearExplainer`` (and some
        tree explainers) return one vector explaining the log-odds of
        ``classes_[1]``. Reported as-is against a ``classes_[0]`` prediction,
        every sign is inverted: a terrible credit score would appear to argue
        *against* the high-risk verdict it caused. The signs are therefore
        flipped when the explained class is index 0 of a binary classifier.
        """
        try:
            raw = explainer.shap_values(row_transformed)
        except Exception as exc:
            logger.warning("SHAP evaluation failed: %s", exc)
            return None

        expected = getattr(explainer, "expected_value", 0.0)
        resolved_per_class = False

        values = np.asarray(raw, dtype=float) if not isinstance(raw, list) else None
        if isinstance(raw, list):
            index = class_index if class_index is not None and class_index < len(raw) else 0
            values = np.asarray(raw[index], dtype=float)
            if isinstance(expected, (list, np.ndarray)) and len(np.atleast_1d(expected)) > index:
                expected = np.atleast_1d(expected)[index]
            resolved_per_class = True
        elif values is not None and values.ndim == 3:
            # (rows, features, classes)
            index = class_index if class_index is not None and class_index < values.shape[2] else 0
            values = values[:, :, index]
            if isinstance(expected, (list, np.ndarray)) and len(np.atleast_1d(expected)) > index:
                expected = np.atleast_1d(expected)[index]
            resolved_per_class = True

        if values is None:
            return None
        values = np.asarray(values, dtype=float).reshape(-1)
        base = float(np.atleast_1d(expected)[0]) if np.ndim(expected) else float(expected)

        if not resolved_per_class and self._is_binary_classifier() and class_index == 0:
            values = -values
            base = -base

        return values, base

    def _is_binary_classifier(self) -> bool:
        """True when the estimator is a two-class classifier."""
        if self.task_type != "classification":
            return False
        classes = getattr(self.estimator, "classes_", None)
        return classes is not None and len(classes) == 2

    def _occlusion_contributions(
        self,
        X: pd.DataFrame,
        row_index: int,
        background: pd.DataFrame,
        class_index: Optional[int],
    ) -> Dict[str, float]:
        """Model-agnostic fallback: measure each column by replacing it.

        For every column the row's value is swapped for a typical value from
        the background (median for numbers, mode for categories) and the change
        in predicted score is recorded. Less principled than SHAP, but it uses
        only the pipeline's own ``predict``, so it always works.
        """
        row = X.iloc[[row_index]]
        baseline = self._score(row, class_index)
        contributions: Dict[str, float] = {}

        for column in X.columns:
            perturbed = row.copy()
            series = background[column].dropna()
            if series.empty:
                continue
            if pd.api.types.is_numeric_dtype(series.dtype):
                replacement = series.median()
            else:
                modes = series.mode()
                replacement = modes.iloc[0] if not modes.empty else series.iloc[0]

            perturbed.iloc[0, perturbed.columns.get_loc(column)] = replacement
            try:
                contributions[str(column)] = float(baseline - self._score(perturbed, class_index))
            except Exception:
                continue
        return contributions

    def _score(self, rows: pd.DataFrame, class_index: Optional[int]) -> float:
        """Return the model's score for a single row."""
        if self.task_type == "classification" and hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(rows)
            index = class_index if class_index is not None else 0
            return float(proba[0][index])
        return float(np.asarray(self.pipeline.predict(rows)).reshape(-1)[0])

    def explain_row(
        self,
        X: pd.DataFrame,
        row_index: int,
        background: Optional[pd.DataFrame] = None,
        top_n: int = 10,
    ) -> LocalExplanation:
        """Explain a single prediction.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature frame (no target column).
        row_index : int
            Positional index of the row to explain.
        background : pd.DataFrame | None
            Reference distribution. Defaults to ``X``.
        top_n : int
            Number of contributing features to keep.

        Returns
        -------
        LocalExplanation
            The prediction plus what pushed it there.

        Raises
        ------
        IndexError
            If ``row_index`` is out of range.
        """
        if not 0 <= row_index < len(X):
            raise IndexError(f"Row {row_index} is out of range for {len(X)} rows.")

        reference = background if background is not None else X
        row = X.iloc[[row_index]]

        prediction = self.pipeline.predict(row)[0]
        confidence: Optional[float] = None
        class_index: Optional[int] = None

        if self.task_type == "classification" and hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(row)[0]
            classes = list(getattr(self.estimator, "classes_", []))
            if classes and prediction in classes:
                class_index = classes.index(prediction)
            else:
                class_index = int(np.argmax(proba))
            confidence = float(proba[class_index])

        contributions: List[FeatureContribution] = []
        base_value: Optional[float] = None

        explainer = self._build_shap_explainer(reference)
        if explainer is not None:
            result = self._shap_row_values(
                explainer, self._transform(row), class_index
            )
            if result is not None:
                values, base_value = result
                encoded = self.encoded_feature_names()
                if len(encoded) == len(values):
                    # Sum the one-hot fragments back into their source column.
                    mapping = self.encoded_to_original(list(X.columns))
                    totals: Dict[str, float] = {}
                    for name, value in zip(encoded, values):
                        column = mapping.get(name, name)
                        totals[column] = totals.get(column, 0.0) + float(value)
                    contributions = [
                        FeatureContribution(
                            feature=column,
                            value=row.iloc[0].get(column, None),
                            contribution=total,
                        )
                        for column, total in totals.items()
                    ]

        if not contributions:
            occluded = self._occlusion_contributions(X, row_index, reference, class_index)
            contributions = [
                FeatureContribution(
                    feature=column,
                    value=row.iloc[0].get(column, None),
                    contribution=value,
                )
                for column, value in occluded.items()
            ]

        contributions.sort(key=lambda c: -abs(c.contribution))
        return LocalExplanation(
            row_index=int(row_index),
            prediction=prediction,
            confidence=confidence,
            base_value=base_value,
            contributions=contributions[:top_n],
            task_type=self.task_type,
        )

    # ------------------------------------------------------------------
    # Combined report
    # ------------------------------------------------------------------

    def explain(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        rows: Optional[List[int]] = None,
        model_name: str = "",
        target_column: Optional[str] = None,
        n_repeats: int = _DEFAULT_REPEATS,
        top_n: int = 10,
    ) -> ExplanationReport:
        """Produce global importance plus explanations for selected rows.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature frame.
        y : pd.Series | None
            Labels. Required for permutation importance; without them the
            estimator's own weights are used instead.
        rows : list[int] | None
            Row positions to explain individually.
        model_name : str
            Name shown in the report.
        target_column : str | None
            Recorded for context.
        n_repeats : int
            Permutation repeats.
        top_n : int
            Contributions kept per row.

        Returns
        -------
        ExplanationReport
            Global and local explanations.
        """
        if y is not None:
            importance, method = self.global_importance(X, y, n_repeats=n_repeats)
        else:
            importance, method = self._native_importance(list(X.columns))
            importance.sort(key=lambda f: -abs(f.importance))
            for rank, entry in enumerate(importance, start=1):
                entry.rank = rank

        report = ExplanationReport(
            model_name=model_name or type(self.estimator).__name__,
            task_type=self.task_type,
            method=method,
            global_importance=importance,
            target_column=target_column,
        )

        for row_index in (rows or []):
            try:
                report.local_explanations.append(
                    self.explain_row(X, row_index, background=X, top_n=top_n)
                )
            except Exception as exc:
                logger.warning("Could not explain row %s: %s", row_index, exc)

        if report.local_explanations:
            used_shap = self._shap_explainer is not None and not self._shap_failed
            report.method = f"{method} + {'shap' if used_shap else 'occlusion'}"

        logger.info("Explainability complete: %s", report.summary())
        return report


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-friendly Python values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _short(value: Any, width: int = 24) -> str:
    """Render a value compactly for narrative text."""
    text = str(_jsonable(value))
    return text if len(text) <= width else text[: width - 3] + "..."


__all__ = [
    "ExplanationReport",
    "FeatureContribution",
    "FeatureImportance",
    "LocalExplanation",
    "ModelExplainer",
]
