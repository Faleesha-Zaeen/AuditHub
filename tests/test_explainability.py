"""
Tests for src.ml.explainer
===========================

Explanations are checked against data with a *known* generating process, so a
wrong answer fails rather than merely a missing one. The sign convention gets
particular attention: a local explanation whose signs are inverted is worse
than none, because it reads as confident and is exactly backwards.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.explainer import (
    ExplanationReport,
    LocalExplanation,
    ModelExplainer,
)


# ---------------------------------------------------------------------------
# Fixtures: data with a signal we control
# ---------------------------------------------------------------------------


@pytest.fixture
def credit_data():
    """Risk depends overwhelmingly on credit_score, mildly on income/age."""
    rng = np.random.default_rng(20260905)
    n = 400
    credit = rng.normal(650, 90, n)
    income = rng.normal(60000, 18000, n)
    age = rng.integers(21, 75, n)
    logit = -0.02 * (credit - 650) - 0.00002 * (income - 60000) + 0.01 * (age - 45)
    risk = np.where(1 / (1 + np.exp(-logit)) > 0.5, "high_risk", "low_risk")
    df = pd.DataFrame({
        "credit_score": credit.round(0),
        "income": income.round(0),
        "age": age,
        "city": rng.choice(["Delhi", "Mumbai", "Pune"], n),
        "risk": risk,
    })
    return df.drop(columns=["risk"]), df["risk"]


@pytest.fixture
def regression_data():
    """Price is driven by size; colour is noise."""
    rng = np.random.default_rng(7)
    n = 400
    size = rng.normal(100, 25, n)
    noise = rng.normal(0, 5, n)
    df = pd.DataFrame({
        "size": size.round(1),
        "rooms": rng.integers(1, 6, n),
        "colour": rng.choice(["red", "blue"], n),
    })
    y = pd.Series(size * 1000 + noise * 100, name="price")
    return df, y


def _build_pipeline(X: pd.DataFrame, y: pd.Series, estimator) -> Pipeline:
    """Build the same shape of pipeline the trainer produces."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c].dtype)]
    categorical = [c for c in X.columns if c not in numeric]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), categorical),
    ])
    pipeline = Pipeline([("preprocessor", pre), ("model", estimator)])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def logistic_model(credit_data):
    X, y = credit_data
    return _build_pipeline(X, y, LogisticRegression(max_iter=500, random_state=42))


@pytest.fixture
def forest_model(credit_data):
    X, y = credit_data
    return _build_pipeline(X, y, RandomForestClassifier(n_estimators=40, random_state=42))


@pytest.fixture
def ridge_model(regression_data):
    X, y = regression_data
    return _build_pipeline(X, y, Ridge(random_state=42))


# ---------------------------------------------------------------------------
# Pipeline introspection
# ---------------------------------------------------------------------------


class TestPipelineIntrospection:
    """Explanations must be expressed in the user's own column names."""

    def test_encoded_names_are_read_from_the_fitted_preprocessor(self, logistic_model, credit_data):
        X, _ = credit_data
        explainer = ModelExplainer(logistic_model)
        encoded = explainer.encoded_feature_names()
        assert "num__credit_score" in encoded
        assert any(name.startswith("cat__city_") for name in encoded)

    def test_one_hot_columns_map_back_to_their_source(self, logistic_model, credit_data):
        X, _ = credit_data
        mapping = ModelExplainer(logistic_model).encoded_to_original(list(X.columns))
        assert mapping["num__credit_score"] == "credit_score"
        assert all(
            v == "city" for k, v in mapping.items() if k.startswith("cat__city_")
        )

    def test_prefix_collisions_resolve_to_the_longest_match(self):
        # 'city_code' must not be swallowed by 'city'.
        rng = np.random.default_rng(1)
        X = pd.DataFrame({
            "city": rng.choice(["A", "B"], 60),
            "city_code": rng.choice(["X", "Y"], 60),
        })
        y = pd.Series(rng.choice([0, 1], 60))
        model = _build_pipeline(X, y, LogisticRegression(max_iter=200))
        mapping = ModelExplainer(model).encoded_to_original(list(X.columns))
        assert mapping["cat__city_code_X"] == "city_code"
        assert mapping["cat__city_A"] == "city"


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------


class TestGlobalImportance:
    """The dominant feature must come out on top."""

    def test_dominant_feature_ranks_first(self, logistic_model, credit_data):
        X, y = credit_data
        importance, method = ModelExplainer(logistic_model).global_importance(
            X, y, n_repeats=5
        )
        assert method == "permutation_importance"
        assert importance[0].feature == "credit_score"

    def test_importance_is_per_original_column(self, logistic_model, credit_data):
        X, y = credit_data
        importance, _ = ModelExplainer(logistic_model).global_importance(X, y, n_repeats=3)
        assert {f.feature for f in importance} == set(X.columns)

    def test_ranks_are_assigned_in_order(self, logistic_model, credit_data):
        X, y = credit_data
        importance, _ = ModelExplainer(logistic_model).global_importance(X, y, n_repeats=3)
        assert [f.rank for f in importance] == list(range(1, len(importance) + 1))

    def test_tree_model_also_ranks_correctly(self, forest_model, credit_data):
        X, y = credit_data
        importance, _ = ModelExplainer(forest_model).global_importance(X, y, n_repeats=5)
        assert importance[0].feature == "credit_score"

    def test_native_weights_fallback_aggregates_one_hot(self, logistic_model, credit_data):
        X, _ = credit_data
        explainer = ModelExplainer(logistic_model)
        importance, method = explainer._native_importance(list(X.columns))
        assert method == "model_weights"
        # 'city' appears once, not once per category.
        assert [f.feature for f in importance].count("city") == 1

    def test_works_without_labels(self, logistic_model, credit_data):
        X, _ = credit_data
        report = ModelExplainer(logistic_model).explain(X, y=None)
        assert report.global_importance
        assert report.method == "model_weights"


# ---------------------------------------------------------------------------
# Local explanations
# ---------------------------------------------------------------------------


class TestLocalExplanationSigns:
    """The sign convention: positive means 'pushed toward the prediction'."""

    def test_poor_credit_score_drives_a_high_risk_verdict(self, logistic_model, credit_data):
        X, _ = credit_data
        worst = int(np.argmin(X["credit_score"].values))
        explanation = ModelExplainer(logistic_model).explain_row(X, worst)

        assert explanation.prediction == "high_risk"
        credit = next(c for c in explanation.contributions if c.feature == "credit_score")
        # A 399 credit score must argue FOR high risk, not against it.
        assert credit.contribution > 0
        assert credit.direction == "increases"

    def test_excellent_credit_score_drives_a_low_risk_verdict(self, logistic_model, credit_data):
        X, _ = credit_data
        best = int(np.argmax(X["credit_score"].values))
        explanation = ModelExplainer(logistic_model).explain_row(X, best)

        assert explanation.prediction == "low_risk"
        credit = next(c for c in explanation.contributions if c.feature == "credit_score")
        assert credit.contribution > 0, "the strongest evidence must support the prediction made"

    def test_signs_are_consistent_across_both_classes(self, logistic_model, credit_data):
        """The same feature cannot support opposite verdicts with the same sign."""
        X, _ = credit_data
        explainer = ModelExplainer(logistic_model)
        worst = int(np.argmin(X["credit_score"].values))
        best = int(np.argmax(X["credit_score"].values))

        low = explainer.explain_row(X, worst)
        high = explainer.explain_row(X, best)
        assert low.prediction != high.prediction
        # Both explanations are stated relative to their own prediction, so the
        # driving feature is positive in each.
        for explanation in (low, high):
            credit = next(c for c in explanation.contributions if c.feature == "credit_score")
            assert credit.contribution > 0

    def test_tree_model_signs_are_also_correct(self, forest_model, credit_data):
        X, _ = credit_data
        worst = int(np.argmin(X["credit_score"].values))
        explanation = ModelExplainer(forest_model).explain_row(X, worst)
        credit = next(c for c in explanation.contributions if c.feature == "credit_score")
        assert explanation.prediction == "high_risk"
        assert credit.contribution > 0


class TestLocalExplanationShape:
    """Structure of a single-row explanation."""

    def test_contributions_use_original_column_names(self, logistic_model, credit_data):
        X, _ = credit_data
        explanation = ModelExplainer(logistic_model).explain_row(X, 0)
        assert {c.feature for c in explanation.contributions} <= set(X.columns)

    def test_reports_prediction_and_confidence(self, logistic_model, credit_data):
        X, _ = credit_data
        explanation = ModelExplainer(logistic_model).explain_row(X, 0)
        assert explanation.prediction in ("high_risk", "low_risk")
        assert 0.0 <= explanation.confidence <= 1.0

    def test_actual_row_values_are_attached(self, logistic_model, credit_data):
        X, _ = credit_data
        explanation = ModelExplainer(logistic_model).explain_row(X, 5)
        credit = next(c for c in explanation.contributions if c.feature == "credit_score")
        assert credit.value == X.iloc[5]["credit_score"]

    def test_toward_and_against_are_partitioned(self, logistic_model, credit_data):
        X, _ = credit_data
        explanation = ModelExplainer(logistic_model).explain_row(X, 0)
        toward = explanation.pushing_toward()
        against = explanation.pushing_against()
        assert all(c.contribution > 0 for c in toward)
        assert all(c.contribution < 0 for c in against)
        assert len(toward) + len(against) <= len(explanation.contributions)

    def test_narrative_reads_as_prose(self, logistic_model, credit_data):
        X, _ = credit_data
        text = ModelExplainer(logistic_model).explain_row(X, 0).narrative()
        assert text.startswith("Prediction:")
        assert "credit_score" in text or "income" in text

    def test_top_n_is_respected(self, logistic_model, credit_data):
        X, _ = credit_data
        explanation = ModelExplainer(logistic_model).explain_row(X, 0, top_n=2)
        assert len(explanation.contributions) == 2

    def test_contributions_are_sorted_by_magnitude(self, logistic_model, credit_data):
        X, _ = credit_data
        contributions = ModelExplainer(logistic_model).explain_row(X, 0).contributions
        magnitudes = [abs(c.contribution) for c in contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_out_of_range_row_raises(self, logistic_model, credit_data):
        X, _ = credit_data
        with pytest.raises(IndexError):
            ModelExplainer(logistic_model).explain_row(X, 99999)


class TestRegression:
    """Regression models are explained too."""

    def test_dominant_feature_ranks_first(self, ridge_model, regression_data):
        X, y = regression_data
        importance, _ = ModelExplainer(ridge_model, task_type="regression").global_importance(
            X, y, n_repeats=5
        )
        assert importance[0].feature == "size"

    def test_local_explanation_has_no_confidence(self, ridge_model, regression_data):
        X, _ = regression_data
        explanation = ModelExplainer(ridge_model, task_type="regression").explain_row(X, 0)
        assert explanation.confidence is None
        assert isinstance(explanation.prediction, (float, np.floating))


class TestOcclusionFallback:
    """When SHAP is unavailable the model-agnostic path must still work."""

    def test_occlusion_is_used_when_shap_is_disabled(self, logistic_model, credit_data):
        X, _ = credit_data
        explainer = ModelExplainer(logistic_model)
        explainer._shap_failed = True  # simulate shap being unavailable

        explanation = explainer.explain_row(X, int(np.argmin(X["credit_score"].values)))
        assert explanation.contributions
        assert {c.feature for c in explanation.contributions} <= set(X.columns)

    def test_occlusion_signs_are_also_correct(self, logistic_model, credit_data):
        X, _ = credit_data
        explainer = ModelExplainer(logistic_model)
        explainer._shap_failed = True

        worst = int(np.argmin(X["credit_score"].values))
        explanation = explainer.explain_row(X, worst)
        credit = next(c for c in explanation.contributions if c.feature == "credit_score")
        assert credit.contribution > 0


class TestExplanationReport:
    """The combined report used by the UI, API and HTML report."""

    def test_report_contains_global_and_local(self, logistic_model, credit_data):
        X, y = credit_data
        report = ModelExplainer(logistic_model).explain(
            X, y, rows=[0, 1], model_name="LogReg", target_column="risk", n_repeats=3
        )
        assert isinstance(report, ExplanationReport)
        assert report.global_importance
        assert len(report.local_explanations) == 2
        assert report.target_column == "risk"

    def test_report_is_json_serialisable(self, logistic_model, credit_data):
        X, y = credit_data
        report = ModelExplainer(logistic_model).explain(X, y, rows=[0], n_repeats=3)
        json.dumps(report.to_dict())

    def test_method_records_which_engines_ran(self, logistic_model, credit_data):
        X, y = credit_data
        report = ModelExplainer(logistic_model).explain(X, y, rows=[0], n_repeats=3)
        assert "permutation_importance" in report.method
        assert "shap" in report.method or "occlusion" in report.method

    def test_importance_frame_is_chartable(self, logistic_model, credit_data):
        X, y = credit_data
        report = ModelExplainer(logistic_model).explain(X, y, n_repeats=3)
        frame = report.importance_frame()
        assert list(frame.columns) == ["feature", "importance", "std", "rank"]
        assert len(frame) == len(X.columns)

    def test_summary_names_the_top_features(self, logistic_model, credit_data):
        X, y = credit_data
        report = ModelExplainer(logistic_model).explain(X, y, n_repeats=3)
        assert "credit_score" in report.summary()

    def test_bad_row_does_not_abort_the_report(self, logistic_model, credit_data):
        X, y = credit_data
        report = ModelExplainer(logistic_model).explain(X, y, rows=[0, 99999], n_repeats=3)
        # The valid row is still explained; the invalid one is skipped.
        assert len(report.local_explanations) == 1


class TestDatetimeFeatures:
    """A date column must not break training, prediction or explanation.

    A real cafe-sales export exposed this: the datetime conversion was applied
    to the training frame but not baked into the pipeline, so the saved model
    raised ``TypeError: Cannot cast DatetimeArray to dtype float64`` the moment
    anyone called predict() on the data it was trained from.
    """

    @pytest.fixture
    def dated_frame(self):
        rng = np.random.default_rng(11)
        n = 300
        return pd.DataFrame({
            "amount": rng.normal(50, 12, n).round(2),
            "seen_on": pd.to_datetime("2023-01-01") + pd.to_timedelta(
                rng.integers(0, 400, n), unit="D"
            ),
            "channel": rng.choice(["web", "store"], n),
            "label": rng.choice(["yes", "no"], n),
        })

    @pytest.fixture
    def dated_model(self, dated_frame):
        from src.ml.trainer import MLTrainingEngine

        model, metrics, path = MLTrainingEngine().train(
            dated_frame, target_column="label", dataset_name="dates"
        )
        return model, metrics, path

    def test_training_succeeds_with_a_datetime_column(self, dated_model):
        model, metrics, _ = dated_model
        assert "f1_score" in metrics

    def test_pipeline_predicts_on_the_raw_frame(self, dated_model, dated_frame):
        model, _, _ = dated_model
        X = dated_frame.drop(columns=["label"])
        assert pd.api.types.is_datetime64_any_dtype(X["seen_on"].dtype)
        # Must work without the caller pre-converting anything.
        assert len(model.predict(X.head(5))) == 5

    def test_saved_model_still_predicts_on_raw_data(self, dated_model, dated_frame):
        import joblib

        _, _, path = dated_model
        reloaded = joblib.load(path)
        X = dated_frame.drop(columns=["label"])
        assert len(reloaded.predict(X.head(3))) == 3

    def test_datetime_encoder_is_part_of_the_pipeline(self, dated_model):
        model, _, _ = dated_model
        assert "datetime_encoder" in model.named_steps

    def test_date_is_used_as_a_number_not_one_hot(self, dated_model):
        # One dummy column per distinct date would explode the feature space.
        model, _, _ = dated_model
        encoded = list(model.named_steps["preprocessor"].get_feature_names_out())
        assert "num__seen_on" in encoded
        assert not any(name.startswith("cat__seen_on") for name in encoded)

    def test_explanation_works_with_dates(self, dated_model, dated_frame):
        model, _, _ = dated_model
        X = dated_frame.drop(columns=["label"])
        explanation = ModelExplainer(model, task_type="classification").explain_row(X, 0)
        assert explanation.contributions
        assert {c.feature for c in explanation.contributions} <= set(X.columns)

    def test_encoder_leaves_non_datetime_columns_alone(self, dated_frame):
        from src.ml.trainer import encode_datetime_columns

        out = encode_datetime_columns(dated_frame)
        assert out["channel"].equals(dated_frame["channel"])
        assert out["amount"].equals(dated_frame["amount"])
        assert pd.api.types.is_numeric_dtype(out["seen_on"].dtype)
