"""
Tests for src.drift
====================

Drift is judged on data with a *known* shift, so the assertions check that the
detector reaches the right verdict rather than merely that it returns a number.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.drift.detector import DriftDetector, DriftReport, DriftStatus


@pytest.fixture
def rng():
    return np.random.default_rng(20260905)


@pytest.fixture
def reference(rng):
    """A baseline dataset standing in for training data."""
    n = 1500
    return pd.DataFrame({
        "age": rng.normal(45, 12, n),
        "income": rng.normal(60000, 15000, n),
        "steady": rng.normal(10, 2, n),
        "city": rng.choice(["Delhi", "Mumbai", "Pune"], n, p=[0.5, 0.3, 0.2]),
        "plan": rng.choice(["basic", "pro"], n, p=[0.7, 0.3]),
    })


@pytest.fixture
def unchanged(rng):
    """A second sample from the same distributions as ``reference``."""
    n = 1500
    return pd.DataFrame({
        "age": rng.normal(45, 12, n),
        "income": rng.normal(60000, 15000, n),
        "steady": rng.normal(10, 2, n),
        "city": rng.choice(["Delhi", "Mumbai", "Pune"], n, p=[0.5, 0.3, 0.2]),
        "plan": rng.choice(["basic", "pro"], n, p=[0.7, 0.3]),
    })


def _by_name(report: DriftReport):
    return {c.column: c for c in report.columns}


class TestNoDrift:
    """Two samples from the same distribution must not raise an alarm."""

    def test_all_columns_stable(self, reference, unchanged):
        report = DriftDetector().detect(reference, unchanged)
        for col in report.compared_columns:
            assert col.status == DriftStatus.STABLE, f"{col.column}: {col.explanation}"

    def test_dataset_verdict_is_stable(self, reference, unchanged):
        report = DriftDetector().detect(reference, unchanged)
        assert report.status == DriftStatus.STABLE
        assert report.overall_score < 0.10

    def test_identical_frames_score_zero(self, reference):
        report = DriftDetector().detect(reference, reference.copy())
        assert report.overall_score == pytest.approx(0.0, abs=1e-6)
        assert report.status == DriftStatus.STABLE


class TestNumericDrift:
    """A shifted numeric column must be detected and described."""

    def test_large_mean_shift_is_critical(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["age"] = rng.normal(60, 12, len(current))

        report = DriftDetector().detect(reference, current)
        age = _by_name(report)["age"]
        assert age.status == DriftStatus.CRITICAL
        assert age.psi > 0.25
        assert age.column_kind == "numeric"

    def test_explanation_states_the_direction(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["age"] = rng.normal(60, 12, len(current))
        report = DriftDetector().detect(reference, current)
        assert "higher" in _by_name(report)["age"].explanation

    def test_ks_test_is_reported(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["age"] = rng.normal(60, 12, len(current))
        age = _by_name(DriftDetector().detect(reference, current))["age"]
        assert age.test_name == "ks_2samp"
        assert age.p_value is not None and age.p_value < 0.05
        assert age.significant is True

    def test_untouched_columns_stay_stable(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["age"] = rng.normal(60, 12, len(current))
        report = DriftDetector().detect(reference, current)
        assert _by_name(report)["steady"].status == DriftStatus.STABLE

    def test_variance_change_is_detected(self, reference, unchanged, rng):
        # Same mean, much wider spread -- a shift the mean alone would miss.
        current = unchanged.copy()
        current["steady"] = rng.normal(10, 8, len(current))
        steady = _by_name(DriftDetector().detect(reference, current))["steady"]
        assert steady.drifted


class TestCategoricalDrift:
    """Category mix changes and new categories."""

    def test_category_mix_change_is_flagged(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["city"] = rng.choice(["Delhi", "Mumbai", "Pune"], len(current), p=[0.1, 0.2, 0.7])
        city = _by_name(DriftDetector().detect(reference, current))["city"]
        assert city.drifted
        assert city.column_kind == "categorical"

    def test_new_category_is_named(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["city"] = rng.choice(["Delhi", "Mumbai", "Pune", "Goa"], len(current))
        city = _by_name(DriftDetector().detect(reference, current))["city"]
        assert city.status == DriftStatus.CRITICAL
        assert "Goa" in city.explanation
        assert "Goa" in city.current_stats["new_categories"]

    def test_missing_category_is_named(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["city"] = rng.choice(["Delhi", "Mumbai"], len(current))
        city = _by_name(DriftDetector().detect(reference, current))["city"]
        assert "Pune" in city.current_stats["missing_categories"]

    def test_chi_square_is_reported(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["city"] = rng.choice(["Delhi", "Mumbai", "Pune"], len(current), p=[0.1, 0.2, 0.7])
        city = _by_name(DriftDetector().detect(reference, current))["city"]
        assert city.test_name == "chi2_contingency"
        assert city.p_value is not None


class TestSchemaChanges:
    """Structural differences are reported, never silently ignored."""

    def test_removed_column(self, reference, unchanged):
        current = unchanged.drop(columns=["income"])
        report = DriftDetector().detect(reference, current)
        assert "income" in report.removed_columns
        assert _by_name(report)["income"].status == DriftStatus.REMOVED_COLUMN
        assert report.status == DriftStatus.CRITICAL

    def test_added_column(self, reference, unchanged):
        current = unchanged.copy()
        current["segment"] = "A"
        report = DriftDetector().detect(reference, current)
        assert "segment" in report.added_columns
        assert _by_name(report)["segment"].status == DriftStatus.NEW_COLUMN

    def test_type_change_is_flagged_not_scored(self, reference, unchanged):
        current = unchanged.copy()
        current["age"] = current["age"].astype(str)
        report = DriftDetector().detect(reference, current)
        age = _by_name(report)["age"]
        assert age.status == DriftStatus.TYPE_CHANGED
        assert age.psi is None
        assert "age" in report.type_changed_columns

    def test_completely_disjoint_schemas(self, reference):
        current = pd.DataFrame({"totally": [1, 2, 3], "different": [4, 5, 6]})
        report = DriftDetector().detect(reference, current)
        assert len(report.removed_columns) == len(reference.columns)
        assert len(report.added_columns) == 2
        assert report.compared_columns == []


class TestMissingValues:
    """A column arriving mostly empty has changed even if its values have not."""

    def test_missing_rate_jump_raises_status(self, reference, unchanged):
        current = unchanged.copy()
        current.loc[current.sample(frac=0.4, random_state=1).index, "steady"] = np.nan
        steady = _by_name(DriftDetector().detect(reference, current))["steady"]
        assert steady.status != DriftStatus.STABLE
        # The verdict is driven by the missing rate, not by the observed values,
        # and the explanation must say so rather than claiming a distribution shift.
        assert "missing rate" in steady.explanation
        assert steady.missing_rate_drift is True

    def test_too_few_rows_is_not_comparable(self, reference):
        current = reference.head(5)
        report = DriftDetector().detect(reference, current)
        for col in report.columns:
            assert col.status == DriftStatus.NOT_COMPARABLE


class TestThresholds:
    """Thresholds are configurable and actually applied."""

    def test_stricter_threshold_promotes_status(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["age"] = rng.normal(47, 12, len(current))

        lenient = DriftDetector().detect(reference, current)
        strict = DriftDetector(thresholds={
            "psi_warning_threshold": 0.0001, "psi_critical_threshold": 0.001,
        }).detect(reference, current)

        assert _by_name(lenient)["age"].status == DriftStatus.STABLE
        assert _by_name(strict)["age"].status != DriftStatus.STABLE

    def test_thresholds_are_recorded_in_the_report(self, reference, unchanged):
        report = DriftDetector(thresholds={"psi_warning_threshold": 0.42}).detect(
            reference, unchanged
        )
        assert report.thresholds["psi_warning_threshold"] == 0.42

    def test_column_subset_is_respected(self, reference, unchanged):
        report = DriftDetector().detect(reference, unchanged, columns=["age", "city"])
        assert {c.column for c in report.compared_columns} == {"age", "city"}


class TestReportShape:
    """The report must serialise cleanly for the API and the HTML report."""

    def test_to_dict_is_json_serialisable(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["age"] = rng.normal(60, 12, len(current))
        report = DriftDetector().detect(reference, current)
        encoded = json.dumps(report.to_dict())
        assert "overall_score" in encoded

    def test_to_frame_has_a_row_per_column(self, reference, unchanged):
        report = DriftDetector().detect(reference, unchanged)
        assert len(report.to_frame()) == len(report.columns)

    def test_distribution_frame_is_long_format(self, reference, unchanged):
        report = DriftDetector().detect(reference, unchanged)
        result = report.compared_columns[0]
        frame = result.distribution_frame()
        assert set(frame["dataset"].unique()) == {"reference", "current"}
        assert len(frame) == 2 * len(result.distribution)

    def test_results_are_sorted_worst_first(self, reference, unchanged, rng):
        current = unchanged.copy()
        current["age"] = rng.normal(60, 12, len(current))
        report = DriftDetector().detect(reference, current)
        assert report.columns[0].status == DriftStatus.CRITICAL

    def test_summary_counts_add_up(self, reference, unchanged):
        report = DriftDetector().detect(reference, unchanged)
        assert sum(report.summary_counts().values()) == len(report.columns)


class TestEdgeCases:
    """Degenerate inputs must not crash."""

    def test_empty_dataset_raises(self, reference):
        with pytest.raises(ValueError, match="empty"):
            DriftDetector().detect(reference, pd.DataFrame())

    def test_constant_column_is_handled(self):
        ref = pd.DataFrame({"c": [5.0] * 100})
        cur = pd.DataFrame({"c": [5.0] * 100})
        report = DriftDetector().detect(ref, cur)
        assert report.compared_columns[0].status == DriftStatus.STABLE

    def test_constant_column_that_moves_is_flagged(self):
        ref = pd.DataFrame({"c": [5.0] * 100})
        cur = pd.DataFrame({"c": [9.0] * 100})
        report = DriftDetector().detect(ref, cur)
        assert report.compared_columns[0].drifted

    def test_all_null_column_is_not_comparable(self, reference, unchanged):
        current = unchanged.copy()
        current["steady"] = np.nan
        steady = _by_name(DriftDetector().detect(reference, current))["steady"]
        assert steady.status == DriftStatus.NOT_COMPARABLE

    def test_low_cardinality_numeric_treated_as_categorical(self):
        ref = pd.DataFrame({"flag": [0, 1] * 100})
        cur = pd.DataFrame({"flag": [0, 1] * 100})
        report = DriftDetector().detect(ref, cur)
        assert report.compared_columns[0].column_kind == "categorical"

    def test_psi_is_finite_when_a_bin_empties(self):
        # An empty reference bin would divide by zero without the epsilon floor.
        ref = pd.DataFrame({"v": list(range(100))})
        cur = pd.DataFrame({"v": [500] * 100})
        result = DriftDetector().detect(ref, cur).compared_columns[0]
        assert np.isfinite(result.psi)
        assert result.status == DriftStatus.CRITICAL
