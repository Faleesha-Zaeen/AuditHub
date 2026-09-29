"""
Tests for src.ingestion.cleaning
=================================

Covers the load-time normalisation that makes "missing" mean the same thing
everywhere: disguised nulls become real nulls, numeric-looking text becomes
numeric, and integer columns stay integral.
"""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.cleaning import (
    AMBIGUOUS_SENTINELS,
    CleaningReport,
    MISSING_SENTINELS,
    UNAMBIGUOUS_SENTINELS,
    clean_dataframe,
    coerce_datetime_like,
    coerce_numeric_like,
    deduplicate_columns,
    downcast_integral_floats,
    drop_empty_rows_and_columns,
    drop_unnamed_index,
    normalize_headers,
    normalize_missing_sentinels,
    strip_text_whitespace,
)


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------


class TestMissingSentinels:
    """Disguised missing markers must become real nulls."""

    @pytest.mark.parametrize(
        "token", ["?", "??", "-", "--", "N/A", "n/a", "N.A.", "", "  ", ".",
                  "#N/A", "#VALUE!", "#REF!", "<NA>", "nan"],
    )
    def test_unambiguous_sentinels_always_become_null(self, token):
        # No dataset uses "#REF!" or "?" as an answer, so these are converted
        # wherever they appear regardless of capitalisation or column.
        # drop_empty is off so the row is not removed for being wholly null --
        # this asserts the sentinel conversion itself, not row pruning.
        df = pd.DataFrame({"col": ["real", token, "also real"]})
        out, report = clean_dataframe(
            df, coerce_numeric=False, coerce_datetime=False, drop_empty=False
        )
        assert out["col"].isna().sum() == 1
        assert not report.is_empty

    def test_legitimate_values_are_not_nulled(self):
        # "0", "false" and "no" are real data and must survive. Numeric zero in
        # particular is the one people lose most often to over-eager cleaning.
        df = pd.DataFrame({"col": ["0", "false", "no", "-1", "none of the above"]})
        out, _ = normalize_missing_sentinels(df, CleaningReport()), None
        assert out["col"].isna().sum() == 0

    @pytest.mark.parametrize("token", ["ERROR", "UNKNOWN", "NULL", "NA", "MISSING"])
    def test_shouting_placeholders_are_treated_as_missing(self, token):
        # A real cafe-sales export carried 1,931 ERROR/UNKNOWN cells. Counted as
        # categories they showed up in results as though "ERROR" were a payment
        # method, so they are missing data, not values. Capitalisation is the
        # tell: a machine shouted these into a column of Title Case words.
        df = pd.DataFrame({"id": [1, 2, 3], "payment": ["Cash", token, "Card"]})
        out, _ = clean_dataframe(df)
        assert out["payment"].isna().sum() == 1
        assert token not in out["payment"].dropna().tolist()

    def test_sentinel_set_is_lowercase(self):
        # Matching is done on lowercased values, so the set must be too.
        assert all(s == s.lower() for s in MISSING_SENTINELS)

    def test_the_two_sets_do_not_overlap(self):
        assert not (UNAMBIGUOUS_SENTINELS & AMBIGUOUS_SENTINELS)
        assert MISSING_SENTINELS == UNAMBIGUOUS_SENTINELS | AMBIGUOUS_SENTINELS


class TestAmbiguousTokensAreRealWords:
    """"None", "Unknown" and "Null" are answers as often as they are gaps.

    Deleting a real value is unrecoverable; keeping a junk one is a nuisance.
    So an ambiguous token is only stripped when its capitalisation betrays it
    as something a machine wrote.
    """

    def test_none_means_no_allergies(self):
        # The case that made this necessary: wiping "None" here and then mode-
        # filling it wrote "Penicillin" into the record of a patient who has no
        # allergies at all.
        df = pd.DataFrame({
            "patient": range(6),
            "allergy": ["Penicillin", "None", "None", "Latex", "None", "None"],
        })
        out, _ = clean_dataframe(df)
        assert out["allergy"].isna().sum() == 0
        assert (out["allergy"] == "None").sum() == 4

    def test_unknown_is_a_real_answer_on_a_form(self):
        df = pd.DataFrame({
            "ancestry": ["European", "Unknown", "Unknown", "African", "Asian", "Unknown"],
        })
        out, _ = clean_dataframe(df)
        assert out["ancestry"].isna().sum() == 0

    def test_null_is_a_surname(self):
        df = pd.DataFrame({
            "last_name": ["Null", "Nguyen", "None", "Ali", "MacLeod", "Okafor"],
        })
        out, _ = clean_dataframe(df)
        assert out["last_name"].isna().sum() == 0

    def test_multi_word_answers_survive(self):
        df = pd.DataFrame({
            "insurance": ["Aetna", "Not applicable", "Aetna", "Cigna",
                          "Not applicable", "Aetna"],
        })
        out, _ = clean_dataframe(df)
        assert out["insurance"].isna().sum() == 0

    def test_an_all_caps_column_keeps_its_tokens(self):
        # Nothing to contrast against: if the whole column shouts, a shouted
        # "UNKNOWN" carries no signal, so it is kept rather than guessed at.
        df = pd.DataFrame({"code": ["ALPHA", "UNKNOWN", "BETA", "GAMMA"] * 5})
        out, _ = clean_dataframe(df)
        assert out["code"].isna().sum() == 0

    def test_free_text_is_never_sentinel_matched(self):
        # A review that discusses the unknown is a review, not a gap.
        df = pd.DataFrame({
            "review": ["the cause is unknown to me",
                       "none of it worked for us at all",
                       "great flat white and very fast service"] * 5,
        })
        out, _ = clean_dataframe(df)
        assert out["review"].isna().sum() == 0

    def test_preserved_values_are_reported_not_silent(self):
        df = pd.DataFrame({"allergy": ["Penicillin", "None", "Latex", "None"]})
        _, report = clean_dataframe(df)
        steps = {a.step for a in report.actions}
        assert "preserve_ambiguous_value" in steps


class TestSentinelOverrides:
    """The heuristic is a default, not a verdict."""

    def test_force_sentinels_converts_what_the_heuristic_kept(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "payment": ["Cash", "unknown", "Card", "none"]})
        kept, _ = clean_dataframe(df)
        assert kept["payment"].isna().sum() == 0

        forced, _ = clean_dataframe(df, force_sentinels=["unknown", "none"])
        assert forced["payment"].isna().sum() == 2

    def test_preserve_tokens_keeps_what_the_heuristic_stripped(self):
        # drop_empty is off: these frames are one column wide, so a nulled cell
        # would otherwise take its whole row with it.
        df = pd.DataFrame({"Item": ["Coffee", "UNKNOWN", "Cake", "Tea"] * 5})
        stripped, _ = clean_dataframe(df, drop_empty=False)
        assert stripped["Item"].isna().sum() == 5

        kept, _ = clean_dataframe(df, preserve_tokens=["unknown"], drop_empty=False)
        assert kept["Item"].isna().sum() == 0

    def test_preserving_beats_forcing(self):
        # Conflicting instructions resolve toward keeping data.
        df = pd.DataFrame({"Item": ["Coffee", "UNKNOWN", "Cake", "Tea"] * 5})
        out, _ = clean_dataframe(
            df, force_sentinels=["unknown"], preserve_tokens=["unknown"],
            drop_empty=False,
        )
        assert out["Item"].isna().sum() == 0

    def test_overrides_are_case_insensitive(self):
        df = pd.DataFrame({"payment": ["Cash", "unknown", "Card"]})
        out, _ = clean_dataframe(df, force_sentinels=["UNKNOWN"], drop_empty=False)
        assert out["payment"].isna().sum() == 1

    def test_mode_no_longer_fills_with_a_sentinel(self):
        # The original bug: mode imputation filled a gap with "-", another
        # missing marker, because "-" was counted as a real category.
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "score": ["3.5", "4.0", None, "-", "-"],
        })
        out, _ = clean_dataframe(df)
        assert out["score"].isna().sum() == 3
        assert "-" not in out["score"].dropna().tolist()


# ---------------------------------------------------------------------------
# Headers and structure
# ---------------------------------------------------------------------------


class TestStructuralCleaning:
    """Headers, duplicate labels, leftover index columns and empty regions."""

    def test_headers_are_trimmed(self):
        df = pd.DataFrame({" age ": [1], "na  me": [2]})
        out, report = clean_dataframe(df)
        assert list(out.columns) == ["age", "na me"]
        assert any(a.step == "normalize_headers" for a in report.actions)

    def test_duplicate_column_labels_are_renamed(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
        out, _ = deduplicate_columns(df, CleaningReport()), None
        assert len(set(out.columns)) == 3
        assert isinstance(out["a"], pd.Series)

    def test_leftover_index_column_is_dropped(self):
        df = pd.DataFrame({"Unnamed: 0": [0, 1, 2], "value": [9, 8, 7]})
        out, _ = clean_dataframe(df)
        assert "Unnamed: 0" not in out.columns
        assert "value" in out.columns

    def test_unnamed_column_with_real_data_is_kept(self):
        # Not an index: values are not a 0/1-based unique run.
        df = pd.DataFrame({"Unnamed: 0": [5, 5, 9], "value": [9, 8, 7]})
        out, _ = clean_dataframe(df)
        assert "Unnamed: 0" in out.columns

    def test_fully_empty_column_is_dropped(self):
        df = pd.DataFrame({"keep": [1, 2], "empty": [None, None]})
        out, report = clean_dataframe(df)
        assert list(out.columns) == ["keep"]
        assert any(a.step == "drop_empty_columns" for a in report.actions)

    def test_fully_empty_row_is_dropped(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
        out, _ = clean_dataframe(df)
        assert len(out) == 2

    def test_whitespace_in_cells_is_trimmed(self):
        df = pd.DataFrame({"city": ["Delhi ", " Delhi", "Delhi"]})
        out, _ = clean_dataframe(df)
        assert out["city"].nunique() == 1


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


class TestNumericCoercion:
    """Numbers stored as text must become numbers."""

    def test_thousands_separator_and_currency(self):
        df = pd.DataFrame({"salary": ["$50,000", "60000", "70,500"]})
        out, _ = clean_dataframe(df)
        assert pd.api.types.is_integer_dtype(out["salary"].dtype)
        assert out["salary"].tolist() == [50000, 60000, 70500]

    def test_accounting_negative(self):
        df = pd.DataFrame({"pnl": ["(1,200)", "3400", "(50)"]})
        out, _ = clean_dataframe(df)
        assert out["pnl"].tolist() == [-1200, 3400, -50]

    def test_percentages_scale_to_fractions(self):
        df = pd.DataFrame({"rate": ["45%", "50%", "12.5%"]})
        out, _ = clean_dataframe(df)
        assert out["rate"].tolist() == pytest.approx([0.45, 0.50, 0.125])

    def test_leading_zeros_are_preserved(self):
        # Zip codes and account numbers must not become integers.
        df = pd.DataFrame({"zip": ["01234", "02115", "90210"]})
        out, _ = clean_dataframe(df)
        assert out["zip"].dtype == object
        assert out["zip"].iloc[0] == "01234"

    def test_mostly_text_column_is_left_alone(self):
        df = pd.DataFrame({"note": ["alpha", "beta", "7", "gamma", "delta"]})
        out, _ = clean_dataframe(df)
        assert out["note"].dtype == object

    def test_integer_column_with_gaps_becomes_nullable_int(self):
        # pandas floats an int column that merely has a blank; without this the
        # mean of an age column would be written back as 32.5.
        df = pd.DataFrame({"age": [25.0, np.nan, 30.0, 35.0]})
        out, report = downcast_integral_floats(df, CleaningReport()), None
        assert pd.api.types.is_integer_dtype(out["age"].dtype)

    def test_genuine_float_column_stays_float(self):
        df = pd.DataFrame({"price": [10.5, np.nan, 20.25]})
        out, _ = downcast_integral_floats(df, CleaningReport()), None
        assert pd.api.types.is_float_dtype(out["price"].dtype)

    def test_complete_float_column_is_untouched(self):
        # No nulls means the float dtype was not forced by missingness.
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        out, _ = downcast_integral_floats(df, CleaningReport()), None
        assert pd.api.types.is_float_dtype(out["v"].dtype)


class TestDatetimeCoercion:
    """Dates stored as text must become datetimes, but codes must not."""

    def test_iso_and_slash_dates_are_parsed(self):
        df = pd.DataFrame({"signup": ["2021-01-05", "2022/03/14", "2020-11-30"]})
        out, _ = clean_dataframe(df)
        assert pd.api.types.is_datetime64_any_dtype(out["signup"].dtype)

    def test_plain_integers_are_not_treated_as_dates(self):
        df = pd.DataFrame({"code": ["1001", "1002", "1003"]})
        out, _ = clean_dataframe(df)
        assert not pd.api.types.is_datetime64_any_dtype(out["code"].dtype)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class TestCleanDataframe:
    """The full sequence, and its reporting."""

    def test_report_records_every_action(self):
        df = pd.DataFrame({" a ": ["1", "?", "3"], "empty": [None, None, None]})
        out, report = clean_dataframe(df)
        steps = {a.step for a in report.actions}
        assert "normalize_headers" in steps
        assert "normalize_missing_sentinels" in steps
        assert "drop_empty_columns" in steps
        assert all(isinstance(d, dict) for d in report.to_dicts())

    def test_clean_frame_produces_no_actions(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out, report = clean_dataframe(df)
        assert report.is_empty
        pd.testing.assert_frame_equal(out, df)

    def test_toggles_disable_stages(self):
        df = pd.DataFrame({"a": ["1", "?", "3"]})
        out, _ = clean_dataframe(df, normalize_sentinels=False, coerce_numeric=False)
        assert "?" in out["a"].tolist()

    def test_empty_frame_is_handled(self):
        out, report = clean_dataframe(pd.DataFrame())
        assert out.empty
        assert report.is_empty

    def test_column_of_only_sentinels_is_dropped(self):
        # Becomes fully empty only after sentinel normalisation, which is why
        # the empty-column drop runs last.
        df = pd.DataFrame({"keep": [1, 2], "junk": ["?", "N/A"]})
        out, _ = clean_dataframe(df)
        assert "junk" not in out.columns


class TestNullRepresentation:
    """Nulls must be the flavour the rest of the stack can consume.

    scikit-learn's imputers test object arrays with ``X != X``. ``pd.NA``
    raises "boolean value of NA is ambiguous" there, so training on a freshly
    loaded (unrepaired) file crashed. Both read as null to pandas; only
    ``np.nan`` survives a trip through a model pipeline.
    """

    def test_object_columns_use_nan_not_pd_na(self):
        df = pd.DataFrame({"id": [1, 2, 3], "city": ["Delhi", "UNKNOWN", "Pune"]})
        out, _ = clean_dataframe(df)
        nulls = out["city"][out["city"].isna()]
        assert len(nulls) == 1
        assert all(v is not pd.NA for v in nulls)
        assert all(isinstance(v, float) for v in nulls)

    def test_cleaned_frame_survives_sklearn_imputation(self):
        from sklearn.impute import SimpleImputer

        df = pd.DataFrame({
            "city": ["Delhi", "ERROR", "Pune", "UNKNOWN", "Delhi"],
            "note": ["a", "b", "?", "d", "e"],
        })
        out, _ = clean_dataframe(df)
        # This raised TypeError: boolean value of NA is ambiguous.
        imputed = SimpleImputer(strategy="most_frequent").fit_transform(out)
        assert imputed.shape == out.shape
