"""
Tests for src.ingestion.text_columns
=====================================

The role a text column plays decides how it may be cleaned. Getting this wrong
in either direction is costly: treat prose as categorical and the repairer
copies one customer's review into another's empty field; treat a category as
prose and a perfectly good mode fill is refused.
"""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.text_columns import (
    casing_class,
    classify_text_column,
    dominant_casing,
    is_free_text,
    is_imputable_text,
)


class TestClassifyTextColumn:
    """Categorical, free text, identifier, or not text at all."""

    def test_short_labels_are_categorical(self):
        s = pd.Series(["Cash", "Card", "Cash", "Digital Wallet"] * 25)
        assert classify_text_column(s) == "categorical"

    def test_two_word_categories_stay_categorical(self):
        # "Digital Wallet" and "In-store" must not trip the prose detector, or
        # the cafe file's payment column would stop being imputable.
        s = pd.Series(["Digital Wallet", "In-store", "Takeaway", "Credit Card"] * 25)
        assert classify_text_column(s) == "categorical"

    def test_sentences_are_free_text(self):
        s = pd.Series([
            "Great coffee, but the queue was long.",
            "Terrible service and I will not be returning.",
            "The croissant was excellent although a little cold.",
        ] * 20)
        assert classify_text_column(s) == "free_text"

    def test_long_single_values_are_free_text(self):
        s = pd.Series(["x" * 80] * 50)
        assert classify_text_column(s) == "free_text"

    def test_mostly_distinct_values_are_identifiers(self):
        s = pd.Series([f"user{i}@example.com" for i in range(100)])
        assert classify_text_column(s) == "identifier"

    def test_numeric_columns_are_not_text(self):
        assert classify_text_column(pd.Series([1, 2, 3])) == "non_text"
        assert classify_text_column(pd.Series([1.5, 2.5])) == "non_text"

    def test_datetime_columns_are_not_text(self):
        s = pd.Series(pd.to_datetime(["2023-01-01", "2023-02-01"]))
        assert classify_text_column(s) == "non_text"

    def test_all_null_column_is_categorical(self):
        # Nothing to judge; the harmless default keeps the column imputable.
        assert classify_text_column(pd.Series([None, None], dtype=object)) == "categorical"

    def test_small_column_is_not_called_an_identifier(self):
        # Three distinct values out of three is not evidence of an identifier.
        assert classify_text_column(pd.Series(["a", "b", "c"])) == "categorical"

    def test_repeated_names_are_categorical_not_identifiers(self):
        s = pd.Series(["Nguyen", "Ali", "Okafor", "MacLeod"] * 30)
        assert classify_text_column(s) == "categorical"


class TestConvenienceWrappers:
    """The two questions the rest of the codebase actually asks."""

    def test_is_free_text(self):
        assert is_free_text(pd.Series(["a sentence with several words in it"] * 30))
        assert not is_free_text(pd.Series(["Cash", "Card"] * 30))

    @pytest.mark.parametrize(
        "values,expected",
        [
            (["Cash", "Card"] * 30, True),
            (["a sentence with several words in it"] * 30, False),
            ([f"id-{i}" for i in range(100)], False),
            ([1, 2, 3] * 30, False),
        ],
    )
    def test_is_imputable_text(self, values, expected):
        assert is_imputable_text(pd.Series(values)) is expected


class TestCasing:
    """Capitalisation is the signal that separates a placeholder from an answer."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("UNKNOWN", "upper"),
            ("unknown", "lower"),
            ("Unknown", "title"),
            ("Not applicable", "mixed"),
            ("iPhone", "mixed"),
            ("123", "none"),
            ("-", "none"),
            ("", "none"),
        ],
    )
    def test_casing_class(self, value, expected):
        assert casing_class(value) == expected

    def test_dominant_casing_of_title_column(self):
        s = pd.Series(["Coffee", "Cake", "Cookie", "Tea"] * 10)
        assert dominant_casing(s) == "title"

    def test_dominant_casing_ignores_the_token_under_test(self):
        # The candidate must not vote on the convention it is judged against,
        # or a column full of UNKNOWN would declare itself an upper-case column
        # and thereby exonerate its own placeholders.
        s = pd.Series(["Coffee", "UNKNOWN", "UNKNOWN", "Cake"] * 10)
        assert dominant_casing(s, ignore={"unknown"}) == "title"

    def test_dominant_casing_of_numbers_is_none(self):
        assert dominant_casing(pd.Series(["1", "2", "3"])) is None

    def test_dominant_casing_of_empty_is_none(self):
        assert dominant_casing(pd.Series([], dtype=object)) is None


class TestRepairerUsesTheRoles:
    """The behaviour these roles exist to produce."""

    def test_prose_gaps_are_marked_not_invented(self):
        from src.repair.repairer import MISSING_TEXT_MARKER, DatasetRepairer

        reviews = [
            "Great coffee, but the queue was long.",
            "Terrible service and I will not be returning.",
            "The croissant was excellent although a little cold.",
        ]
        df = pd.DataFrame({
            "id": range(40),
            "review": [reviews[i % 3] if i % 4 else None for i in range(40)],
        })
        repairer = DatasetRepairer(df)
        result = repairer.auto_repair()

        gaps = df["review"].isna()
        assert (repairer.df.loc[gaps, "review"] == MISSING_TEXT_MARKER).all()
        assert "review" in result.marked
        assert "review" not in result.columns_imputed
        # No real review was copied into a gap.
        for text in reviews:
            assert (repairer.df.loc[gaps, "review"] == text).sum() == 0

    def test_identifier_gaps_are_marked_not_invented(self):
        from src.repair.repairer import MISSING_TEXT_MARKER, DatasetRepairer

        df = pd.DataFrame({
            "email": [f"user{i}@mail.com" if i % 5 else None for i in range(60)],
            "plan": ["free", "pro", "enterprise", "trial", "edu"] * 12,
            "seats": range(60),
        })
        repairer = DatasetRepairer(df)
        result = repairer.auto_repair()

        gaps = df["email"].isna()
        assert (repairer.df.loc[gaps, "email"] == MISSING_TEXT_MARKER).all()
        assert result.marked["email"]["role"] == "identifier"

    def test_categorical_gaps_are_still_mode_imputed(self):
        # The narrowing must not cost the ordinary case its estimate.
        from src.repair.repairer import DatasetRepairer

        df = pd.DataFrame({
            "payment": ["Cash", "Card", "Cash", None, "Cash"] * 20,
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
        })
        repairer = DatasetRepairer(df)
        result = repairer.auto_repair()

        assert "payment" in result.columns_imputed
        assert result.columns_imputed["payment"]["fill_value"] == "Cash"
        assert repairer.df["payment"].isna().sum() == 0

    def test_marked_cells_are_counted_apart_from_estimates(self):
        from src.repair.repairer import DatasetRepairer

        df = pd.DataFrame({
            "note": ["a fairly long free text note here"] * 30 + [None] * 10,
            "cat": (["x", "y"] * 20)[:40],
            "num": list(range(40)),
        })
        df.loc[df.index[:5], "cat"] = None
        repairer = DatasetRepairer(df)
        result = repairer.auto_repair()

        assert result.cells_marked == 10
        assert result.cells_imputed == 5
        assert result.to_dict()["cells_marked"] == 10
        assert "marked" in result.to_dict()
