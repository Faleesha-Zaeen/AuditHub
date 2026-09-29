"""
Tests for src.repair.relationships
===================================

Arithmetic relationship discovery turns a guess into a computation: when
``total = quantity * price`` holds across a file, a missing total is not a
median to be estimated, it is a number to be worked out.

The fixtures here deliberately span unrelated domains -- invoicing, payroll,
physics, budgeting -- because the discovery must come from the data, not from
column names. Nothing in the module knows what a "quantity" is.
"""

import numpy as np
import pandas as pd
import pytest

from src.repair.relationships import (
    ColumnRelationship,
    apply_relationships,
    check_violations,
    discover_relationships,
)


# ---------------------------------------------------------------------------
# Fixtures - three unrelated domains, three different identities
# ---------------------------------------------------------------------------


@pytest.fixture
def invoice_frame() -> pd.DataFrame:
    """``amount = units * unit_price``, with prices that repeat and units that vary."""
    rng = np.random.default_rng(11)
    units = rng.integers(1, 9, 200)
    price = rng.choice([1.5, 2.0, 3.25, 7.0], 200)
    return pd.DataFrame({
        "invoice_id": [f"INV{i:04d}" for i in range(200)],
        "units": units,
        "unit_price": price,
        "amount": units * price,
    })


@pytest.fixture
def payroll_frame() -> pd.DataFrame:
    """``gross = base + bonus`` -- an additive identity, not multiplicative."""
    rng = np.random.default_rng(23)
    base = rng.integers(30000, 90000, 150).astype(float)
    bonus = rng.integers(0, 15000, 150).astype(float)
    return pd.DataFrame({
        "employee": [f"E{i:03d}" for i in range(150)],
        "base": base,
        "bonus": bonus,
        "gross": base + bonus,
    })


@pytest.fixture
def physics_frame() -> pd.DataFrame:
    """``distance = speed * time`` with float operands throughout."""
    rng = np.random.default_rng(37)
    speed = rng.uniform(5, 120, 120).round(2)
    time = rng.uniform(0.5, 8, 120).round(2)
    return pd.DataFrame({
        "run": range(120),
        "speed": speed,
        "time": time,
        "distance": speed * time,
    })


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    """Finding the identity, and refusing to invent one."""

    def test_multiplicative_identity_is_found(self, invoice_frame):
        found = discover_relationships(invoice_frame)
        assert len(found) == 1
        assert found[0].target == "amount"
        assert found[0].operator == "*"
        assert {found[0].left, found[0].right} == {"units", "unit_price"}
        assert found[0].confidence == pytest.approx(1.0)

    def test_additive_identity_is_found(self, payroll_frame):
        found = discover_relationships(payroll_frame)
        assert len(found) == 1
        assert found[0].target == "gross"
        assert found[0].operator == "+"

    def test_float_operands_are_handled(self, physics_frame):
        found = discover_relationships(physics_frame)
        assert len(found) == 1
        assert found[0].expression.startswith("distance = ")

    def test_unrelated_columns_yield_nothing(self):
        # Three independent columns. Any identity found here would be noise, and
        # acting on noise would overwrite good data with fabrications.
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "a": rng.normal(size=300),
            "b": rng.normal(size=300),
            "c": rng.normal(size=300),
        })
        assert discover_relationships(df) == []

    def test_coincidence_on_few_rows_is_not_extrapolated(self):
        # The identity holds on all 8 rows, but 8 rows is not evidence. Below
        # min_support the relationship is never trusted.
        df = pd.DataFrame({
            "q": [1, 2, 3, 4, 5, 6, 7, 8],
            "p": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        })
        df["t"] = df["q"] * df["p"]
        assert discover_relationships(df) == []

    def test_near_miss_relationship_is_rejected(self):
        # 5% of rows use a different formula (a discount was applied). That is a
        # different business rule, not float noise, so the identity must not be
        # used to overwrite the other 95%.
        rng = np.random.default_rng(13)
        q = rng.integers(1, 10, 400)
        p = rng.choice([2.0, 4.0], 400)
        t = (q * p).astype(float)
        t[:20] *= 0.9
        df = pd.DataFrame({"q": q, "p": p, "t": t})
        assert discover_relationships(df) == []

    def test_constant_operand_is_not_a_relationship(self):
        # "total = x * 1" is arithmetically true and completely useless: it
        # says the two columns are copies, not that a rule governs them.
        df = pd.DataFrame({
            "x": np.arange(1, 201, dtype=float),
            "one": np.ones(200),
        })
        df["total"] = df["x"] * df["one"]
        assert discover_relationships(df) == []

    def test_fewer_than_three_numeric_columns_returns_empty(self):
        df = pd.DataFrame({"a": range(100), "b": range(100), "label": ["x"] * 100})
        assert discover_relationships(df) == []

    def test_empty_frame_returns_empty(self):
        assert discover_relationships(pd.DataFrame()) == []

    def test_gaps_do_not_prevent_discovery(self, invoice_frame):
        # The identity is established from complete rows; incomplete rows are
        # what the identity then goes on to fill.
        df = invoice_frame.copy()
        df.loc[df.index[:40], "amount"] = np.nan
        found = discover_relationships(df)
        assert len(found) == 1
        assert found[0].support == 160


# ---------------------------------------------------------------------------
# The relationship object
# ---------------------------------------------------------------------------


class TestColumnRelationship:
    """Evaluating and rearranging a discovered identity."""

    @pytest.fixture
    def relationship(self) -> ColumnRelationship:
        return ColumnRelationship(
            target="amount", left="units", right="unit_price",
            operator="*", support=200, confidence=1.0,
        )

    def test_expression_is_readable(self, relationship):
        assert relationship.expression == "amount = units * unit_price"

    def test_solves_for_target(self, relationship, invoice_frame):
        solved = relationship.solve_for(invoice_frame, "amount")
        assert np.allclose(solved, invoice_frame["amount"])

    def test_solves_for_each_operand(self, relationship, invoice_frame):
        assert np.allclose(
            relationship.solve_for(invoice_frame, "units"), invoice_frame["units"]
        )
        assert np.allclose(
            relationship.solve_for(invoice_frame, "unit_price"),
            invoice_frame["unit_price"],
        )

    def test_unknown_column_returns_none(self, relationship, invoice_frame):
        assert relationship.solve_for(invoice_frame, "invoice_id") is None

    def test_division_by_zero_yields_null_not_infinity(self, relationship):
        # A zero-priced line cannot reveal how many units were sold. The answer
        # is "unknown", not "inf" -- an infinity would poison every downstream
        # statistic it touched.
        df = pd.DataFrame({"units": [np.nan], "unit_price": [0.0], "amount": [0.0]})
        solved = relationship.solve_for(df, "units")
        assert solved.isna().all()

    def test_to_dict_is_serialisable(self, relationship):
        import json
        payload = relationship.to_dict()
        assert json.loads(json.dumps(payload))["expression"] == relationship.expression


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------


class TestApplyRelationships:
    """Deriving the exact value instead of estimating one."""

    def test_missing_target_is_computed(self, invoice_frame):
        df = invoice_frame.copy()
        truth = df["amount"].copy()
        df.loc[df.index[:50], "amount"] = np.nan

        filled, records = apply_relationships(df, discover_relationships(df))
        assert filled["amount"].notna().all()
        assert np.allclose(filled["amount"], truth)
        assert sum(r["cells_filled"] for r in records) == 50

    def test_missing_operand_is_computed(self, invoice_frame):
        df = invoice_frame.copy()
        truth = df["unit_price"].copy()
        df.loc[df.index[:50], "unit_price"] = np.nan

        filled, _ = apply_relationships(df, discover_relationships(df))
        assert np.allclose(filled["unit_price"], truth)

    def test_additive_identity_fills_both_directions(self, payroll_frame):
        df = payroll_frame.copy()
        truth = df.copy()
        df.loc[df.index[:30], "gross"] = np.nan
        df.loc[df.index[30:60], "bonus"] = np.nan

        filled, _ = apply_relationships(df, discover_relationships(df))
        assert np.allclose(filled["gross"], truth["gross"])
        assert np.allclose(filled["bonus"], truth["bonus"])

    def test_row_missing_two_of_three_is_left_alone(self, invoice_frame):
        # One equation, two unknowns. Leaving the gap for imputation is honest;
        # inventing a split would not be.
        df = invoice_frame.copy()
        df.loc[df.index[0], ["amount", "units"]] = np.nan

        filled, _ = apply_relationships(df, discover_relationships(df))
        assert filled.loc[df.index[0], "units"] != filled.loc[df.index[0], "units"]

    def test_integer_column_only_takes_whole_answers(self):
        # 8.0 / 3.0 = 2.667. Rounding to 3 would fabricate a value *and* break
        # the identity it came from, so the gap survives for imputation.
        rng = np.random.default_rng(3)
        q = rng.integers(1, 6, 300)
        p = rng.choice([2.0, 4.0, 5.0], 300)
        df = pd.DataFrame({"q": q, "p": p, "t": q * p})
        df["q"] = df["q"].astype("Int64")

        df.loc[df.index[0], "q"] = pd.NA
        df.loc[df.index[0], ["p", "t"]] = [3.0, 8.0]

        filled, _ = apply_relationships(df, discover_relationships(df))
        assert pd.isna(filled.loc[df.index[0], "q"])

    def test_integer_column_accepts_an_exact_answer(self):
        rng = np.random.default_rng(4)
        q = rng.integers(1, 6, 300)
        p = rng.choice([2.0, 4.0, 5.0], 300)
        df = pd.DataFrame({"q": q, "p": p, "t": q * p})
        df["q"] = df["q"].astype("Int64")

        df.loc[df.index[0], "q"] = pd.NA
        df.loc[df.index[0], ["p", "t"]] = [4.0, 12.0]

        filled, _ = apply_relationships(df, discover_relationships(df))
        assert filled.loc[df.index[0], "q"] == 3

    def test_derived_integers_stay_integers(self):
        rng = np.random.default_rng(9)
        q = rng.integers(1, 6, 300)
        p = rng.choice([2.0, 4.0], 300)
        df = pd.DataFrame({"q": q, "p": p, "t": q * p})
        df["q"] = df["q"].astype("Int64")
        df.loc[df.index[:20], "q"] = pd.NA

        filled, _ = apply_relationships(df, discover_relationships(df))
        assert pd.api.types.is_integer_dtype(filled["q"].dtype)

    def test_second_pass_uses_the_first_pass_result(self):
        # Two chained identities: subtotal = units * price, and
        # total = subtotal + shipping. Filling subtotal is what makes total
        # computable, so a single pass would leave the second gap open.
        rng = np.random.default_rng(21)
        units = rng.integers(1, 9, 400)
        price = rng.choice([2.0, 5.0], 400)
        shipping = rng.choice([0.0, 4.99, 9.99], 400)
        df = pd.DataFrame({"units": units, "price": price, "shipping": shipping})
        df["subtotal"] = df["units"] * df["price"]
        df["total"] = df["subtotal"] + df["shipping"]

        df.loc[df.index[0], ["subtotal", "total"]] = np.nan
        found = discover_relationships(df)
        filled, _ = apply_relationships(df, found)

        assert filled.loc[df.index[0], "subtotal"] == units[0] * price[0]
        assert filled.loc[df.index[0], "total"] == units[0] * price[0] + shipping[0]

    def test_no_relationships_is_a_no_op(self, invoice_frame):
        out, records = apply_relationships(invoice_frame, [])
        assert records == []
        pd.testing.assert_frame_equal(out, invoice_frame)

    def test_input_frame_is_not_mutated(self, invoice_frame):
        df = invoice_frame.copy()
        df.loc[df.index[:10], "amount"] = np.nan
        before = df.copy()

        apply_relationships(df, discover_relationships(df))
        pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


class TestCheckViolations:
    """Rows where every value is present but the arithmetic disagrees."""

    def test_consistent_frame_reports_nothing(self, invoice_frame):
        found = discover_relationships(invoice_frame)
        assert check_violations(invoice_frame, found) == []

    def test_corrupted_rows_are_reported(self, invoice_frame):
        found = discover_relationships(invoice_frame)
        df = invoice_frame.copy()
        df.loc[df.index[:7], "amount"] = 999.0

        findings = check_violations(df, found)
        assert len(findings) == 1
        assert findings[0]["violations"] == 7
        assert findings[0]["checked"] == 200
        assert len(findings[0]["example_rows"]) <= 5

    def test_missing_column_is_skipped_not_crashed(self, invoice_frame):
        found = discover_relationships(invoice_frame)
        assert check_violations(invoice_frame.drop(columns=["amount"]), found) == []


# ---------------------------------------------------------------------------
# End to end, through the repairer
# ---------------------------------------------------------------------------


class TestRepairerIntegration:
    """Derivation must beat imputation where the answer is computable."""

    def test_repairer_derives_rather_than_imputes(self, invoice_frame):
        from src.repair.repairer import DatasetRepairer

        df = invoice_frame.copy()
        truth = df["amount"].copy()
        df.loc[df.index[:60], "amount"] = np.nan

        repairer = DatasetRepairer(df)
        result = repairer.auto_repair()

        assert result.cells_derived == 60
        assert np.allclose(repairer.df["amount"], truth)

    def test_median_would_have_been_wrong(self, invoice_frame):
        # The point of the whole module: the median is a plausible number that
        # is simply not the one that was on the receipt.
        from src.repair.repairer import DatasetRepairer

        df = invoice_frame.copy()
        truth = df["amount"].copy()
        df.loc[df.index[:60], "amount"] = np.nan
        median = df["amount"].median()

        repairer = DatasetRepairer(df)
        repairer.auto_repair()

        derived = repairer.df["amount"].iloc[:60]
        assert np.allclose(derived, truth.iloc[:60])
        assert not np.allclose(derived, median)

    def test_dataset_without_relationships_still_repairs(self):
        # Most files have no arithmetic identity at all. Discovery costs one
        # pass and then gets out of the way.
        from src.repair.repairer import DatasetRepairer

        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "age": rng.integers(18, 80, 200).astype(float),
            "city": rng.choice(["Delhi", "Pune", "Chennai"], 200),
            "score": rng.normal(50, 12, 200),
        })
        df.loc[df.index[:20], "age"] = np.nan
        df.loc[df.index[20:40], "city"] = None

        repairer = DatasetRepairer(df)
        result = repairer.auto_repair()

        assert result.relationships == []
        assert result.cells_derived == 0
        assert repairer.df.isna().sum().sum() == 0
