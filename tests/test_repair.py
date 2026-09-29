"""
AuditHub Tests - Repair Module
================================

Tests for the DatasetRepairer class, repair operations, previews, and rollback functions.
"""

import pytest
import pandas as pd
import numpy as np

from src.repair import DatasetRepairer


def test_repairer_duplicates():
    """Test removing duplicate rows."""
    df = pd.DataFrame({
        "a": [1, 2, 2, 3],
        "b": [4, 5, 5, 6],
    })

    repairer = DatasetRepairer(df)
    assert len(repairer.df) == 4

    # Apply duplicate removal
    repairer.remove_duplicates()
    assert len(repairer.df) == 3

    # Revert duplicate removal
    repairer.revert()
    assert len(repairer.df) == 4


def test_repairer_imputation():
    """Test missing value imputation strategies."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": ["x", "y", "y", None],
    })

    # Test mean imputation
    repairer = DatasetRepairer(df)
    repairer.impute_missing("a", strategy="mean")
    assert repairer.df["a"].isna().sum() == 0
    assert repairer.df["a"].iloc[2] == 2.3333333333333335  # Mean of [1, 2, 4]

    # Test constant imputation
    repairer.impute_missing("b", strategy="constant", fill_value="missing")
    assert repairer.df["b"].isna().sum() == 0
    assert repairer.df["b"].iloc[3] == "missing"


def test_repairer_revert_chain():
    """Test sequential reverts (undoing multiple actions)."""
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": [10.0, 10.0, np.nan],
    })

    repairer = DatasetRepairer(df)

    repairer.remove_duplicates()  # Action 1
    repairer.impute_missing("b", strategy="median")  # Action 2

    assert len(repairer.df) == 2
    assert repairer.df["b"].isna().sum() == 0

    # Revert Action 2 (imputation)
    repairer.revert()
    assert len(repairer.df) == 2
    assert repairer.df["b"].isna().sum() == 1

    # Revert Action 1 (remove_duplicates)
    repairer.revert()
    assert len(repairer.df) == 3
    assert repairer.df["b"].isna().sum() == 1


def test_repairer_preview():
    """Test that preview_repair returns the repaired dataframe without changing internal state."""
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
    })

    repairer = DatasetRepairer(df)
    preview = repairer.preview_repair("impute_missing", column="a", strategy="mean")

    # Preview should be repaired
    assert preview["a"].isna().sum() == 0
    # Original state in repairer should remain unrepaired
    assert repairer.df["a"].isna().sum() == 1
