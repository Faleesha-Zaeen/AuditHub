"""
AuditHub Tests - Mutation Lab Module
======================================

Tests for the DatasetMutator class, including corruption and reproducibility verification.
"""

import pytest
import pandas as pd
import numpy as np

from src.mutation_lab import DatasetMutator


def test_mutator_missing_injection(sample_dataframe):
    """Test injecting missing values."""
    mutator = DatasetMutator(seed=42)
    df_mut = mutator.inject_missing_values(sample_dataframe, fraction=0.33, columns=["value"])

    # Expect at least one value in column 'value' to be null
    assert df_mut["value"].isna().sum() >= 1


def test_mutator_duplicates(sample_dataframe):
    """Test injecting duplicate rows."""
    mutator = DatasetMutator(seed=42)
    df_mut = mutator.inject_duplicates(sample_dataframe, fraction=0.5)

    assert len(df_mut) == len(sample_dataframe) + 1


def test_mutator_outliers(sample_dataframe):
    """Test outlier injection in numeric columns."""
    mutator = DatasetMutator(seed=42)
    df_mut = mutator.inject_outliers(sample_dataframe, fraction=0.33, columns=["value"])

    # Maximum value should increase drastically due to outlier addition
    assert df_mut["value"].max() > sample_dataframe["value"].max() * 2


def test_mutator_reproducibility(sample_dataframe):
    """Test seed-based reproducibility."""
    mutator1 = DatasetMutator(seed=100)
    df_mut1 = mutator1.inject_gaussian_noise(sample_dataframe, noise_level=0.1)

    mutator2 = DatasetMutator(seed=100)
    df_mut2 = mutator2.inject_gaussian_noise(sample_dataframe, noise_level=0.1)

    # Identical seeds must produce identical values
    pd.testing.assert_frame_equal(df_mut1, df_mut2)

    # Different seeds must produce different values
    mutator3 = DatasetMutator(seed=200)
    df_mut3 = mutator3.inject_gaussian_noise(sample_dataframe, noise_level=0.1)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df_mut1, df_mut3)
