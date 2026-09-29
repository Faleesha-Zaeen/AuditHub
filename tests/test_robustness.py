"""
AuditHub Tests - Robustness Evaluation Module
===============================================

Tests for the RobustnessEvaluator, ensuring model decay curves and elasticity scores.
"""

import pytest
import pandas as pd
import numpy as np

from src.robustness import RobustnessEvaluator


def test_robustness_evaluator_classification(sample_dataframe):
    """Test evaluating robustness of a classification model under missing values."""
    # Let's augment the sample dataframe to make it robustly trainable
    df = pd.DataFrame({
        "feature_1": np.random.uniform(0.0, 1.0, size=50),
        "feature_2": np.random.uniform(10.0, 20.0, size=50),
        "target": np.random.choice([0, 1], size=50),
    })

    evaluator = RobustnessEvaluator(seed=42)
    results = evaluator.evaluate_robustness(
        df,
        target_column="target",
        mutation_type="missing_values",
        levels=[0.0, 0.1, 0.3]
    )

    assert isinstance(results, dict)
    assert results["task_type"] == "classification"
    assert results["mutation_type"] == "missing_values"
    assert results["metric_name"] == "accuracy"
    assert "baseline_performance" in results
    assert "elasticity_score" in results
    assert 0.0 <= results["elasticity_score"] <= 1.0
