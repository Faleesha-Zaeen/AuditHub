"""
AuditHub Tests - ML Module
============================

Tests for MLTrainingEngine, MLEvaluator, and MLflowTracker logging.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.ml import MLTrainingEngine, MLEvaluator, MLflowTracker


def test_ml_training_classification(config_manager):
    """Test full classification Auto-ML training loop."""
    # Build synthetic classification dataset
    df = pd.DataFrame({
        "feat_1": np.random.normal(0.0, 1.0, size=100),
        "feat_2": np.random.normal(5.0, 2.0, size=100),
        "cat_feat": np.random.choice(["cat1", "cat2", "cat3"], size=100),
        "target": np.random.choice([0, 1], size=100),
    })

    trainer = MLTrainingEngine(seed=42)
    model, metrics, save_path = trainer.train(df, target_column="target", dataset_name="synthetic_cls")

    assert model is not None
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert Path(save_path).exists()
    assert Path(save_path).suffix == ".joblib"


def test_ml_training_regression(config_manager):
    """Test full regression Auto-ML training loop."""
    # Build synthetic regression dataset
    df = pd.DataFrame({
        "feat_1": np.random.normal(0.0, 1.0, size=100),
        "feat_2": np.random.normal(5.0, 2.0, size=100),
        "target": np.random.normal(10.0, 5.0, size=100),
    })

    trainer = MLTrainingEngine(seed=42)
    model, metrics, save_path = trainer.train(df, target_column="target", dataset_name="synthetic_reg")

    assert model is not None
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2_score" in metrics
    assert Path(save_path).exists()
