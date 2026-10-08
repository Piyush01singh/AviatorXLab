"""Unit tests for expanding-window walk-forward validation engine."""

import numpy as np
import pandas as pd
import pytest

from src.training.walk_forward import WalkForwardEngine


def test_walk_forward_engine_execution(tmp_path):
    # Create small synthetic dataframe for fast testing
    np.random.seed(42)
    n = 600
    features = pd.DataFrame(
        np.random.randn(n, 5),
        columns=[f"feat_{i}" for i in range(5)],
    )
    targets = pd.DataFrame({
        "target_2x": np.random.binomial(1, 0.48, size=n),
    })

    engine = WalkForwardEngine()
    engine.reports_dir = tmp_path

    # Run with small window configuration for speed
    folds_df, summary_df = engine.run_walk_forward(
        X=features,
        y_thresh=targets,
        target_col="target_2x",
        initial_train_size=300,
        step_size=100,
        test_size=100,
    )

    assert not folds_df.empty
    assert not summary_df.empty
    assert "fold" in folds_df.columns
    assert "model" in folds_df.columns
    assert "roc_auc" in folds_df.columns

    models_evaluated = set(summary_df["model"])
    # Check that baseline and ML models were evaluated
    assert "Historical Baseline" in models_evaluated
    assert "Logistic Regression" in models_evaluated
    assert "Random Forest" in models_evaluated
    assert "XGBoost" in models_evaluated
    assert "MLP" in models_evaluated
    assert "LSTM" in models_evaluated
    assert "GRU" in models_evaluated
