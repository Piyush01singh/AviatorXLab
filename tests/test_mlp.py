"""Unit tests for PyTorch Multi-Layer Perceptron (MLP)."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.mlp import MLPNetwork, MLPThresholdClassifier


@pytest.fixture
def toy_tabular_data():
    np.random.seed(42)
    n = 80
    d = 10
    X = pd.DataFrame(np.random.randn(n, d), columns=[f"feat_{i}" for i in range(d)])
    # 5 binary threshold targets
    y = pd.DataFrame({
        "t1": (X["feat_0"] > 0).astype(int),
        "t2": (X["feat_1"] > 0).astype(int),
        "t3": (X["feat_2"] > 0).astype(int),
        "t4": (X["feat_3"] > 0).astype(int),
        "t5": (X["feat_4"] > 0).astype(int),
    })
    return X, y


def test_mlp_network_forward():
    net = MLPNetwork(input_dim=10, hidden_dims=[32, 16], dropout=0.1, output_dim=5)
    x = torch.randn(8, 10)
    out = net(x)
    assert out.shape == (8, 5)


def test_mlp_classifier_training_and_probabilities(toy_tabular_data):
    X, y = toy_tabular_data
    model = MLPThresholdClassifier(
        hidden_dims=[32, 16],
        epochs=5,
        batch_size=16,
        early_stopping_patience=3,
        device_preference="cpu",
    )
    model.fit(X, y, X_val=X, y_val=y)

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 5)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    preds = model.predict(X, threshold=0.5)
    assert preds.shape == (len(X), 5)
    assert set(np.unique(preds)).issubset({0, 1})


def test_mlp_save_and_load(toy_tabular_data):
    X, y = toy_tabular_data
    model = MLPThresholdClassifier(hidden_dims=[16, 8], epochs=3, batch_size=16, device_preference="cpu")
    model.fit(X, y)
    probs = model.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mlp_test.joblib"
        model.save(str(path))

        loaded = MLPThresholdClassifier.load(str(path))
        loaded_probs = loaded.predict_proba(X)
        np.testing.assert_allclose(probs, loaded_probs, atol=1e-5)
