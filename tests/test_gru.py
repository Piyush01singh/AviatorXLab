"""Unit tests for PyTorch GRU model."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.gru import GRUNetwork, GRUThresholdClassifier


def test_gru_network_forward_and_causality():
    net = GRUNetwork(input_dim=8, hidden_size=32, num_layers=1, dropout=0.0, output_dim=5)
    assert net.gru.bidirectional is False

    x = torch.randn(4, 10, 8)
    logits = net(x)
    assert logits.shape == (4, 5)


def test_gru_classifier_inference():
    n_samples = 20
    seq_len = 10
    n_features = 6
    X_seq = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)

    model = GRUThresholdClassifier(
        sequence_length=seq_len,
        hidden_size=32,
        device_preference="cpu",
    )
    model.model = GRUNetwork(input_dim=n_features, hidden_size=32, output_dim=5).to("cpu")
    model.is_fitted = True

    probs = model.predict_proba(X_seq)
    assert probs.shape == (n_samples, 5)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_gru_save_and_load():
    n_samples = 15
    seq_len = 5
    n_features = 4
    X_seq = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)

    model = GRUThresholdClassifier(sequence_length=seq_len, hidden_size=16, device_preference="cpu")
    model.model = GRUNetwork(input_dim=n_features, hidden_size=16, output_dim=5).to("cpu")
    model.is_fitted = True

    probs = model.predict_proba(X_seq)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "gru_test.joblib"
        model.save(str(path))

        loaded = GRUThresholdClassifier.load(str(path))
        loaded_probs = loaded.predict_proba(X_seq)
        np.testing.assert_allclose(probs, loaded_probs, atol=1e-5)
