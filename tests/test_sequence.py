"""Unit tests for sequential window tensor generation."""

import numpy as np
import pytest
from src.features.sequential import create_sequence_windows


def test_create_sequence_windows():
    n_samples = 25
    n_features = 4
    seq_len = 5

    X = np.arange(n_samples * n_features).reshape(n_samples, n_features)
    y = np.arange(n_samples)

    X_seq, y_seq = create_sequence_windows(X, y, sequence_length=seq_len)

    # For n=25 and seq_len=5, targets are y[5] through y[24] -> 20 targets
    expected_windows = n_samples - seq_len
    assert X_seq.shape == (expected_windows, seq_len, n_features)
    assert y_seq.shape == (expected_windows,)

    # Verify first window corresponds to target y[5] and contains X[0:5] (i.e. rows 0,1,2,3,4)
    np.testing.assert_array_equal(X_seq[0], X[0:5])
    assert y_seq[0] == y[5]

    # Verify last window corresponds to target y[24] and contains X[19:24] (i.e. rows 19..23)
    np.testing.assert_array_equal(X_seq[-1], X[19:24])
    assert y_seq[-1] == y[24]
