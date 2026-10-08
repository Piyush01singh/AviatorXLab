"""Unit tests for Chronological and Walk-Forward Data Splitters."""

import pandas as pd
import pytest
from src.data.splitter import chronological_split, walk_forward_split


@pytest.fixture
def dummy_timeseries_df():
    n = 100
    return pd.DataFrame({
        "round_id": range(1000, 1000 + n),
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="15s"),
        "multiplier": [1.5 + (i % 5) for i in range(n)],
    })


def test_chronological_split_strict_ordering_and_no_overlap(dummy_timeseries_df):
    split = chronological_split(dummy_timeseries_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    assert len(split.train) == 70
    assert len(split.val) == 15
    assert len(split.test) == 15

    # Check temporal ordering
    assert split.train["timestamp"].max() < split.val["timestamp"].min()
    assert split.val["timestamp"].max() < split.test["timestamp"].min()

    # Check index non-overlap
    assert split.verify_no_overlap() is True


def test_walk_forward_split(dummy_timeseries_df):
    folds = list(walk_forward_split(dummy_timeseries_df, initial_train_size=60, step_size=10, test_size=10))

    assert len(folds) == 4  # (60, 10), (70, 10), (80, 10), (90, 10)

    for train_slice, test_slice, fold_idx in folds:
        assert len(test_slice) == 10
        assert train_slice["timestamp"].max() < test_slice["timestamp"].min()
        # Verify expanding window
        assert len(train_slice) == 60 + fold_idx * 10
