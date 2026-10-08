"""Unit tests for feature engineering computations."""

import numpy as np
import pandas as pd
import pytest
from src.features.builder import THRESHOLD_COLUMN_MAP, FeatureBuilder
from src.features.statistical import (
    compute_count_and_distribution_features,
    compute_lag_features,
    compute_rolling_features,
    compute_streak_features,
)


@pytest.fixture
def sample_series():
    # Multipliers: 1.0, 1.2, 1.8, 2.5, 6.0, 1.1, 1.3
    return pd.DataFrame({
        "multiplier": [1.0, 1.2, 1.8, 2.5, 6.0, 1.1, 1.3],
        "round_id": range(1, 8),
    })


def test_lag_features(sample_series):
    lags = [1, 2]
    lag_df = compute_lag_features(sample_series, lags)

    assert pd.isna(lag_df.loc[0, "previous_1"])
    assert lag_df.loc[1, "previous_1"] == 1.0
    assert lag_df.loc[2, "previous_1"] == 1.2
    assert lag_df.loc[2, "previous_2"] == 1.0


def test_streak_features():
    # Sequence of multipliers:
    # 0: 1.2 (<1.5, <2)
    # 1: 1.3 (<1.5, <2)
    # 2: 1.4 (<1.5, <2)
    # 3: 5.5 (>=5, >=2)
    # 4: 6.0 (>=5, >=2)
    # 5: 1.1 (<1.5, <2)
    df = pd.DataFrame({"multiplier": [1.2, 1.3, 1.4, 5.5, 6.0, 1.1]})
    streaks = compute_streak_features(df)

    # At row 0 (before row 0), no history
    assert streaks.loc[0, "low_multiplier_streak"] == 0

    # At row 1, history is [1.2] -> streak 1
    assert streaks.loc[1, "low_multiplier_streak"] == 1

    # At row 2, history is [1.2, 1.3] -> streak 2
    assert streaks.loc[2, "low_multiplier_streak"] == 2

    # At row 3, history is [1.2, 1.3, 1.4] -> streak 3
    assert streaks.loc[3, "low_multiplier_streak"] == 3

    # At row 4, round 3 was 5.5 -> low streak broken (0), high streak is 1
    assert streaks.loc[4, "low_multiplier_streak"] == 0
    assert streaks.loc[4, "high_multiplier_streak"] == 1

    # At row 5, round 4 was 6.0 -> high streak is 2
    assert streaks.loc[5, "high_multiplier_streak"] == 2


def test_target_generation():
    df = pd.DataFrame({
        "round_id": range(100),
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="15s"),
        "multiplier": [1.0 + i * 0.1 for i in range(100)],
    })
    builder = FeatureBuilder(thresholds=[1.5, 2.0, 5.0])
    X, y_thresh, y_reg, _ = builder.build(df, drop_warmup=False)

    # Check threshold logic
    for th in [1.5, 2.0, 5.0]:
        col = THRESHOLD_COLUMN_MAP[th]
        expected = (df["multiplier"] >= th).astype(int)
        assert (y_thresh[col] == expected).all()
