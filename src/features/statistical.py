"""Statistical and causal feature extraction for crash game multipliers.

Strict Anti-Leakage Guarantee:
All features for round t are derived strictly from rounds < t.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Compute lag features: previous_k = multiplier at t-k.
    
    Args:
        df: Input DataFrame with 'multiplier' column.
        lags: List of lag steps, e.g. [1, 2, 3, 5, 10, 20, 50].
        
    Returns:
        DataFrame with lag columns.
    """
    features = pd.DataFrame(index=df.index)
    for lag in lags:
        # shift(k) takes the value from k rows before
        features[f"previous_{lag}"] = df["multiplier"].shift(lag)
    return features


def compute_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Compute rolling statistical features on past rounds strictly prior to t.
    
    Uses df['multiplier'].shift(1) so that the current round's outcome is never included.
    
    Args:
        df: Input DataFrame with 'multiplier' column.
        windows: Rolling window sizes, e.g. [5, 10, 20, 50].
        
    Returns:
        DataFrame with rolling mean, median, std, min, max, and percentiles.
    """
    past = df["multiplier"].shift(1)
    features = pd.DataFrame(index=df.index)

    for w in windows:
        roll = past.rolling(window=w, min_periods=w)
        features[f"mean_{w}"] = roll.mean()
        features[f"median_{w}"] = roll.median()
        features[f"std_{w}"] = roll.std()
        features[f"min_{w}"] = roll.min()
        features[f"max_{w}"] = roll.max()
        features[f"quantile_25_{w}"] = roll.quantile(0.25)
        features[f"quantile_75_{w}"] = roll.quantile(0.75)

    return features


def compute_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute consecutive streaks of outcomes strictly prior to round t.
    
    Streaks:
      - low_multiplier_streak: consecutive rounds < 1.5
      - high_multiplier_streak: consecutive rounds >= 5.0
      - below_2x_streak: consecutive rounds < 2.0
      - above_5x_streak: consecutive rounds >= 5.0
    """
    past = df["multiplier"].shift(1).to_numpy()
    n = len(past)

    low_streak = np.zeros(n, dtype=np.int32)
    high_streak = np.zeros(n, dtype=np.int32)
    below_2x_streak = np.zeros(n, dtype=np.int32)
    above_5x_streak = np.zeros(n, dtype=np.int32)

    c_low = c_high = c_below_2 = c_above_5 = 0

    for i in range(n):
        val = past[i]
        if np.isnan(val):
            c_low = c_high = c_below_2 = c_above_5 = 0
        else:
            c_low = (c_low + 1) if val < 1.5 else 0
            c_high = (c_high + 1) if val >= 5.0 else 0
            c_below_2 = (c_below_2 + 1) if val < 2.0 else 0
            c_above_5 = (c_above_5 + 1) if val >= 5.0 else 0

        low_streak[i] = c_low
        high_streak[i] = c_high
        below_2x_streak[i] = c_below_2
        above_5x_streak[i] = c_above_5

    features = pd.DataFrame(
        {
            "low_multiplier_streak": low_streak,
            "high_multiplier_streak": high_streak,
            "below_2x_streak": below_2x_streak,
            "above_5x_streak": above_5x_streak,
        },
        index=df.index,
    )
    return features


def compute_count_and_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute count and frequency distribution features strictly on past rounds."""
    past = df["multiplier"].shift(1)
    features = pd.DataFrame(index=df.index)

    # Specific count features required by Section 7
    features["number_below_1_2_last_10"] = (past < 1.2).rolling(window=10, min_periods=10).sum()
    features["number_below_1_5_last_20"] = (past < 1.5).rolling(window=20, min_periods=20).sum()
    features["number_above_2_last_20"] = (past >= 2.0).rolling(window=20, min_periods=20).sum()
    features["number_above_5_last_50"] = (past >= 5.0).rolling(window=50, min_periods=50).sum()

    # Distribution fractions over rolling window of 50
    roll_50 = past.rolling(window=50, min_periods=50)
    features["fraction_below_1_5"] = (past < 1.5).rolling(window=50, min_periods=50).mean()
    features["fraction_above_2"] = (past >= 2.0).rolling(window=50, min_periods=50).mean()
    features["fraction_above_5"] = (past >= 5.0).rolling(window=50, min_periods=50).mean()

    return features


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal features available strictly prior to round t."""
    features = pd.DataFrame(index=df.index)

    if "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"], format="ISO8601")
        except Exception:
            ts = pd.to_datetime(df["timestamp"])
        if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        # Timestamp at round start is prior to round t; use prior completed round's timestamp
        past_ts = ts.shift(1)
        features["hour"] = past_ts.dt.hour
        features["minute"] = past_ts.dt.minute
        features["day_of_week"] = past_ts.dt.dayofweek

        # Duration of prior completed round (t-1 vs t-2) to prevent round t flight duration leakage
        time_diff = ts.diff().dt.total_seconds().shift(1)
        # Cap outliers to reasonable bounds (e.g. 300 seconds)
        features["time_since_previous_round"] = time_diff.fillna(15.0).clip(lower=0.0, upper=300.0)

    return features
