"""Feature-level timing and leakage audit for crash game prediction.

Strict Anti-Leakage Protocol:
- Documents exact availability timing for every feature.
- Distinguishes information available:
    1. before_waiting: Historical lags, shifted rolling statistics, shifted temporal values.
    2. during_waiting: Round ID and start timestamp.
    3. during_flying: In-flight ticks (PROHIBITED for pre-round prediction).
    4. after_crash: Target multiplier, flight duration, post-round pause, revealed seeds (PROHIBITED).
- Provides runtime verification that pre-round features strictly precede the prediction cutoff.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("timing_audit")

# Explicit availability stage definition for every feature family
FEATURE_AVAILABILITY_STAGES = {
    # Lag features: shifted by >= 1 round, available before round N starts
    "lag": "before_waiting",
    "previous_": "before_waiting",

    # Rolling statistics: computed strictly over rounds < N via shift(1)
    "mean_": "before_waiting",
    "median_": "before_waiting",
    "std_": "before_waiting",
    "min_": "before_waiting",
    "max_": "before_waiting",
    "quantile_": "before_waiting",

    # Streaks: computed strictly over rounds < N via shift(1)
    "low_multiplier_streak": "before_waiting",
    "high_multiplier_streak": "before_waiting",
    "below_2x_streak": "before_waiting",
    "above_5x_streak": "before_waiting",

    # Count & Distribution features: strictly shift(1)
    "number_below_1_2_last_10": "before_waiting",
    "number_below_1_5_last_20": "before_waiting",
    "number_above_2_last_20": "before_waiting",
    "number_above_5_last_50": "before_waiting",
    "fraction_below_1_5": "before_waiting",
    "fraction_above_2": "before_waiting",
    "fraction_above_5": "before_waiting",

    # Temporal features: based on prior completed round's timestamp (shift(1))
    "hour": "before_waiting",
    "minute": "before_waiting",
    "day_of_week": "before_waiting",
    # Pre-round interval: difference between t-1 and t-2; strictly before waiting
    "time_since_previous_round": "before_waiting",

    # Round identifiers available during waiting
    "round_id": "during_waiting",
    "waiting_timestamp": "during_waiting",

    # FORBIDDEN FOR PRE-ROUND FORECASTS:
    "multiplier_tick": "during_flying",
    "x_tick": "during_flying",
    "in_flight": "during_flying",
    "multiplier": "after_crash",
    "target_multiplier": "after_crash",
    "actual_multiplier": "after_crash",
    "crash_multiplier": "after_crash",
    "flight_duration": "after_crash",
    "post_round_interval": "after_crash",
    "revealed_server_seed": "after_crash",
    "server_seed": "after_crash",
}


def classify_feature_availability(feature_name: str) -> str:
    """Classify the availability stage of a given feature name."""
    clean_name = feature_name.lower().strip()

    # Exact match check
    if clean_name in FEATURE_AVAILABILITY_STAGES:
        return FEATURE_AVAILABILITY_STAGES[clean_name]

    # Prefix match checks
    for prefix, stage in FEATURE_AVAILABILITY_STAGES.items():
        if clean_name.startswith(prefix):
            return stage

    # Unknown features default to requiring verification
    return "unverified"


def audit_feature_set(feature_names: List[str]) -> Dict[str, Any]:
    """Audit a list of feature names against pre-round eligibility.

    A feature is eligible for pre-round prediction ONLY if its availability stage
    is 'before_waiting' or 'during_waiting'.
    """
    eligible = []
    forbidden = []
    unverified = []

    for feat in feature_names:
        stage = classify_feature_availability(feat)
        if stage in ("before_waiting", "during_waiting"):
            eligible.append((feat, stage))
        elif stage in ("during_flying", "after_crash"):
            forbidden.append((feat, stage))
        else:
            unverified.append((feat, stage))

    is_leakage_free = len(forbidden) == 0

    return {
        "is_leakage_free": is_leakage_free,
        "total_features": len(feature_names),
        "eligible_count": len(eligible),
        "forbidden_count": len(forbidden),
        "unverified_count": len(unverified),
        "eligible_features": eligible,
        "forbidden_features": forbidden,
        "unverified_features": unverified,
    }


def verify_feature_causality_invariance(
    df: pd.DataFrame,
    builder_fn: Any,
    target_idx: int = -1,
) -> Tuple[bool, float]:
    """Automated leakage invariance test.

    Proves that modifying the target round's future multiplier value produces
    EXACTLY zero difference in the generated feature vector for that round.
    """
    df_copy1 = df.copy()
    df_copy2 = df.copy()

    # Modify the target row's multiplier to two radically different values
    row_pos = target_idx if target_idx >= 0 else len(df) + target_idx
    df_copy1.iloc[row_pos, df_copy1.columns.get_loc("multiplier")] = 1.05
    df_copy2.iloc[row_pos, df_copy2.columns.get_loc("multiplier")] = 999.99

    X1, _, _, _ = builder_fn(df_copy1, drop_warmup=False)
    X2, _, _, _ = builder_fn(df_copy2, drop_warmup=False)

    vec1 = X1.iloc[row_pos].to_numpy(dtype=float)
    vec2 = X2.iloc[row_pos].to_numpy(dtype=float)

    # Calculate maximum absolute difference
    max_diff = float(np.nanmax(np.abs(vec1 - vec2)))
    is_invariant = max_diff < 1e-9

    return is_invariant, max_diff


def audit_prediction_timestamp_cutoff(
    round_waiting_ts: str,
    prediction_ts: str,
    cutoff_tolerance_seconds: float = 0.5,
) -> Dict[str, Any]:
    """Verify that a prediction was strictly generated prior to or at the round-start cutoff.

    Forecasts generated after the cutoff are rejected or flagged as non-prospective.
    """
    try:
        t_waiting = pd.to_datetime(round_waiting_ts, format="ISO8601", utc=True)
    except Exception:
        t_waiting = pd.to_datetime(round_waiting_ts, utc=True)

    try:
        t_pred = pd.to_datetime(prediction_ts, format="ISO8601", utc=True)
    except Exception:
        t_pred = pd.to_datetime(prediction_ts, utc=True)

    diff_seconds = (t_pred - t_waiting).total_seconds()
    is_valid_prospective = diff_seconds <= cutoff_tolerance_seconds

    return {
        "is_valid_prospective": is_valid_prospective,
        "round_waiting_time": t_waiting.isoformat(),
        "prediction_time": t_pred.isoformat(),
        "delta_seconds": round(diff_seconds, 4),
        "cutoff_tolerance_seconds": cutoff_tolerance_seconds,
        "status": "VALID_PROSPECTIVE" if is_valid_prospective else "FLAGGED_POST_CUTOFF",
    }
