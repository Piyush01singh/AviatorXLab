"""Neural sequence leakage audit test.

Verifies programmatically that predicting target y(t) uses strictly [x(t-N), ..., x(t-1)],
and that perturbing round t or later has ZERO effect on the input sequence used to predict round t.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.builder import FeatureBuilder
from src.features.sequential import create_sequence_windows


def test_neural_sequence_zero_leakage():
    """Verify mathematically that modifying round t does NOT affect the input sequence for target t."""
    n = 80
    seq_len = 10
    np.random.seed(42)

    multipliers = np.random.uniform(1.05, 5.0, size=n)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="15s")

    df_orig = pd.DataFrame({
        "round_id": range(1000, 1000 + n),
        "timestamp": timestamps,
        "multiplier": multipliers,
    })

    builder = FeatureBuilder(
        lags=[1, 2, 3],
        rolling_windows=[5],
        thresholds=[1.5, 2.0],
        include_temporal=True,
    )

    X_orig, y_orig, _, _ = builder.build(df_orig, drop_warmup=False)
    # Fill any warmup NaNs consistently to test sequence slices
    X_orig_filled = X_orig.fillna(0.0).values
    y_orig_arr = y_orig.values

    X_seq_orig, y_seq_orig = create_sequence_windows(X_orig_filled, y_orig_arr, sequence_length=seq_len)

    # In create_sequence_windows, window k corresponds to target at index t = k + seq_len
    # Select target round t_star to perturb
    t_star = 45
    window_idx_for_t_star = t_star - seq_len

    # Perturb round t_star drastically
    df_pert = df_orig.copy()
    df_pert.loc[t_star, "multiplier"] = 99999.0

    X_pert, y_pert, _, _ = builder.build(df_pert, drop_warmup=False)
    X_pert_filled = X_pert.fillna(0.0).values
    y_pert_arr = y_pert.values

    X_seq_pert, y_seq_pert = create_sequence_windows(X_pert_filled, y_pert_arr, sequence_length=seq_len)

    # 1. The input sequence used to predict round t_star MUST BE COMPLETELY UNCHANGED
    np.testing.assert_array_equal(
        X_seq_orig[window_idx_for_t_star],
        X_seq_pert[window_idx_for_t_star],
        err_msg=f"LEAKAGE DETECTED! Input sequence for target {t_star} changed when round {t_star} was perturbed!",
    )

    # 2. All input sequences for rounds BEFORE t_star MUST ALSO BE COMPLETELY UNCHANGED
    for k in range(window_idx_for_t_star):
        np.testing.assert_array_equal(
            X_seq_orig[k],
            X_seq_pert[k],
            err_msg=f"LEAKAGE DETECTED! Input sequence at index {k} (before {t_star}) was altered!",
        )

    # 3. Window k = window_idx_for_t_star + 2 includes row t_star + 1 where previous_1 = multiplier[t_star]
    assert not np.array_equal(
        X_seq_orig[window_idx_for_t_star + 2],
        X_seq_pert[window_idx_for_t_star + 2],
    )
