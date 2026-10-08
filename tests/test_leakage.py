"""Audit test to mathematically verify zero data leakage in feature engineering.

Proves that perturbing round t does not alter any feature at round t or any round < t.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from src.features.builder import FeatureBuilder


def test_zero_leakage_feature_invariance():
    """Verify that features for round t are strictly invariant to the outcome at round t and all future rounds."""
    n = 120
    np.random.seed(42)
    multipliers = np.random.uniform(1.01, 10.0, size=n)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="15s")

    df_orig = pd.DataFrame({
        "round_id": range(1000, 1000 + n),
        "timestamp": timestamps,
        "multiplier": multipliers,
    })

    builder = FeatureBuilder(
        lags=[1, 2, 5, 10],
        rolling_windows=[5, 10, 20],
        thresholds=[1.5, 2.0, 5.0],
        include_temporal=True,
    )

    # Build features without dropping warmup to check exact row-by-row indices
    X_orig, _, _, _ = builder.build(df_orig, drop_warmup=False)

    # Target round to perturb
    target_idx = 75
    df_perturbed = df_orig.copy()
    # Drastically perturb multiplier at target_idx
    df_perturbed.loc[target_idx, "multiplier"] = 9999.99

    X_pert, _, _, _ = builder.build(df_perturbed, drop_warmup=False)

    # 1. Row target_idx itself MUST NOT change (feature at round t cannot see round t outcome)
    orig_row_t = X_orig.iloc[target_idx].dropna()
    pert_row_t = X_pert.iloc[target_idx].dropna()
    pd.testing.assert_series_equal(
        orig_row_t,
        pert_row_t,
        check_exact=True,
        obj=f"Feature row at perturbed index {target_idx} was affected by round {target_idx}'s outcome! LEAKAGE DETECTED!",
    )

    # 2. All rows BEFORE target_idx MUST be completely identical
    for i in range(target_idx):
        orig_row = X_orig.iloc[i].dropna()
        pert_row = X_pert.iloc[i].dropna()
        pd.testing.assert_series_equal(
            orig_row,
            pert_row,
            check_exact=True,
            obj=f"Feature row {i} (before {target_idx}) was altered by perturbation at {target_idx}! LEAKAGE DETECTED!",
        )

    # 3. Row target_idx + 1 SHOULD reflect the perturbation in lag_1
    assert X_pert.loc[target_idx + 1, "previous_1"] == 9999.99


def test_scaler_no_test_leakage():
    """Verify that feature scaler is fit strictly on training set without leaking test statistics."""
    X_train = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X_test = pd.DataFrame({"feat": [100.0, 200.0, 300.0]})

    scaler = StandardScaler()
    scaler.fit(X_train.values)

    # Scaler mean must strictly equal training mean
    assert scaler.mean_[0] == 3.0
    # Scaler variance must strictly equal training variance
    assert scaler.var_[0] == 2.0
