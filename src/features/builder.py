"""Feature matrix and target builder for crash game modeling."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from src.features.statistical import (
    compute_count_and_distribution_features,
    compute_lag_features,
    compute_rolling_features,
    compute_streak_features,
    compute_temporal_features,
)
from src.utils.logging import get_logger

logger = get_logger("feature_builder")

DEFAULT_THRESHOLDS = [1.5, 2.0, 3.0, 5.0, 10.0]
THRESHOLD_COLUMN_MAP = {
    1.5: "target_1_5x",
    2.0: "target_2x",
    3.0: "target_3x",
    5.0: "target_5x",
    10.0: "target_10x",
}


class FeatureBuilder:
    """Builds tabular features and threshold targets with strict anti-leakage guards."""

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        thresholds: Optional[List[float]] = None,
        include_temporal: bool = True,
    ):
        self.lags = lags or [1, 2, 3, 4, 5, 10, 20, 50]
        self.rolling_windows = rolling_windows or [5, 10, 20, 50]
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.include_temporal = include_temporal

    def build(self, df: pd.DataFrame, drop_warmup: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
        """Construct features and target variables from cleaned game records.
        
        Args:
            df: Cleaned DataFrame with ['round_id', 'timestamp', 'multiplier'].
            drop_warmup: If True, drop initial rows that contain NaN due to lag/rolling window size.
            
        Returns:
            Tuple of:
              - X: Feature DataFrame
              - y_thresh: DataFrame of binary threshold targets (target_1_5x, etc.)
              - y_reg: Series of exact multiplier outcomes
              - feature_names: List of feature column names
        """
        logger.info(f"Extracting features from {len(df)} rounds...")

        # Feature groups
        lag_feats = compute_lag_features(df, self.lags)
        roll_feats = compute_rolling_features(df, self.rolling_windows)
        streak_feats = compute_streak_features(df)
        dist_feats = compute_count_and_distribution_features(df)

        feature_dfs = [lag_feats, roll_feats, streak_feats, dist_feats]

        if self.include_temporal and "timestamp" in df.columns:
            temp_feats = compute_temporal_features(df)
            feature_dfs.append(temp_feats)

        X = pd.concat(feature_dfs, axis=1)

        # Build targets
        y_thresh = pd.DataFrame(index=df.index)
        for th in self.thresholds:
            col_name = THRESHOLD_COLUMN_MAP.get(th, f"target_{str(th).replace('.', '_')}x")
            y_thresh[col_name] = (df["multiplier"] >= th).astype(int)

        y_reg = df["multiplier"].copy()

        # Preserve round_id and timestamp for tracking/indexing
        meta_cols = pd.DataFrame(index=df.index)
        if "round_id" in df.columns:
            meta_cols["round_id"] = df["round_id"]
        if "timestamp" in df.columns:
            meta_cols["timestamp"] = df["timestamp"]

        combined = pd.concat([meta_cols, X, y_thresh, y_reg.rename("multiplier")], axis=1)

        if drop_warmup:
            # Maximum window length is max(lags, rolling_windows)
            max_window = max(max(self.lags), max(self.rolling_windows))
            initial_len = len(combined)
            combined = combined.dropna().reset_index(drop=True)
            logger.info(
                f"Dropped {initial_len - len(combined)} warmup rows (window size={max_window}). "
                f"Remaining clean samples: {len(combined)}"
            )

        # Split back
        feature_cols = [c for c in X.columns]
        target_cols = [c for c in y_thresh.columns]

        X_final = combined[feature_cols].copy()
        y_thresh_final = combined[target_cols].copy()
        y_reg_final = combined["multiplier"].copy()

        # Attach meta tracking info as attributes or separate DataFrame
        if "round_id" in combined.columns:
            X_final.attrs["round_id"] = combined["round_id"]
        if "timestamp" in combined.columns:
            X_final.attrs["timestamp"] = combined["timestamp"]

        return X_final, y_thresh_final, y_reg_final, feature_cols
