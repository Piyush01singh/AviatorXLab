"""Statistical and naive baseline models.

Implements scikit-learn compatible interfaces for all required baseline predictors.
"""

from typing import Any, Optional, Union
import numpy as np
import pandas as pd


class MajorityClassBaseline:
    """Predicts the most frequent class observed in training data."""

    def __init__(self):
        self.majority_class: Optional[int] = None
        self.class_prob: Optional[float] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "MajorityClassBaseline":
        y_arr = np.asarray(y)
        counts = np.bincount(y_arr.astype(int))
        self.majority_class = int(np.argmax(counts))
        self.class_prob = float(np.mean(y_arr == self.majority_class))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        n = len(X)
        return np.full(n, self.majority_class, dtype=int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns deterministic probability vector matching the majority class."""
        n = len(X)
        probs = np.zeros((n, 2), dtype=float)
        probs[:, self.majority_class] = 1.0
        probs[:, 1 - self.majority_class] = 0.0
        return probs


class HistoricalProbabilityBaseline:
    """Predicts the unconditional empirical training class probability P(Y=1)."""

    def __init__(self):
        self.p_positive: float = 0.5

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "HistoricalProbabilityBaseline":
        y_arr = np.asarray(y)
        self.p_positive = float(np.mean(y_arr == 1))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        n = len(X)
        pred_label = 1 if self.p_positive >= threshold else 0
        return np.full(n, pred_label, dtype=int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns constant empirical probability [1 - p, p] for all rows."""
        n = len(X)
        probs = np.zeros((n, 2), dtype=float)
        probs[:, 0] = 1.0 - self.p_positive
        probs[:, 1] = self.p_positive
        return probs


class RollingHistoricalProbabilityBaseline:
    """Predicts the rolling empirical frequency of the event over the past W rounds.
    
    Strictly causal: uses outcomes strictly prior to round t.
    """

    def __init__(self, window: int = 50):
        self.window = window
        self.fallback_p: float = 0.5

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "RollingHistoricalProbabilityBaseline":
        y_arr = np.asarray(y)
        self.fallback_p = float(np.mean(y_arr == 1))
        return self

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        past_target_series: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """Predict probabilities based on rolling historical outcome frequencies.
        
        If past_target_series is provided, computes rolling mean of past rounds.
        Otherwise falls back to constant training probability.
        """
        n = len(X)
        probs = np.zeros((n, 2), dtype=float)

        if past_target_series is not None and len(past_target_series) == n:
            # Shifted rolling mean of the target series
            roll_p = past_target_series.shift(1).rolling(window=self.window, min_periods=1).mean()
            p1 = roll_p.fillna(self.fallback_p).to_numpy()
        else:
            p1 = np.full(n, self.fallback_p)

        p1 = np.clip(p1, 0.001, 0.999)
        probs[:, 0] = 1.0 - p1
        probs[:, 1] = p1
        return probs

    def predict(self, X: Union[pd.DataFrame, np.ndarray], past_target_series: Optional[pd.Series] = None) -> np.ndarray:
        probs = self.predict_proba(X, past_target_series)
        return (probs[:, 1] >= 0.5).astype(int)


class MedianRegressionBaseline:
    """Predicts the unconditional median multiplier observed in training data."""

    def __init__(self):
        self.median_val: float = 1.0

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "MedianRegressionBaseline":
        y_arr = np.asarray(y)
        self.median_val = float(np.median(y_arr))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        n = len(X)
        return np.full(n, self.median_val, dtype=float)


class RollingMedianRegressionBaseline:
    """Predicts the rolling median multiplier of previous W rounds prior to round t."""

    def __init__(self, window: int = 50):
        self.window = window
        self.fallback_median: float = 1.5

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "RollingMedianRegressionBaseline":
        self.fallback_median = float(np.median(np.asarray(y)))
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        past_multiplier_series: Optional[pd.Series] = None,
    ) -> np.ndarray:
        n = len(X)
        if past_multiplier_series is not None and len(past_multiplier_series) == n:
            roll_m = past_multiplier_series.shift(1).rolling(window=self.window, min_periods=1).median()
            return roll_m.fillna(self.fallback_median).to_numpy()
        return np.full(n, self.fallback_median, dtype=float)


class RandomUniformBenchmark:
    """Outputs random uniform probabilities in [0, 1]."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X: Any, y: Any) -> "RandomUniformBenchmark":
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        np.random.seed(self.seed)
        n = len(X)
        p1 = np.random.uniform(0.0, 1.0, size=n)
        probs = np.zeros((n, 2), dtype=float)
        probs[:, 0] = 1.0 - p1
        probs[:, 1] = p1
        return probs

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
