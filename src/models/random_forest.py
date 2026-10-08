"""Random Forest threshold classifier with probability outputs."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.utils.logging import get_logger

logger = get_logger("random_forest")


class RandomForestThresholdClassifier:
    """Random Forest classifier for crash multiplier threshold probability prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 6,
        min_samples_leaf: int = 20,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestThresholdClassifier":
        """Fit Random Forest on training features."""
        self.feature_names = list(X.columns)
        logger.info(
            f"Fitting Random Forest (trees={self.n_estimators}, max_depth={self.max_depth}) "
            f"on {len(X)} samples with {len(self.feature_names)} features..."
        )
        self.model.fit(X.values, y.values)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities [P(0), P(1)]."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        return self.model.predict_proba(X[self.feature_names].values)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels based on threshold probability."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

    def get_feature_importances(self) -> pd.DataFrame:
        """Return Gini feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        importances = self.model.feature_importances_
        df = pd.DataFrame({"feature": self.feature_names, "importance": importances})
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    def save(self, filepath: str) -> None:
        """Serialize model and metadata to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "config": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "n_jobs": self.n_jobs,
                "random_state": self.random_state,
            },
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, filepath: str) -> "RandomForestThresholdClassifier":
        """Load serialized model."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")
        payload = joblib.load(path)
        instance = cls(**payload["config"])
        instance.model = payload["model"]
        instance.feature_names = payload["feature_names"]
        instance.is_fitted = True
        return instance
