"""Logistic Regression threshold classifier with strict training-only scaling."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.utils.logging import get_logger

logger = get_logger("logistic_model")


class LogisticThresholdClassifier:
    """Logistic Regression model for crash multiplier threshold probability prediction."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: Optional[str] = None,
        random_state: int = 42,
    ):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticThresholdClassifier":
        """Fit scaler and model strictly on training data.
        
        Args:
            X: Training feature matrix.
            y: Training binary target.
        """
        self.feature_names = list(X.columns)
        logger.info(f"Fitting Logistic Regression on {len(X)} samples with {len(self.feature_names)} features...")

        # Scale features - fit strictly on training set
        X_scaled = self.scaler.fit_transform(X.values)
        self.model.fit(X_scaled, y.values)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities [P(0), P(1)]."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        X_scaled = self.scaler.transform(X[self.feature_names].values)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels based on threshold probability."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

    def get_feature_importances(self) -> pd.DataFrame:
        """Return standardized coefficient weights."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        coefs = self.model.coef_[0]
        df = pd.DataFrame({"feature": self.feature_names, "coefficient": coefs})
        df["abs_coefficient"] = df["coefficient"].abs()
        return df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)

    def save(self, filepath: str) -> None:
        """Serialize model, scaler, and metadata to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": {
                "C": self.C,
                "penalty": self.penalty,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
            },
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, filepath: str) -> "LogisticThresholdClassifier":
        """Load serialized model and artifacts."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")
        payload = joblib.load(path)
        instance = cls(**payload["config"])
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.feature_names = payload["feature_names"]
        instance.is_fitted = True
        return instance
