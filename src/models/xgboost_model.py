"""XGBoost threshold classifier and regression models with early stopping."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from src.utils.logging import get_logger

logger = get_logger("xgboost_model")


class XGBoostThresholdClassifier:
    """XGBoost classifier for crash threshold probability prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 15,
        eval_metric: str = "logloss",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostThresholdClassifier":
        """Fit XGBoost model with early stopping on validation split."""
        self.feature_names = list(X_train.columns)
        eval_set = [(X_train.values, y_train.values)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val[self.feature_names].values, y_val.values))

        logger.info(
            f"Fitting XGBoost Classifier (trees={self.n_estimators}, depth={self.max_depth}) "
            f"on {len(X_train)} samples with early stopping ({self.early_stopping_rounds} rounds)..."
        )
        self.model.fit(
            X_train.values,
            y_train.values,
            eval_set=eval_set,
            verbose=False,
        )
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
        """Return XGBoost feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        importances = self.model.feature_importances_
        df = pd.DataFrame({"feature": self.feature_names, "importance": importances})
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    def save(self, filepath: str) -> None:
        """Serialize model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "config": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "early_stopping_rounds": self.early_stopping_rounds,
                "eval_metric": self.eval_metric,
                "random_state": self.random_state,
            },
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, filepath: str) -> "XGBoostThresholdClassifier":
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


class XGBoostMultiplierRegressor:
    """XGBoost regressor for secondary exact multiplier prediction experiment."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 15,
        eval_metric: str = "rmse",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state

        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostMultiplierRegressor":
        self.feature_names = list(X_train.columns)
        eval_set = [(X_train.values, y_train.values)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val[self.feature_names].values, y_val.values))

        self.model.fit(
            X_train.values,
            y_train.values,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        return self.model.predict(X[self.feature_names].values)

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "config": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "early_stopping_rounds": self.early_stopping_rounds,
                "eval_metric": self.eval_metric,
                "random_state": self.random_state,
            },
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, filepath: str) -> "XGBoostMultiplierRegressor":
        path = Path(filepath)
        payload = joblib.load(path)
        instance = cls(**payload["config"])
        instance.model = payload["model"]
        instance.feature_names = payload["feature_names"]
        instance.is_fitted = True
        return instance
