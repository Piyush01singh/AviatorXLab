"""Ensemble blending model combining top tabular predictors (XGBoost, MLP, Logistic Regression).

Strict Anti-Leakage Guarantee:
Ensemble blending weights are optimized strictly on validation set predictions,
then evaluated out-of-sample on untouched test folds.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss

from src.utils.logging import get_logger

logger = get_logger("ensemble")


class WeightedBlendEnsemble:
    """Optimal convex weighted blend of predicted probabilities."""

    def __init__(self, model_names: Optional[List[str]] = None):
        self.model_names = model_names or ["XGBoost", "MLP", "Logistic Regression"]
        self.weights: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit_weights(
        self,
        y_val: np.ndarray,
        pred_matrices_val: List[np.ndarray],
        criterion: str = "brier",
    ) -> "WeightedBlendEnsemble":
        """Optimize blend weights strictly on validation data.
        
        Args:
            y_val: Validation ground truth binary targets {0, 1}.
            pred_matrices_val: List of probability arrays [P_model1, P_model2, ...], each of shape (N,).
            criterion: 'brier' or 'logloss'
        """
        y_val_arr = np.asarray(y_val).astype(int)
        M = len(pred_matrices_val)
        P_stack = np.column_stack([np.asarray(p).ravel() for p in pred_matrices_val])

        # Objective function
        def loss_fn(weights: np.ndarray) -> float:
            # Softmax to guarantee non-negative weights summing to 1.0
            exp_w = np.exp(weights - np.max(weights))
            norm_w = exp_w / np.sum(exp_w)
            p_blend = np.dot(P_stack, norm_w)
            p_blend = np.clip(p_blend, 1e-12, 1.0 - 1e-12)
            if criterion == "logloss":
                return float(log_loss(y_val_arr, p_blend))
            return float(brier_score_loss(y_val_arr, p_blend))

        # Initial equal weights
        init_w = np.zeros(M)
        res = minimize(loss_fn, init_w, method="BFGS")

        exp_w = np.exp(res.x - np.max(res.x))
        self.weights = exp_w / np.sum(exp_w)
        self.is_fitted = True

        weight_str = ", ".join(f"{name}: {w:.3f}" for name, w in zip(self.model_names, self.weights))
        logger.info(f"Optimal ensemble weights ({criterion}): {weight_str}")
        return self

    def predict_proba(self, pred_matrices: List[np.ndarray]) -> np.ndarray:
        """Compute convex weighted probability blend.
        
        Args:
            pred_matrices: List of probability arrays from component models.
            
        Returns:
            Blended probabilities array of shape (N,).
        """
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Ensemble weights have not been fitted.")

        P_stack = np.column_stack([np.asarray(p).ravel() for p in pred_matrices])
        return np.dot(P_stack, self.weights)

    def predict(self, pred_matrices: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(pred_matrices)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_names": self.model_names, "weights": self.weights}, path)

    @classmethod
    def load(cls, filepath: str) -> "WeightedBlendEnsemble":
        payload = joblib.load(filepath)
        instance = cls(model_names=payload["model_names"])
        instance.weights = payload["weights"]
        instance.is_fitted = True
        return instance
