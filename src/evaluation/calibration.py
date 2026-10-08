"""Probability calibration assessment, Platt scaling, Isotonic regression, and ECE evaluation.

Strict Anti-Leakage Guarantee:
Calibrators are fitted strictly on validation predictions and evaluated on out-of-sample test predictions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.utils.logging import get_logger

logger = get_logger("calibration")


def compute_calibration_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """Assess calibration quality of predicted probabilities.
    
    Computes:
      - Expected Calibration Error (ECE)
      - Maximum Calibration Error (MCE)
      - Fraction of positives per bin
      - Mean predicted probability per bin
      - Sample count per bin
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_prob).astype(float)
    if yp.ndim == 2:
        yp = yp[:, 1] if yp.shape[1] > 1 else yp[:, 0]

    # Handle edge case where predictions are constant
    if np.allclose(yp, yp[0]):
        actual_mean = float(np.mean(yt))
        pred_val = float(yp[0])
        ece = float(abs(actual_mean - pred_val))
        return {
            "ece": round(ece, 5),
            "mce": round(ece, 5),
            "prob_true": [round(actual_mean, 5)],
            "prob_pred": [round(pred_val, 5)],
            "bin_counts": [len(yt)],
        }

    prob_true, prob_pred = calibration_curve(yt, yp, n_bins=n_bins, strategy=strategy)

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_assignments = np.digitize(yp, bins) - 1
        bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.quantile(yp, quantiles)
        bins[0] = 0.0
        bins[-1] = 1.0
        bin_assignments = np.digitize(yp, bins) - 1
        bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    bin_counts = []
    ece = 0.0
    mce = 0.0
    n_total = len(yt)

    for b in range(n_bins):
        mask = bin_assignments == b
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            bin_acc = float(np.mean(yt[mask]))
            bin_conf = float(np.mean(yp[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / n_total) * gap
            mce = max(mce, gap)

    return {
        "ece": round(float(ece), 5),
        "mce": round(float(mce), 5),
        "prob_true": [round(float(x), 5) for x in prob_true],
        "prob_pred": [round(float(x), 5) for x in prob_pred],
        "bin_counts": bin_counts,
    }


class ProbabilityCalibrator:
    """Post-hoc probability calibrator supporting Platt (Logistic) Scaling and Isotonic Regression."""

    def __init__(self, method: str = "platt"):
        self.method = method.lower()
        if self.method not in ["platt", "isotonic"]:
            raise ValueError(f"Unknown calibration method '{method}'. Choose 'platt' or 'isotonic'.")

        self.calibrator = None
        self.is_fitted = False

    def fit(self, y_val: np.ndarray, y_prob_val: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibration model strictly on validation partition.
        
        Args:
            y_val: Validation ground truth binary labels {0, 1}.
            y_prob_val: Raw validation model probabilities P(Y=1).
        """
        y_val_arr = np.asarray(y_val).astype(int)
        p_val_arr = np.asarray(y_prob_val).astype(float)
        if p_val_arr.ndim == 2:
            p_val_arr = p_val_arr[:, 1] if p_val_arr.shape[1] > 1 else p_val_arr[:, 0]

        # Clip bounds to avoid infinite logits
        eps = 1e-12
        p_clipped = np.clip(p_val_arr, eps, 1.0 - eps)

        if self.method == "platt":
            # Logistic regression on log-odds
            logits = logit(p_clipped).reshape(-1, 1)
            clf = LogisticRegression(C=1.0, solver="lbfgs")
            clf.fit(logits, y_val_arr)
            self.calibrator = clf
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p_clipped, y_val_arr)
            self.calibrator = iso

        self.is_fitted = True
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply fitted calibration to unseen probabilities."""
        if not self.is_fitted or self.calibrator is None:
            raise RuntimeError("Calibrator has not been fitted.")

        p_arr = np.asarray(y_prob).astype(float)
        is_2d = p_arr.ndim == 2 and p_arr.shape[1] > 1
        p_eval = p_arr[:, 1] if is_2d else p_arr.ravel()

        eps = 1e-12
        p_clipped = np.clip(p_eval, eps, 1.0 - eps)

        if self.method == "platt":
            logits = logit(p_clipped).reshape(-1, 1)
            cal_probs = self.calibrator.predict_proba(logits)[:, 1]
        elif self.method == "isotonic":
            cal_probs = self.calibrator.predict(p_clipped)

        cal_probs = np.clip(cal_probs, eps, 1.0 - eps)

        if is_2d:
            res = np.zeros_like(p_arr)
            res[:, 0] = 1.0 - cal_probs
            res[:, 1] = cal_probs
            return res
        return cal_probs

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method": self.method, "calibrator": self.calibrator}, path)

    @classmethod
    def load(cls, filepath: str) -> "ProbabilityCalibrator":
        payload = joblib.load(filepath)
        instance = cls(method=payload["method"])
        instance.calibrator = payload["calibrator"]
        instance.is_fitted = True
        return instance


def evaluate_calibration_comparison(
    y_true_test: np.ndarray,
    y_prob_uncal_test: np.ndarray,
    y_prob_cal_test: np.ndarray,
) -> Dict[str, Any]:
    """Compare uncalibrated vs calibrated probabilities on out-of-sample test data.
    
    Returns metrics before and after: Brier Score, Log Loss, and ECE.
    """
    yt = np.asarray(y_true_test).astype(int)
    p_uncal = np.asarray(y_prob_uncal_test).astype(float)
    p_cal = np.asarray(y_prob_cal_test).astype(float)

    if p_uncal.ndim == 2:
        p_uncal = p_uncal[:, 1] if p_uncal.shape[1] > 1 else p_uncal[:, 0]
    if p_cal.ndim == 2:
        p_cal = p_cal[:, 1] if p_cal.shape[1] > 1 else p_cal[:, 0]

    eps = 1e-15
    p_uncal_c = np.clip(p_uncal, eps, 1.0 - eps)
    p_cal_c = np.clip(p_cal, eps, 1.0 - eps)

    uncal_brier = float(brier_score_loss(yt, p_uncal_c))
    cal_brier = float(brier_score_loss(yt, p_cal_c))

    uncal_loss = float(log_loss(yt, p_uncal_c))
    cal_loss = float(log_loss(yt, p_cal_c))

    uncal_calib = compute_calibration_metrics(yt, p_uncal_c)
    cal_calib = compute_calibration_metrics(yt, p_cal_c)

    return {
        "uncalibrated": {
            "brier_score": round(uncal_brier, 5),
            "log_loss": round(uncal_loss, 5),
            "ece": uncal_calib["ece"],
            "prob_true": uncal_calib["prob_true"],
            "prob_pred": uncal_calib["prob_pred"],
        },
        "calibrated": {
            "brier_score": round(cal_brier, 5),
            "log_loss": round(cal_loss, 5),
            "ece": cal_calib["ece"],
            "prob_true": cal_calib["prob_true"],
            "prob_pred": cal_calib["prob_pred"],
        },
        "diff_brier": round(cal_brier - uncal_brier, 5),
        "diff_ece": round(cal_calib["ece"] - uncal_calib["ece"], 5),
    }
