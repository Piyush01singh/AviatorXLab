"""Comprehensive classification evaluation metrics for threshold predictions."""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate all required classification metrics for a predicted probability vector.
    
    Metrics:
      - Accuracy
      - Balanced Accuracy
      - Precision
      - Recall
      - F1 Score
      - ROC-AUC
      - PR-AUC (Average Precision)
      - Log Loss (Cross-Entropy)
      - Brier Score
      
    Args:
        y_true: Binary ground truth outcomes {0, 1}.
        y_prob: Predicted probabilities P(Y=1) in [0, 1].
        threshold: Decision threshold for discrete classification (default 0.5).
        
    Returns:
        Dict of metric names to calculated floats.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)

    # Handle shape if 2D probabilities [P(0), P(1)] or single column [[P(1)]]
    if y_p.ndim == 2:
        y_p = y_p[:, 1] if y_p.shape[1] > 1 else y_p[:, 0]

    # Clip probabilities to prevent log(0)
    eps = 1e-15
    y_p_clipped = np.clip(y_p, eps, 1.0 - eps)

    # Discrete predictions based on decision threshold
    y_pred = (y_p >= threshold).astype(int)

    acc = float(accuracy_score(y_t, y_pred))
    bal_acc = float(balanced_accuracy_score(y_t, y_pred))

    # Precision, recall, f1 with zero_division=0
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))

    # ROC-AUC: check for single class or constant predictions
    if len(np.unique(y_t)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        try:
            # If all probabilities are identical, ROC-AUC is mathematically 0.5
            if np.allclose(y_p, y_p[0]):
                roc_auc = 0.5
            else:
                roc_auc = float(roc_auc_score(y_t, y_p))
        except Exception:
            roc_auc = 0.5

        try:
            if np.allclose(y_p, y_p[0]):
                pr_auc = float(np.mean(y_t == 1))
            else:
                pr_auc = float(average_precision_score(y_t, y_p))
        except Exception:
            pr_auc = float(np.mean(y_t == 1))

    # Log Loss
    try:
        loss = float(log_loss(y_t, y_p_clipped))
    except Exception:
        loss = float("nan")

    # Brier Score
    brier = float(brier_score_loss(y_t, y_p))

    return {
        "accuracy": round(acc, 5),
        "balanced_accuracy": round(bal_acc, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5),
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "log_loss": round(loss, 5),
        "brier_score": round(brier, 5),
    }
