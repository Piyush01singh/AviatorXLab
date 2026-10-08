"""Regression evaluation metrics for exact multiplier predictions."""

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """Calculate regression metrics: MAE, RMSE, R2, and Median Absolute Error.
    
    Args:
        y_true: Observed ground truth multipliers.
        y_pred: Predicted continuous multipliers.
        
    Returns:
        Dict of regression metric names to values.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    med_ae = float(median_absolute_error(yt, yp))

    return {
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "r2": round(r2, 5),
        "median_ae": round(med_ae, 5),
    }
