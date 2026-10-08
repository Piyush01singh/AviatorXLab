"""Unit tests for evaluation metrics and comparison tables."""

import numpy as np
import pandas as pd
import pytest
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.classification import compute_classification_metrics
from src.evaluation.comparison import generate_classification_comparison_table
from src.evaluation.regression import compute_regression_metrics


def test_classification_metrics_standard():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_classification_metrics(y_true, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier_score"] < 0.05
    assert metrics["log_loss"] < 0.3


def test_classification_metrics_constant_prob():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])

    metrics = compute_classification_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == 0.5
    assert metrics["brier_score"] == 0.25


def test_regression_metrics():
    y_true = np.array([1.5, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 3.0])

    reg = compute_regression_metrics(y_true, y_pred)
    assert reg["mae"] == 0.0
    assert reg["rmse"] == 0.0
    assert reg["r2"] == 1.0
    assert reg["median_ae"] == 0.0


def test_calibration_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.2, 0.2, 0.8, 0.8, 0.9, 0.9])

    calib = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    assert "ece" in calib
    assert "mce" in calib
    assert calib["ece"] <= 0.20


def test_comparison_table():
    results = [
        {"model": "Historical Baseline", "threshold": "2x", "accuracy": 0.60, "brier_score": 0.24, "roc_auc": 0.50, "pr_auc": 0.60, "log_loss": 0.67},
        {"model": "Logistic Regression", "threshold": "2x", "accuracy": 0.62, "brier_score": 0.23, "roc_auc": 0.53, "pr_auc": 0.61, "log_loss": 0.66},
    ]
    table = generate_classification_comparison_table(results)

    assert "diff_acc_vs_baseline" in table.columns
    row_ml = table[table["model"] == "Logistic Regression"].iloc[0]
    assert round(row_ml["diff_acc_vs_baseline"], 2) == 0.02
