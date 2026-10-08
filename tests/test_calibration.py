"""Unit tests for probability calibration, Platt scaling, Isotonic regression, and ECE."""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from src.evaluation.calibration import (
    ProbabilityCalibrator,
    compute_calibration_metrics,
    evaluate_calibration_comparison,
)


def test_compute_calibration_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6, 0.85, 0.95])

    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    assert "ece" in metrics
    assert "mce" in metrics
    assert "prob_true" in metrics
    assert "prob_pred" in metrics
    assert "bin_counts" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["mce"] <= 1.0


def test_platt_calibrator_fit_and_inference(tmp_path):
    np.random.seed(42)
    n = 200
    y_val = np.random.binomial(1, 0.4, size=n)
    # Overconfident uncalibrated probabilities
    y_prob_val = np.clip(y_val * 0.8 + np.random.normal(0, 0.2, size=n), 0.01, 0.99)

    calibrator = ProbabilityCalibrator(method="platt")
    calibrator.fit(y_val, y_prob_val)
    assert calibrator.is_fitted

    test_probs = np.array([0.1, 0.5, 0.9])
    cal_probs = calibrator.calibrate(test_probs)
    assert len(cal_probs) == 3
    assert np.all(cal_probs >= 0.0)
    assert np.all(cal_probs <= 1.0)

    # Test save and load
    save_path = tmp_path / "platt_cal.joblib"
    calibrator.save(str(save_path))
    loaded = ProbabilityCalibrator.load(str(save_path))
    loaded_probs = loaded.calibrate(test_probs)
    np.testing.assert_allclose(cal_probs, loaded_probs)


def test_isotonic_calibrator_fit_and_inference(tmp_path):
    np.random.seed(42)
    n = 200
    y_val = np.random.binomial(1, 0.5, size=n)
    y_prob_val = np.clip(y_val * 0.7 + np.random.uniform(0, 0.3, size=n), 0.01, 0.99)

    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(y_val, y_prob_val)
    assert calibrator.is_fitted

    test_probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    cal_probs = calibrator.calibrate(test_probs)
    assert cal_probs.shape == (2, 2)
    np.testing.assert_allclose(cal_probs.sum(axis=1), np.ones(2), atol=1e-5)


def test_evaluate_calibration_comparison():
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    uncal = np.array([0.9, 0.2, 0.8, 0.1, 0.2, 0.9, 0.85, 0.15])  # badly miscalibrated
    cal = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9])      # well calibrated

    res = evaluate_calibration_comparison(y_true, uncal, cal)
    assert "uncalibrated" in res
    assert "calibrated" in res
    assert res["calibrated"]["brier_score"] < res["uncalibrated"]["brier_score"]
