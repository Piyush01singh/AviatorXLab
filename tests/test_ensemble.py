"""Unit tests for WeightedBlendEnsemble."""

import numpy as np
import pytest

from src.models.ensemble import WeightedBlendEnsemble


def test_ensemble_weight_optimization_and_prediction(tmp_path):
    np.random.seed(42)
    n = 300
    y_val = np.random.binomial(1, 0.5, size=n)

    # Model 1 is good, Model 2 is random noise, Model 3 is inverted
    p1 = np.clip(y_val * 0.8 + 0.1 + np.random.normal(0, 0.05, size=n), 0.01, 0.99)
    p2 = np.random.uniform(0.1, 0.9, size=n)
    p3 = np.clip((1 - y_val) * 0.7 + 0.15, 0.01, 0.99)

    ensemble = WeightedBlendEnsemble(model_names=["GoodModel", "NoiseModel", "BadModel"])
    ensemble.fit_weights(y_val, [p1, p2, p3], criterion="brier")

    assert ensemble.is_fitted
    assert len(ensemble.weights) == 3
    # Weights must sum to 1.0 (convex blend)
    assert np.isclose(np.sum(ensemble.weights), 1.0)
    # Weights must be non-negative
    assert np.all(ensemble.weights >= 0.0)
    # Good model should receive highest weight
    assert ensemble.weights[0] > ensemble.weights[1]
    assert ensemble.weights[0] > ensemble.weights[2]

    # Predict test
    p_blend = ensemble.predict_proba([p1[:10], p2[:10], p3[:10]])
    assert len(p_blend) == 10
    assert np.all(p_blend >= 0.0)
    assert np.all(p_blend <= 1.0)

    discrete_preds = ensemble.predict([p1[:10], p2[:10], p3[:10]], threshold=0.5)
    assert len(discrete_preds) == 10
    assert set(discrete_preds).issubset({0, 1})

    # Test serialization
    save_path = tmp_path / "test_ensemble.joblib"
    ensemble.save(str(save_path))
    loaded = WeightedBlendEnsemble.load(str(save_path))
    np.testing.assert_allclose(ensemble.weights, loaded.weights)
    p_loaded = loaded.predict_proba([p1[:10], p2[:10], p3[:10]])
    np.testing.assert_allclose(p_blend, p_loaded)
