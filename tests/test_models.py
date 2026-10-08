"""Unit tests for ML models (Logistic Regression, Random Forest, XGBoost)."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.models.logistic import LogisticThresholdClassifier
from src.models.random_forest import RandomForestThresholdClassifier
from src.models.xgboost_model import XGBoostMultiplierRegressor, XGBoostThresholdClassifier


@pytest.fixture
def toy_data():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
    })
    y_class = pd.Series((X["feat_1"] + X["feat_2"] > 0).astype(int))
    y_reg = pd.Series(np.maximum(1.0, 2.0 + X["feat_1"] + np.random.exponential(1.0, size=n)))
    return X, y_class, y_reg


def test_logistic_classifier(toy_data):
    X, y_class, _ = toy_data
    model = LogisticThresholdClassifier(max_iter=100)
    model.fit(X, y_class)

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    preds = model.predict(X)
    assert len(preds) == len(X)

    # Save & reload test
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "logistic.joblib"
        model.save(str(path))
        loaded = LogisticThresholdClassifier.load(str(path))
        loaded_probs = loaded.predict_proba(X)
        np.testing.assert_allclose(probs, loaded_probs)


def test_random_forest_classifier(toy_data):
    X, y_class, _ = toy_data
    model = RandomForestThresholdClassifier(n_estimators=10, max_depth=3)
    model.fit(X, y_class)

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "rf.joblib"
        model.save(str(path))
        loaded = RandomForestThresholdClassifier.load(str(path))
        np.testing.assert_allclose(probs, loaded.predict_proba(X))


def test_xgboost_classifier_and_regressor(toy_data):
    X, y_class, y_reg = toy_data

    # Classifier
    clf = XGBoostThresholdClassifier(n_estimators=10, max_depth=2, early_stopping_rounds=5)
    clf.fit(X, y_class, X_val=X, y_val=y_class)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 2)

    # Regressor
    reg = XGBoostMultiplierRegressor(n_estimators=10, max_depth=2, early_stopping_rounds=5)
    reg.fit(X, y_reg, X_val=X, y_val=y_reg)
    preds = reg.predict(X)
    assert len(preds) == len(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        path_c = Path(tmpdir) / "xgb_clf.joblib"
        clf.save(str(path_c))
        loaded_c = XGBoostThresholdClassifier.load(str(path_c))
        np.testing.assert_allclose(probs, loaded_c.predict_proba(X))

        path_r = Path(tmpdir) / "xgb_reg.joblib"
        reg.save(str(path_r))
        loaded_r = XGBoostMultiplierRegressor.load(str(path_r))
        np.testing.assert_allclose(preds, loaded_r.predict(X))
