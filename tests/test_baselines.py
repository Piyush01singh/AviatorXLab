"""Unit tests for baseline predictors."""

import numpy as np
import pandas as pd
import pytest
from src.models.baseline import (
    HistoricalProbabilityBaseline,
    MajorityClassBaseline,
    MedianRegressionBaseline,
    RandomUniformBenchmark,
    RollingHistoricalProbabilityBaseline,
)


def test_majority_class_baseline():
    X_tr = pd.DataFrame({"f": [1, 2, 3, 4, 5]})
    y_tr = pd.Series([1, 1, 1, 0, 0])  # majority is 1 (60%)
    X_val = pd.DataFrame({"f": [10, 20]})

    model = MajorityClassBaseline().fit(X_tr, y_tr)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    assert (preds == 1).all()
    assert (probs[:, 1] == 1.0).all()
    assert (probs[:, 0] == 0.0).all()


def test_historical_probability_baseline():
    X_tr = pd.DataFrame({"f": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y_tr = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])  # P(1) = 0.4
    X_val = pd.DataFrame({"f": [11, 12]})

    model = HistoricalProbabilityBaseline().fit(X_tr, y_tr)
    probs = model.predict_proba(X_val)

    assert np.allclose(probs[:, 1], 0.4)
    assert np.allclose(probs[:, 0], 0.6)
    # Decision threshold 0.5 -> predict 0
    preds = model.predict(X_val)
    assert (preds == 0).all()


def test_median_regression_baseline():
    X_tr = pd.DataFrame({"f": [1, 2, 3]})
    y_tr = pd.Series([1.2, 2.0, 10.0])  # median is 2.0
    X_val = pd.DataFrame({"f": [4, 5]})

    model = MedianRegressionBaseline().fit(X_tr, y_tr)
    preds = model.predict(X_val)

    assert np.allclose(preds, 2.0)
