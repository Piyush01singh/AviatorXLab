"""Unit tests for sequence training pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.training.train_sequence import SequenceTrainer


@pytest.fixture
def toy_sequence_dataset():
    np.random.seed(42)
    n = 100
    d = 8
    X = pd.DataFrame(np.random.randn(n, d), columns=[f"feat_{i}" for i in range(d)])
    y_thresh = pd.DataFrame({
        "target_1_5x": (X["feat_0"] > 0).astype(int),
        "target_2x": (X["feat_1"] > 0).astype(int),
        "target_3x": (X["feat_2"] > 0).astype(int),
        "target_5x": (X["feat_3"] > 0).astype(int),
        "target_10x": (X["feat_4"] > 0).astype(int),
    })
    return X, y_thresh


def test_sequence_trainer_lstm_and_gru(toy_sequence_dataset, tmp_path):
    X, y_thresh = toy_sequence_dataset
    trainer = SequenceTrainer(config_path="config.yaml")
    trainer.models_dir = tmp_path / "models"
    trainer.predictions_dir = tmp_path / "predictions"
    trainer.models_dir.mkdir(parents=True, exist_ok=True)
    trainer.predictions_dir.mkdir(parents=True, exist_ok=True)
    from src.training.experiment import ExperimentTracker
    trainer.tracker = ExperimentTracker(str(tmp_path / "metrics"))

    # Override for fast test
    trainer.config["models"]["lstm"]["epochs"] = 2
    trainer.config["models"]["lstm"]["batch_size"] = 16
    trainer.config["models"]["lstm"]["sequence_length"] = 5
    trainer.config["models"]["gru"]["epochs"] = 2
    trainer.config["models"]["gru"]["batch_size"] = 16
    trainer.config["models"]["gru"]["sequence_length"] = 5

    # Test LSTM training
    lstm_results = trainer.train_recurrent_model("lstm", X, y_thresh, sequence_length=5)
    assert len(lstm_results) == 5  # 5 thresholds
    assert lstm_results[0]["model"] == "LSTM"
    assert "accuracy" in lstm_results[0]
    assert "brier_score" in lstm_results[0]

    # Test GRU training
    gru_results = trainer.train_recurrent_model("gru", X, y_thresh, sequence_length=5)
    assert len(gru_results) == 5
    assert gru_results[0]["model"] == "GRU"
    assert "accuracy" in gru_results[0]
