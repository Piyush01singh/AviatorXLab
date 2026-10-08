"""PyTorch LSTM Architecture for Causal Sequential Crash Multiplier Prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import torch
import torch.nn as nn

from src.utils.device import get_torch_device
from src.utils.logging import get_logger

logger = get_logger("lstm_model")


class LSTMNetwork(nn.Module):
    """Causal, uni-directional LSTM network for multi-threshold crash prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        output_dim: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Strictly uni-directional to prevent future context leakage
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking input sequence tensor (batch_size, seq_len, input_dim).
        
        Returns:
            Logits tensor of shape (batch_size, output_dim).
        """
        out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        # Take representation at the final sequence timestep
        last_timestep = out[:, -1, :]
        dropped = self.dropout(last_timestep)
        logits = self.fc(dropped)
        return logits


class LSTMThresholdClassifier:
    """LSTM model wrapper for inference and checkpoint management."""

    def __init__(
        self,
        sequence_length: int = 10,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        weight_decay: float = 1e-4,
        device_preference: str = "auto",
        random_state: int = 42,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.device_preference = device_preference
        self.random_state = random_state

        self.device, self.device_info = get_torch_device(preference=self.device_preference)
        self.model: Optional[LSTMNetwork] = None
        self.feature_names: List[str] = []
        self.best_val_loss: float = float("inf")
        self.is_fitted: bool = False

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict sigmoid probabilities from input sequence tensor.
        
        Args:
            X_seq: 3D float32 tensor of shape (N, seq_len, features).
            
        Returns:
            np.ndarray of shape (N, output_dim) with probabilities in [0, 1].
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("LSTM model has not been fitted.")

        self.model.eval()
        t_seq = torch.from_numpy(X_seq.astype(np.float32))
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(t_seq), self.batch_size):
                batch = t_seq[i : i + self.batch_size].to(self.device)
                logits = self.model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, X_seq: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X_seq)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()} if self.model else None,
            "feature_names": self.feature_names,
            "best_val_loss": self.best_val_loss,
            "config": {
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "weight_decay": self.weight_decay,
                "random_state": self.random_state,
            },
        }
        joblib.dump(payload, path)
        logger.info(f"Saved LSTM model to {path}")

    @classmethod
    def load(cls, filepath: str) -> "LSTMThresholdClassifier":
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"LSTM checkpoint not found at {filepath}")
        payload = joblib.load(path)
        config = payload["config"].copy()
        config.pop("best_val_loss", None)
        instance = cls(**config)
        instance.feature_names = payload.get("feature_names", [])
        instance.best_val_loss = payload.get("best_val_loss", float("inf"))

        state_dict = payload["model_state"]
        if state_dict is not None:
            input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
            output_dim = state_dict["fc.weight"].shape[0]
            instance.model = LSTMNetwork(
                input_dim=input_dim,
                hidden_size=instance.hidden_size,
                num_layers=instance.num_layers,
                dropout=instance.dropout,
                output_dim=output_dim,
            ).to(instance.device)
            instance.model.load_state_dict({k: v.to(instance.device) for k, v in state_dict.items()})
            instance.is_fitted = True

        return instance
