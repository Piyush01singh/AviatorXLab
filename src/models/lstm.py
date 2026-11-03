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

    def fit(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        X_val_seq: Optional[np.ndarray] = None,
        y_val_seq: Optional[np.ndarray] = None,
        eval_set: Optional[Union[Tuple[Any, Any], List[Tuple[Any, Any]]]] = None,
    ) -> "LSTMThresholdClassifier":
        """Fit PyTorch LSTM model with early stopping on sequence validation set."""
        from torch.utils.data import DataLoader, TensorDataset
        import torch.nn as nn
        from src.utils.reproducibility import set_seed

        if eval_set is not None and X_val_seq is None:
            if isinstance(eval_set, list) and len(eval_set) > 0:
                eval_set = eval_set[0]
            if isinstance(eval_set, (tuple, list)) and len(eval_set) == 2:
                X_val_seq, y_val_seq = eval_set[0], eval_set[1]

        set_seed(self.random_state)
        n_samples, seq_len, n_features = X_seq.shape
        self.sequence_length = seq_len
        output_dim = y_seq.shape[1] if y_seq.ndim > 1 else 1

        self.model = LSTMNetwork(
            input_dim=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=output_dim,
        ).to(self.device)

        train_ds = TensorDataset(torch.from_numpy(X_seq.astype(np.float32)), torch.from_numpy(y_seq.astype(np.float32)))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        val_loader = None
        if X_val_seq is not None and y_val_seq is not None:
            val_ds = TensorDataset(torch.from_numpy(X_val_seq.astype(np.float32)), torch.from_numpy(y_val_seq.astype(np.float32)))
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        pos_counts = np.sum(y_seq, axis=0)
        neg_counts = len(y_seq) - pos_counts
        pos_weight_vals = np.clip(neg_counts / np.maximum(pos_counts, 1.0), 0.5, 10.0)
        pos_weight_tensor = torch.from_numpy(pos_weight_vals.astype(np.float32)).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self.model.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        logits = self.model(bx)
                        v_loss += criterion(logits, by).item() * len(bx)
                v_loss /= len(val_ds)

                if v_loss < best_loss:
                    best_loss = v_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            self.best_val_loss = best_loss

        self.is_fitted = True
        return self

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
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], cls):
            inst = payload["model"]
            if "feature_names" in payload and not getattr(inst, "feature_names", None):
                inst.feature_names = payload["feature_names"]
            return inst

        config = payload.get("config", {}).copy() if isinstance(payload, dict) and "config" in payload else {}
        config.pop("best_val_loss", None)
        instance = cls(**config)
        instance.feature_names = payload.get("feature_names", []) if isinstance(payload, dict) else []
        instance.best_val_loss = payload.get("best_val_loss", float("inf")) if isinstance(payload, dict) else float("inf")

        state_dict = payload.get("model_state") if isinstance(payload, dict) else None
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
