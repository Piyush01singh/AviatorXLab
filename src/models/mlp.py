"""Multi-Layer Perceptron (MLP) for multi-threshold crash multiplier prediction."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.device import get_torch_device
from src.utils.logging import get_logger
from src.utils.reproducibility import set_seed

logger = get_logger("mlp_model")


class MLPNetwork(nn.Module):
    """Feedforward neural network for multi-label threshold prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.2,
        output_dim: int = 5,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (batch_size, output_dim)."""
        return self.net(x)


class MLPThresholdClassifier:
    """Tabular PyTorch MLP Classifier predicting 5 threshold probabilities simultaneously."""

    def __init__(
        self,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        weight_decay: float = 1e-4,
        device_preference: str = "auto",
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.device_preference = device_preference
        self.random_state = random_state

        self.device, self.device_info = get_torch_device(preference=self.device_preference)
        self.scaler = StandardScaler()
        self.model: Optional[MLPNetwork] = None
        self.feature_names: List[str] = []
        self.best_val_loss: float = float("inf")
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.DataFrame, np.ndarray],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> "MLPThresholdClassifier":
        """Train MLP with early stopping on validation loss.
        
        Args:
            X_train: Training features.
            y_train: Multi-threshold targets matrix (N, num_thresholds).
            X_val: Validation features (for early stopping).
            y_val: Validation targets matrix.
            pos_weight: Optional positive class weighting tensor.
        """
        set_seed(self.random_state)
        self.feature_names = list(X_train.columns)
        input_dim = len(self.feature_names)

        y_tr_arr = np.asarray(y_train, dtype=np.float32)
        if y_tr_arr.ndim == 1:
            y_tr_arr = y_tr_arr.reshape(-1, 1)
        output_dim = y_tr_arr.shape[1]

        # Fit scaler ONLY on training data
        X_tr_scaled = self.scaler.fit_transform(X_train.values).astype(np.float32)

        train_ds = TensorDataset(torch.from_numpy(X_tr_scaled), torch.from_numpy(y_tr_arr))
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_vl_scaled = self.scaler.transform(X_val[self.feature_names].values).astype(np.float32)
            y_vl_arr = np.asarray(y_val, dtype=np.float32)
            if y_vl_arr.ndim == 1:
                y_vl_arr = y_vl_arr.reshape(-1, 1)
            val_ds = TensorDataset(torch.from_numpy(X_vl_scaled), torch.from_numpy(y_vl_arr))
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Initialize network
        self.model = MLPNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            output_dim=output_dim,
        ).to(self.device)

        criterion = (
            nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            if pos_weight is not None
            else nn.BCEWithLogitsLoss()
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        best_state = None
        self.best_val_loss = float("inf")
        patience_counter = 0

        logger.info(
            f"Training MLP (input={input_dim}, hidden={self.hidden_dims}, out={output_dim}) "
            f"on {self.device} for up to {self.epochs} epochs..."
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(bx)

            train_loss /= len(train_ds)

            # Validation step
            if has_val:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        logits = self.model(bx)
                        loss = criterion(logits, by)
                        val_loss += loss.item() * len(bx)
                val_loss /= len(val_ds)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {self.best_val_loss:.5f}")
                    break
            else:
                self.best_val_loss = train_loss

        # Restore best model state
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict sigmoid probabilities for all thresholds.
        
        Returns:
            np.ndarray of shape (N, output_dim) with probabilities in [0, 1].
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been fitted.")

        self.model.eval()
        X_scaled = self.scaler.transform(X[self.feature_names].values).astype(np.float32)
        ds = TensorDataset(torch.from_numpy(X_scaled))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                logits = self.model(bx)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary discrete labels based on threshold probability."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        """Serialize model state, scaler, feature list, and config."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()} if self.model else None,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "best_val_loss": self.best_val_loss,
            "config": {
                "hidden_dims": self.hidden_dims,
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
        logger.info(f"Saved MLP model to {path}")

    @classmethod
    def load(cls, filepath: str) -> "MLPThresholdClassifier":
        """Load serialized MLP model."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"MLP checkpoint not found at {filepath}")
        payload = joblib.load(path)
        instance = cls(**payload["config"])
        instance.scaler = payload["scaler"]
        instance.feature_names = payload["feature_names"]
        instance.best_val_loss = payload.get("best_val_loss", float("inf"))

        state_dict = payload["model_state"]
        if state_dict is not None:
            # Determine output dim from final linear layer
            last_layer_weight = [v for k, v in state_dict.items() if "weight" in k][-1]
            output_dim = last_layer_weight.shape[0]
            instance.model = MLPNetwork(
                input_dim=len(instance.feature_names),
                hidden_dims=instance.hidden_dims,
                dropout=instance.dropout,
                output_dim=output_dim,
            ).to(instance.device)
            instance.model.load_state_dict({k: v.to(instance.device) for k, v in state_dict.items()})
            instance.is_fitted = True

        return instance
