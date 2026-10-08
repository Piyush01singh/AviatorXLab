"""Shared training pipeline for causal sequence models (LSTM and GRU)."""

from datetime import datetime
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

from src.data.cleaner import DataCleaner
from src.data.loader import generate_synthetic_crash_data, load_csv
from src.data.splitter import chronological_split
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.classification import compute_classification_metrics
from src.features.builder import THRESHOLD_COLUMN_MAP, FeatureBuilder
from src.features.sequential import create_sequence_windows
from src.models.gru import GRUNetwork, GRUThresholdClassifier
from src.models.lstm import LSTMNetwork, LSTMThresholdClassifier
from src.training.experiment import ExperimentRecord, ExperimentTracker
from src.utils.config import load_config
from src.utils.device import get_torch_device
from src.utils.logging import get_logger
from src.utils.reproducibility import set_seed

logger = get_logger("train_sequence")


class SequenceTrainer:
    """Unified training and evaluation engine for recurrent neural architectures (LSTM, GRU)."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.seed = self.config["project"]["seed"]
        set_seed(self.seed)

        self.device, self.device_info = get_torch_device(self.config["hardware"].get("device", "auto"))

        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.predictions_dir = Path(self.config["paths"]["results_predictions_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = ExperimentTracker(self.config["paths"]["results_metrics_dir"])
        self.builder = FeatureBuilder(
            lags=self.config["features"]["lag_rounds"],
            rolling_windows=self.config["features"]["rolling_windows"],
            thresholds=self.config["targets"]["thresholds"],
            include_temporal=self.config["features"]["include_temporal"],
        )

    def prepare_data(self, data_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Loads and builds features and target matrices."""
        file_path = data_file or self.config["data"]["default_file"]
        path = Path(file_path)

        if not path.is_file():
            logger.warning(f"Data file {file_path} not found. Generating synthetic verification data...")
            generate_synthetic_crash_data(num_rounds=10000, seed=self.seed, output_path=file_path)

        raw_df = load_csv(file_path)
        cleaner = DataCleaner()
        clean_df, _ = cleaner.clean(raw_df)

        X, y_thresh, y_reg, _ = self.builder.build(clean_df)
        return X, y_thresh, y_reg

    def train_recurrent_model(
        self,
        model_type: str,
        X: pd.DataFrame,
        y_thresh: pd.DataFrame,
        sequence_length: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Train an LSTM or GRU model across all configured threshold targets.
        
        Args:
            model_type: 'lstm' or 'gru'
            X: Tabular feature matrix.
            y_thresh: Multi-threshold binary targets DataFrame.
            sequence_length: Length of lookback sequence window.
        """
        model_type = model_type.lower()
        if model_type not in ["lstm", "gru"]:
            raise ValueError(f"Unsupported model type: {model_type}. Expected 'lstm' or 'gru'.")

        cfg = self.config["models"][model_type]
        seq_len = sequence_length or cfg.get("sequence_length", 10)
        hidden_size = cfg.get("hidden_size", 64)
        num_layers = cfg.get("num_layers", 1)
        dropout = cfg.get("dropout", 0.2)
        lr = cfg.get("learning_rate", 0.001)
        batch_size = cfg.get("batch_size", 64)
        epochs = cfg.get("epochs", 50)
        patience = cfg.get("early_stopping_patience", 10)

        # 1. Chronological splitting of tabular representations
        split_ratios = self.config["data"]["split_ratios"]
        split_X = chronological_split(X, **split_ratios)
        split_y_th = chronological_split(y_thresh, **split_ratios)

        # 2. Fit scaler ONLY on training data
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(split_X.train.values).astype(np.float32)
        X_vl_scaled = scaler.transform(split_X.val.values).astype(np.float32)

        # 3. Create causal 3D sequence tensors
        # Train window
        X_tr_seq, y_tr_seq = create_sequence_windows(
            X_tr_scaled, split_y_th.train.values.astype(np.float32), sequence_length=seq_len
        )
        # Val window
        X_vl_seq, y_vl_seq = create_sequence_windows(
            X_vl_scaled, split_y_th.val.values.astype(np.float32), sequence_length=seq_len
        )

        n_samples_tr, _, n_features = X_tr_seq.shape
        output_dim = y_tr_seq.shape[1]

        logger.info(
            f"Building {model_type.upper()} sequence datasets: "
            f"Train={X_tr_seq.shape}, Val={X_vl_seq.shape}, seq_len={seq_len}"
        )

        train_ds = TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq))
        val_ds = TensorDataset(torch.from_numpy(X_vl_seq), torch.from_numpy(y_vl_seq))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 4. Model instantiation
        if model_type == "lstm":
            net = LSTMNetwork(
                input_dim=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                output_dim=output_dim,
            ).to(self.device)
        else:
            net = GRUNetwork(
                input_dim=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                output_dim=output_dim,
            ).to(self.device)

        # Inspect class imbalance on training targets to calculate pos_weight if justified
        pos_counts = np.sum(y_tr_seq, axis=0)
        neg_counts = len(y_tr_seq) - pos_counts
        pos_weight_vals = np.clip(neg_counts / np.maximum(pos_counts, 1.0), 0.5, 10.0)
        pos_weight_tensor = torch.from_numpy(pos_weight_vals.astype(np.float32)).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

        # 5. Training loop with early stopping
        t_start = time.time()
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        logger.info(f"Training {model_type.upper()} on {self.device} for up to {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            net.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = net(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= len(train_ds)

            # Validation step
            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    logits = net(bx)
                    loss = criterion(logits, by)
                    val_loss += loss.item() * len(bx)
            val_loss /= len(val_ds)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.5f}")
                break

        train_time = time.time() - t_start

        # 6. Restore best validation checkpoint
        if best_state is not None:
            net.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # 7. Validation inference
        net.eval()
        t0 = time.time()
        all_probs = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(self.device)
                logits = net(bx)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        infer_time = time.time() - t0

        val_probs_matrix = np.vstack(all_probs)

        # 8. Compute metrics per threshold
        threshold_list = self.config["targets"]["thresholds"]
        results = []
        preds_df = pd.DataFrame()

        for idx, th in enumerate(threshold_list):
            y_vl_th = y_vl_seq[:, idx]
            th_probs = val_probs_matrix[:, idx]

            metrics = compute_classification_metrics(y_vl_th, th_probs)
            calib = compute_calibration_metrics(y_vl_th, th_probs)
            metrics["ece"] = calib["ece"]

            model_display_name = model_type.upper()
            results.append({
                "model": model_display_name,
                "threshold": f"{th}x",
                **metrics,
            })
            preds_df[f"{model_type}_prob_{th}x"] = th_probs

            # Log experiment record
            exp_record = ExperimentRecord(
                experiment_id=f"exp_{model_type}_{th}x_seq{seq_len}_{int(time.time())}",
                model_name=model_display_name,
                task_type="threshold_classification",
                threshold=f"{th}x",
                dataset_version=self.config["data"]["dataset_version"],
                feature_version=self.config["features"]["feature_version"],
                train_range={"samples": len(X_tr_seq), "seq_len": seq_len},
                validation_range={"samples": len(X_vl_seq), "seq_len": seq_len},
                test_range=None,
                hyperparameters={
                    "sequence_length": seq_len,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "device": self.device,
                    "best_val_loss": round(best_val_loss, 5),
                },
                random_seed=self.seed,
                training_time_sec=round(train_time, 4),
                inference_time_sec=round(infer_time, 4),
                metrics=metrics,
            )
            self.tracker.log(exp_record)

        # 9. Save predictions and model artifact
        preds_file = self.predictions_dir / f"val_predictions_{model_type}.csv"
        preds_df.to_csv(preds_file, index=False)

        model_path = self.models_dir / f"{model_type}_model.joblib"
        save_payload = {
            "model_type": model_type,
            "model_state": {k: v.cpu() for k, v in net.state_dict().items()},
            "scaler": scaler,
            "feature_names": list(X.columns),
            "config": {
                "sequence_length": seq_len,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": lr,
                "batch_size": batch_size,
                "best_val_loss": best_val_loss,
            },
        }
        joblib.dump(save_payload, model_path)
        logger.info(f"Saved {model_type.upper()} model checkpoint to {model_path}")

        return results
