"""Expanding-window walk-forward temporal cross-validation.

Strict Anti-Leakage Guarantee:
Train split is strictly prior to test split in every fold. Scalers are fitted on train split only.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.cleaner import DataCleaner
from src.data.loader import generate_synthetic_crash_data, load_csv
from src.data.splitter import walk_forward_split
from src.evaluation.classification import compute_classification_metrics
from src.evaluation.comparison import to_markdown_table
from src.features.builder import THRESHOLD_COLUMN_MAP, FeatureBuilder
from src.features.sequential import create_sequence_windows
from src.models.baseline import HistoricalProbabilityBaseline, RollingHistoricalProbabilityBaseline
from src.models.gru import GRUNetwork
from src.models.logistic import LogisticThresholdClassifier
from src.models.lstm import LSTMNetwork
from src.models.mlp import MLPThresholdClassifier
from src.models.random_forest import RandomForestThresholdClassifier
from src.models.xgboost_model import XGBoostThresholdClassifier
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("walk_forward")


class WalkForwardEngine:
    """Executes chronological expanding-window walk-forward cross-validation."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.reports_dir = Path(self.config["paths"]["results_reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.builder = FeatureBuilder(
            lags=self.config["features"]["lag_rounds"],
            rolling_windows=self.config["features"]["rolling_windows"],
            thresholds=self.config["targets"]["thresholds"],
            include_temporal=self.config["features"]["include_temporal"],
        )

    def _train_recurrent_fold(
        self,
        model_type: str,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_te: pd.DataFrame,
        y_te: pd.Series,
        seq_len: int = 10,
        epochs: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Causal fold training and evaluation for LSTM / GRU sequence models."""
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr.values).astype(np.float32)
        X_te_sc = scaler.transform(X_te.values).astype(np.float32)

        y_tr_vals = y_tr.values.astype(np.float32).reshape(-1, 1)
        y_te_vals = y_te.values.astype(np.float32).reshape(-1, 1)

        X_tr_s, y_tr_s = create_sequence_windows(X_tr_sc, y_tr_vals, seq_len)

        # Prepend last seq_len train steps to test inputs for zero-drop causal sequence evaluation
        concat_te = np.vstack([X_tr_sc[-seq_len:], X_te_sc])
        concat_y = np.vstack([y_tr_vals[-seq_len:], y_te_vals])
        X_te_s, y_te_s = create_sequence_windows(concat_te, concat_y, seq_len)

        device = torch.device("cpu")
        feat_dim = X_tr_s.shape[2]
        if model_type == "lstm":
            net = LSTMNetwork(input_dim=feat_dim, hidden_size=32, num_layers=1, dropout=0.0, output_dim=1).to(device)
        else:
            net = GRUNetwork(input_dim=feat_dim, hidden_size=32, num_layers=1, dropout=0.0, output_dim=1).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

        ds = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_s))
        loader = DataLoader(ds, batch_size=128, shuffle=True)

        net.train()
        for _ in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = net(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        net.eval()
        with torch.no_grad():
            preds = torch.sigmoid(net(torch.from_numpy(X_te_s))).numpy().ravel()

        return y_te_s.ravel(), preds

    def run_walk_forward(
        self,
        X: pd.DataFrame,
        y_thresh: pd.DataFrame,
        target_col: str = "target_2x",
        initial_train_size: Optional[int] = None,
        step_size: Optional[int] = None,
        test_size: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute expanding-window folds across historical timeline.
        
        Args:
            X: Tabular feature matrix.
            y_thresh: Target DataFrame.
            target_col: Target to evaluate (default 'target_2x').
            initial_train_size: Starting training records count.
            step_size: Expansion step per fold.
            test_size: Size of out-of-sample test window.
            
        Returns:
            Tuple of (fold_results_df, summary_stats_df).
        """
        n_total = len(X)
        init_train = initial_train_size or max(2000, int(n_total * 0.50))
        step = step_size or max(500, int(n_total * 0.15))
        test_s = test_size or step

        combined = pd.concat([X, y_thresh[[target_col]]], axis=1)
        feat_cols = list(X.columns)

        fold_records = []
        folds = list(walk_forward_split(combined, initial_train_size=init_train, step_size=step, test_size=test_s))
        logger.info(f"Running walk-forward validation with {len(folds)} expanding folds...")

        for train_df, test_df, fold_idx in folds:
            X_tr, y_tr = train_df[feat_cols], train_df[target_col]
            X_te, y_te = test_df[feat_cols], test_df[target_col]

            # 1. Historical Baseline
            hist_model = HistoricalProbabilityBaseline().fit(X_tr, y_tr)
            hist_probs = hist_model.predict_proba(X_te)[:, 1]
            fold_records.append({"fold": fold_idx, "model": "Historical Baseline", **compute_classification_metrics(y_te, hist_probs)})

            # 2. Rolling Probability Baseline
            roll_model = RollingHistoricalProbabilityBaseline(window=50).fit(X_tr, y_tr)
            roll_probs = roll_model.predict_proba(X_te, past_target_series=y_te)[:, 1]
            fold_records.append({"fold": fold_idx, "model": "Rolling Baseline", **compute_classification_metrics(y_te, roll_probs)})

            # 3. Logistic Regression
            log_model = LogisticThresholdClassifier(max_iter=500).fit(X_tr, y_tr)
            log_probs = log_model.predict_proba(X_te)[:, 1]
            fold_records.append({"fold": fold_idx, "model": "Logistic Regression", **compute_classification_metrics(y_te, log_probs)})

            # 4. Random Forest
            rf_model = RandomForestThresholdClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
            rf_probs = rf_model.predict_proba(X_te)[:, 1]
            fold_records.append({"fold": fold_idx, "model": "Random Forest", **compute_classification_metrics(y_te, rf_probs)})

            # 5. XGBoost
            xgb_model = XGBoostThresholdClassifier(n_estimators=100, max_depth=3).fit(X_tr, y_tr)
            xgb_probs = xgb_model.predict_proba(X_te)[:, 1]
            fold_records.append({"fold": fold_idx, "model": "XGBoost", **compute_classification_metrics(y_te, xgb_probs)})

            # 6. MLP
            mlp_model = MLPThresholdClassifier(hidden_dims=[32, 16], epochs=15, batch_size=64, device_preference="cpu")
            mlp_model.fit(X_tr, y_tr)
            mlp_probs = mlp_model.predict_proba(X_te)[:, 0]
            fold_records.append({"fold": fold_idx, "model": "MLP", **compute_classification_metrics(y_te, mlp_probs)})

            # 7. LSTM (Recurrent Sequence)
            y_te_lstm, lstm_probs = self._train_recurrent_fold("lstm", X_tr, y_tr, X_te, y_te, seq_len=10, epochs=8)
            fold_records.append({"fold": fold_idx, "model": "LSTM", **compute_classification_metrics(y_te_lstm, lstm_probs)})

            # 8. GRU (Recurrent Sequence)
            y_te_gru, gru_probs = self._train_recurrent_fold("gru", X_tr, y_tr, X_te, y_te, seq_len=10, epochs=8)
            fold_records.append({"fold": fold_idx, "model": "GRU", **compute_classification_metrics(y_te_gru, gru_probs)})

        results_df = pd.DataFrame(fold_records)

        # Compute summary statistics (mean, std, 95% CI)
        metric_keys = ["accuracy", "brier_score", "roc_auc", "pr_auc", "log_loss"]
        summary_rows = []

        for model_name, grp in results_df.groupby("model"):
            row = {"model": model_name, "n_folds": len(grp)}
            for m in metric_keys:
                vals = grp[m].dropna().to_numpy()
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ci_half = (1.96 * std_val / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

                row[f"{m}_mean"] = round(mean_val, 4)
                row[f"{m}_std"] = round(std_val, 4)
                row[f"{m}_ci95"] = f"[{round(mean_val - ci_half, 4)}, {round(mean_val + ci_half, 4)}]"

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        # Save artifacts
        out_csv = self.reports_dir / "walk_forward_results.csv"
        results_df.to_csv(out_csv, index=False)

        summary_csv = self.reports_dir / "walk_forward_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        summary_md = self.reports_dir / "walk_forward_summary.md"
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write(f"# Walk-Forward Temporal Cross-Validation Summary ({target_col})\n\n")
            f.write(to_markdown_table(summary_df))
            f.write("\n")

        logger.info(f"Walk-forward validation complete. Saved reports to {summary_md}")
        return results_df, summary_df
