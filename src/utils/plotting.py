"""Visualization generator for distribution analysis, calibration, ROC/PR curves, and backtest equity curves."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for server and Windows CLI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.utils.logging import get_logger

logger = get_logger("plotting")


class ResearchVisualizer:
    """Generates publication-quality research figures and saves to results/plots/."""

    def __init__(self, plots_dir: str = "results/plots"):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_distributions(self, df: pd.DataFrame) -> Tuple[Path, Path]:
        """Generate linear and log-scale multiplier distribution plots."""
        multipliers = df["multiplier"].to_numpy()

        # 1. Linear scale (capped at 20x for visibility)
        fig, ax = plt.subplots(figsize=(8, 5))
        capped = np.clip(multipliers, 1.0, 20.0)
        ax.hist(capped, bins=50, color="#1f77b4", edgecolor="black", alpha=0.7)
        ax.set_title("Multiplier Distribution (Capped at 20x)")
        ax.set_xlabel("Crash Multiplier")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.5)
        linear_path = self.plots_dir / "multiplier_distribution_linear.png"
        fig.tight_layout()
        fig.savefig(linear_path, dpi=150)
        plt.close(fig)

        # 2. Log-scale
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(multipliers, bins=np.logspace(0, np.log10(max(10.0, float(multipliers.max()))), 50),
                color="#ff7f0e", edgecolor="black", alpha=0.7)
        ax.set_xscale("log")
        ax.set_title("Multiplier Distribution (Log Scale)")
        ax.set_xlabel("Crash Multiplier (log)")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.5)
        log_path = self.plots_dir / "multiplier_distribution_log.png"
        fig.tight_layout()
        fig.savefig(log_path, dpi=150)
        plt.close(fig)

        return linear_path, log_path

    def plot_calibration_curve(
        self,
        uncal_data: Dict[str, Any],
        cal_data: Dict[str, Any],
        model_name: str = "Model",
        output_filename: str = "calibration_curve.png",
    ) -> Path:
        """Plot reliability diagram comparing uncalibrated vs calibrated probabilities."""
        fig, ax = plt.subplots(figsize=(7, 6))

        # Perfectly calibrated diagonal
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        # Uncalibrated curve
        ax.plot(
            uncal_data["prob_pred"],
            uncal_data["prob_true"],
            "s-",
            color="#d62728",
            label=f"Uncalibrated (ECE = {uncal_data['ece']:.3f})",
        )

        # Calibrated curve
        ax.plot(
            cal_data["prob_pred"],
            cal_data["prob_true"],
            "o-",
            color="#2ca02c",
            label=f"Calibrated (ECE = {cal_data['ece']:.3f})",
        )

        ax.set_title(f"Reliability Diagram: {model_name}")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Empirical Frequency of Positive Class")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

        out_path = self.plots_dir / output_filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def plot_roc_and_pr_curves(
        self,
        curves_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],  # name -> (y_true, y_prob)
        threshold_name: str = "2.0x",
    ) -> Tuple[Path, Path]:
        """Plot ROC and Precision-Recall curves comparing multiple models."""
        # 1. ROC Curves
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        ax_roc.plot([0, 1], [0, 1], "k--", label="Random Baseline (AUC = 0.50)")

        # 2. PR Curves
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

        for name, (y_true, y_prob) in curves_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            prec, rec, _ = precision_recall_curve(y_true, y_prob)

            ax_roc.plot(fpr, tpr, label=name)
            ax_pr.plot(rec, prec, label=name)

        ax_roc.set_title(f"ROC Curves (Target >= {threshold_name})")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.grid(True, linestyle="--", alpha=0.5)
        ax_roc.legend(loc="lower right")

        ax_pr.set_title(f"Precision-Recall Curves (Target >= {threshold_name})")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.grid(True, linestyle="--", alpha=0.5)
        ax_pr.legend(loc="best")

        roc_path = self.plots_dir / f"roc_curves_{threshold_name.replace('.', '_')}.png"
        pr_path = self.plots_dir / f"pr_curves_{threshold_name.replace('.', '_')}.png"

        fig_roc.tight_layout()
        fig_roc.savefig(roc_path, dpi=150)
        plt.close(fig_roc)

        fig_pr.tight_layout()
        fig_pr.savefig(pr_path, dpi=150)
        plt.close(fig_pr)

        return roc_path, pr_path

    def plot_backtest_curves(
        self,
        strategy_logs: Dict[str, pd.DataFrame],
        output_filename: str = "backtest_equity_curves.png",
    ) -> Path:
        """Plot cumulative simulated balance and drawdown curves."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for name, df_trades in strategy_logs.items():
            rounds = df_trades["round_idx"]
            balance = df_trades["balance"]
            cum_max = np.maximum.accumulate(balance)
            drawdown = cum_max - balance

            ax1.plot(rounds, balance, label=name)
            ax2.plot(rounds, -drawdown, label=name)

        ax1.set_title("Offline Simulated Bankroll Balance Curve")
        ax1.set_ylabel("Balance ($)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="best")

        ax2.set_title("Drawdown Curve (Drop from Peak)")
        ax2.set_xlabel("Sequential Round Index")
        ax2.set_ylabel("Drawdown ($)")
        ax2.grid(True, linestyle="--", alpha=0.5)

        out_path = self.plots_dir / output_filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
