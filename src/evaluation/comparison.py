"""Model comparison table generation comparing ML models directly against statistical baselines."""

from typing import Any, Dict, List, Optional
import pandas as pd


def generate_classification_comparison_table(
    results: List[Dict[str, Any]],
    baseline_model_name: str = "Historical Baseline",
) -> pd.DataFrame:
    """Generate the standardized comparison table required by Section 20.
    
    Columns:
      Model | Threshold | ROC-AUC | PR-AUC | Log Loss | Brier | Accuracy | Baseline Diff (Acc) | Baseline Diff (Brier)
      
    Args:
        results: List of dicts with keys:
                 ['model', 'threshold', 'roc_auc', 'pr_auc', 'log_loss', 'brier_score', 'accuracy']
        baseline_model_name: Name of the reference baseline model.
        
    Returns:
        DataFrame structured for reporting.
    """
    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Find baseline metrics per threshold
    baseline_lookup = {}
    baseline_rows = df[df["model"].str.lower().str.contains("baseline")]
    for _, row in baseline_rows.iterrows():
        th = row["threshold"]
        baseline_lookup[th] = {
            "accuracy": row.get("accuracy", 0.0),
            "brier_score": row.get("brier_score", 0.0),
            "log_loss": row.get("log_loss", 0.0),
        }

    # Compute differences relative to baseline
    diff_acc = []
    diff_brier = []
    for _, row in df.iterrows():
        th = row["threshold"]
        base = baseline_lookup.get(th)
        if base and row["model"] != baseline_model_name:
            acc_delta = row["accuracy"] - base["accuracy"]
            brier_delta = row["brier_score"] - base["brier_score"]  # lower is better
            diff_acc.append(round(acc_delta, 5))
            diff_brier.append(round(brier_delta, 5))
        else:
            diff_acc.append(0.0)
            diff_brier.append(0.0)

    df["diff_acc_vs_baseline"] = diff_acc
    df["diff_brier_vs_baseline"] = diff_brier

    col_order = [
        "model",
        "threshold",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "accuracy",
        "f1",
        "diff_acc_vs_baseline",
        "diff_brier_vs_baseline",
    ]
    present_cols = [c for c in col_order if c in df.columns]
    return df[present_cols]


def generate_regression_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate regression comparison table: Model | MAE | RMSE | R2 | Median AE | Baseline Diff MAE."""
    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Locate median baseline MAE
    baseline_row = df[df["model"].str.lower().str.contains("median baseline")]
    base_mae = baseline_row.iloc[0]["mae"] if not baseline_row.empty else None

    if base_mae is not None:
        df["diff_mae_vs_baseline"] = (df["mae"] - base_mae).round(5)
    else:
        df["diff_mae_vs_baseline"] = 0.0

    col_order = ["model", "mae", "rmse", "r2", "median_ae", "diff_mae_vs_baseline"]
    present = [c for c in col_order if c in df.columns]
    return df[present]


def to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean Markdown table without external dependencies."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    rows = [[str(val) for val in row] for row in df.itertuples(index=False)]
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))
            
    header_str = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    separator_str = "| " + " | ".join("-" * w for w in col_widths) + " |"
    row_strs = ["| " + " | ".join(val.ljust(w) for val, w in zip(row, col_widths)) + " |" for row in rows]
    
    return "\n".join([header_str, separator_str] + row_strs)
