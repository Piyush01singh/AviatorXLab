"""Experiment tracking and audit logging.

Logs experiment metadata, dataset version, hyperparameters, execution times,
and evaluation metrics to ensure exact reproducibility.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger("experiment_tracker")


@dataclass
class ExperimentRecord:
    """Standardized record for each model evaluation experiment."""
    experiment_id: str
    model_name: str
    task_type: str  # 'threshold_classification' or 'multiplier_regression'
    threshold: Optional[str]
    dataset_version: str
    feature_version: str
    train_range: Dict[str, Any]
    validation_range: Dict[str, Any]
    test_range: Optional[Dict[str, Any]]
    hyperparameters: Dict[str, Any]
    random_seed: int
    training_time_sec: float
    inference_time_sec: float
    metrics: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExperimentTracker:
    """Manages tracking, persistence, and retrieval of experiment runs."""

    def __init__(self, metrics_dir: str = "results/metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.metrics_dir / "experiment_history.jsonl"

    def log(self, record: ExperimentRecord) -> Path:
        """Save experiment record to JSON file and append to JSONL history."""
        rec_dict = asdict(record)

        # 1. Save individual run file
        run_file = self.metrics_dir / f"{record.experiment_id}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(rec_dict, f, indent=2)

        # 2. Append to continuous log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec_dict) + "\n")

        logger.info(f"Logged experiment {record.experiment_id} ({record.model_name}) to {run_file}")
        return run_file

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Load all recorded experiments."""
        if not self.log_file.exists():
            return []
        records = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records

    def to_dataframe(self) -> pd.DataFrame:
        """Convert experiment history to a flattened DataFrame."""
        records = self.get_all_records()
        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            row = {
                "experiment_id": r["experiment_id"],
                "model_name": r["model_name"],
                "task_type": r["task_type"],
                "threshold": r.get("threshold"),
                "dataset_version": r["dataset_version"],
                "training_time_sec": r["training_time_sec"],
                "inference_time_sec": r["inference_time_sec"],
                "random_seed": r["random_seed"],
            }
            # Flatten metrics
            for k, v in r.get("metrics", {}).items():
                if isinstance(v, (int, float)):
                    row[k] = v
            rows.append(row)

        return pd.DataFrame(rows)
