"""Data loading and synthetic benchmark dataset generation."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger("data_loader")


def load_csv(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load historical crash game rounds from CSV.
    
    Args:
        file_path: Path to CSV file.
        nrows: Optional limit on number of rows to read.
        
    Returns:
        DataFrame containing historical records.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file does not exist: {file_path}")

    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(path, nrows=nrows)
    logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
    return df


def generate_synthetic_crash_data(
    num_rounds: int = 10000,
    seed: int = 42,
    house_edge: float = 0.01,
    instant_crash_rate: float = 0.03,
    start_time: str = "2026-01-01T00:00:00",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate statistically faithful synthetic crash game rounds for pipeline verification.
    
    In standard provably fair crash mechanics:
      - A random uniform float U in [0, 1) is generated.
      - With probability `instant_crash_rate` (or house cut), crash is 1.00x.
      - Otherwise, multiplier is (1.0 - house_edge) / (1.0 - U), rounded down to 2 decimals.
      
    NOTE: As per project rules, this is strictly labelled SYNTHETIC for verification only.
    
    Args:
        num_rounds: Number of rounds to simulate.
        seed: Random seed for reproducibility.
        house_edge: Theoretical house edge percentage (default 1%).
        instant_crash_rate: Probability of instant crash at 1.00x (default 3%).
        start_time: ISO start timestamp.
        output_path: Optional path to save CSV.
        
    Returns:
        DataFrame containing synthetic round_id, timestamp, multiplier, and provably fair seeds.
    """
    np.random.seed(seed)
    logger.info(f"Generating {num_rounds} synthetic crash records (seed={seed})...")

    u = np.random.uniform(0.0, 1.0, size=num_rounds)
    instant_crashes = np.random.uniform(0.0, 1.0, size=num_rounds) < instant_crash_rate

    multipliers = np.zeros(num_rounds, dtype=np.float64)
    # Crash formula
    regular_mults = np.floor(((1.0 - house_edge) / (1.0 - u)) * 100.0) / 100.0
    # Ensure min 1.00
    regular_mults = np.maximum(1.00, regular_mults)

    multipliers[instant_crashes] = 1.00
    multipliers[~instant_crashes] = regular_mults[~instant_crashes]

    # Generate sequential timestamps (average 15 seconds per round with duration depending on multiplier)
    base_dt = datetime.fromisoformat(start_time)
    timestamps = []
    current_dt = base_dt

    for m in multipliers:
        # Round duration roughly correlates with multiplier (e.g. plane flies longer)
        round_duration_sec = min(60.0, float(np.log(m + 1.0) * 4.0)) + np.random.uniform(5.0, 12.0)
        current_dt += timedelta(seconds=round_duration_sec)
        timestamps.append(current_dt.isoformat())

    round_ids = np.arange(100000, 100000 + num_rounds)

    df = pd.DataFrame({
        "round_id": round_ids,
        "timestamp": timestamps,
        "multiplier": multipliers,
        "data_source": "SYNTHETIC_VERIFICATION",
    })

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Saved synthetic dataset to {out}")

    return df
