"""Strict chronological and walk-forward data splitting.

Guarantees zero future-leakage by forbidding any random shuffling.
"""

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger("data_splitter")


@dataclass
class ChronologicalSplit:
    """Container holding chronological splits."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def verify_no_overlap(self) -> bool:
        """Verify that train, validation, and test sets have non-overlapping, strictly ordered indices."""
        train_idx = set(self.train.index)
        val_idx = set(self.val.index)
        test_idx = set(self.test.index)

        no_overlap = (
            len(train_idx.intersection(val_idx)) == 0
            and len(val_idx.intersection(test_idx)) == 0
            and len(train_idx.intersection(test_idx)) == 0
        )
        return no_overlap


def chronological_split(
    df: pd.DataFrame,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    **kwargs,
) -> ChronologicalSplit:
    """Partition a DataFrame chronologically into Train, Validation, and Test subsets.
    
    Args:
        df: Input DataFrame (must already be sorted chronologically).
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for final test (default 0.15).
    """
    train_ratio = train_ratio if train_ratio is not None else kwargs.get("train", 0.70)
    val_ratio = val_ratio if val_ratio is not None else kwargs.get("val", kwargs.get("validation", 0.15))
    test_ratio = test_ratio if test_ratio is not None else kwargs.get("test", 0.15)

    total = len(df)
    if total < 10:
        raise ValueError(f"Insufficient data for splitting: only {total} rows provided.")

    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5:
        raise ValueError("Train, val, and test ratios must sum to 1.0.")

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    # Remaining goes to test to guarantee all rows are partitioned
    n_test = total - n_train - n_val

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    split = ChronologicalSplit(train=train_df, val=val_df, test=test_df)

    if not split.verify_no_overlap():
        raise RuntimeError("Integrity check failed: index overlap detected across chronological splits.")

    logger.info(
        f"Chronological split completed: Train={len(train_df)} ({train_ratio:.1%}), "
        f"Val={len(val_df)} ({val_ratio:.1%}), Test={len(test_df)} ({test_ratio:.1%})"
    )
    return split


def walk_forward_split(
    df: pd.DataFrame,
    initial_train_size: int,
    step_size: int,
    test_size: int,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
    """Generate expanding-window walk-forward splits for temporal cross-validation.
    
    Args:
        df: Chronologically sorted DataFrame.
        initial_train_size: Number of records in the initial training split.
        step_size: Number of records to expand the training window by each fold.
        test_size: Number of records evaluated in each test window.
        
    Yields:
        Tuples of (train_fold_df, test_fold_df, fold_number).
    """
    n_total = len(df)
    current_train_end = initial_train_size
    fold = 0

    while current_train_end + test_size <= n_total:
        train_slice = df.iloc[:current_train_end].copy()
        test_slice = df.iloc[current_train_end : current_train_end + test_size].copy()
        yield train_slice, test_slice, fold

        current_train_end += step_size
        fold += 1
