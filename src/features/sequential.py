"""Sequential window generator for time-series deep learning models.

Strict Anti-Leakage Guarantee:
For predicting target at round t, the sequence window contains strictly [x(t-N), ..., x(t-1)].
Target is y(t).
There is zero x(t) or later information in the input.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd


def create_sequence_windows(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    sequence_length: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create causal 3D sequence tensors (samples, sequence_length, features) for recurrent models.
    
    For every sample k with target at round t = k + sequence_length:
      - Input sequence: X[t - sequence_length : t] containing [x(t-N), ..., x(t-1)]
      - Target: y[t]
      
    Args:
        X: Feature matrix of shape (N, D).
        y: Target array or DataFrame of length N.
        sequence_length: Length of historical sequence N.
        
    Returns:
        Tuple of:
          - X_seq: 3D float32 tensor of shape (N - sequence_length, sequence_length, D)
          - y_seq: Target array of shape (N - sequence_length, ...)
    """
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y)

    n_samples, n_features = X_arr.shape

    if n_samples <= sequence_length:
        raise ValueError(
            f"Not enough samples ({n_samples}) to create causal sequences of length {sequence_length}."
        )

    num_windows = n_samples - sequence_length

    # Allocate contiguous float32 tensor
    X_seq = np.zeros((num_windows, sequence_length, n_features), dtype=np.float32)
    for k in range(num_windows):
        # Target is at index t = k + sequence_length
        # Input window is strictly [t - sequence_length : t] = [k : k + sequence_length]
        X_seq[k] = X_arr[k : k + sequence_length]

    # Target is at index t = sequence_length onwards
    y_seq = y_arr[sequence_length:]

    return X_seq, y_seq
