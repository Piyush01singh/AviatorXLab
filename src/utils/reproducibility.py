"""Reproducibility utilities for setting fixed random seeds across all libraries."""

import os
import random
from typing import Optional
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, environment, and PyTorch (if available).
    
    Args:
        seed: The integer random seed to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seed if torch is installed
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(preference: str = "auto") -> str:
    """Determine the compute device (cuda or cpu) safely based on hardware availability.
    
    Args:
        preference: 'auto', 'cuda', or 'cpu'
        
    Returns:
        Device string: 'cuda' or 'cpu'
    """
    if preference == "cpu":
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"
