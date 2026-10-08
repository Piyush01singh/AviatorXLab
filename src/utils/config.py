"""Configuration loading and validation."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the yaml configuration file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration format in {config_path}")

    return config
