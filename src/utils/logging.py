"""Structured logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "aviator_ml", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance with console and optional file logging.
    
    Args:
        name: Name of the logger
        log_file: Optional path to log output file
        level: Logging level (default INFO)
        
    Returns:
        Configured logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
