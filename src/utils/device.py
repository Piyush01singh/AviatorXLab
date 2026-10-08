"""Device management and hardware detection for PyTorch models."""

import os
from typing import Dict, Tuple
from src.utils.logging import get_logger

logger = get_logger("device_manager")


def get_torch_device(preference: str = "auto") -> Tuple[str, Dict[str, str]]:
    """Detect available hardware (CUDA / CPU) and return device string and metadata.
    
    Args:
        preference: 'auto', 'cuda', or 'cpu'
        
    Returns:
        Tuple of:
          - device_str: "cuda" or "cpu"
          - device_info: Dictionary containing device name, capability, and VRAM info.
    """
    if preference == "cpu":
        logger.info("Hardware selection: Forced CPU execution.")
        return "cpu", {"device": "cpu", "type": "CPU"}

    try:
        import torch

        if torch.cuda.is_available() and preference in ["auto", "cuda"]:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info = {
                "device": "cuda",
                "name": gpu_name,
                "vram_gb": f"{total_vram_gb:.2f} GB",
                "cuda_version": torch.version.cuda or "N/A",
            }
            logger.info(
                f"Hardware selection: CUDA available. Using GPU '{gpu_name}' "
                f"with {total_vram_gb:.2f} GB VRAM."
            )
            return "cuda", info
        else:
            logger.info("Hardware selection: CUDA unavailable. Using CPU fallback.")
            return "cpu", {"device": "cpu", "type": "CPU fallback"}
    except ImportError:
        logger.warning("PyTorch not installed. Using CPU.")
        return "cpu", {"device": "cpu", "type": "CPU fallback (torch not found)"}
