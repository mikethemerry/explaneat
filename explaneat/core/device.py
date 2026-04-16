"""Centralized device selection for PyTorch.

Prefers CUDA > CPU by default. MPS (Apple Silicon) is NOT auto-selected
because this codebase uses float64 tensors throughout, and MPS does not
support float64. To force a specific device, set EXPLANEAT_DEVICE
(e.g., "cpu", "cuda", "cuda:0", "mps").
"""
import os
import logging

import torch

logger = logging.getLogger(__name__)

_device = None


def get_device() -> torch.device:
    """Return the PyTorch device to use for training and inference.

    Selection order:
        1. EXPLANEAT_DEVICE env var if set
        2. CUDA if available
        3. CPU fallback

    The selected device is cached after the first call.
    """
    global _device
    if _device is not None:
        return _device

    override = os.environ.get("EXPLANEAT_DEVICE")
    if override:
        _device = torch.device(override)
        logger.info("Using device: %s (EXPLANEAT_DEVICE override)", _device)
        return _device

    if torch.cuda.is_available():
        _device = torch.device("cuda:0")
    else:
        # MPS is intentionally NOT auto-selected: it doesn't support float64,
        # which this codebase relies on throughout. To opt in, set
        # EXPLANEAT_DEVICE=mps (and ensure tensors are float32-compatible).
        _device = torch.device("cpu")

    logger.info("Using device: %s", _device)
    return _device
