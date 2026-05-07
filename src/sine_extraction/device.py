"""Device auto-detection — returns CUDA if available, else CPU."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available torch.device.

    Returns:
        ``torch.device("cuda")`` if a CUDA GPU is available, otherwise
        ``torch.device("cpu")``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
