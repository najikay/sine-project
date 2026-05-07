"""Reproducibility utilities — seed Python, NumPy, and PyTorch RNGs."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch (CPU and CUDA) for reproducibility.

    Args:
        seed: Integer seed value to use for all RNG backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
