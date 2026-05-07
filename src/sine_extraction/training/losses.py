"""Loss functions for sine-wave extraction training."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

_mse = nn.MSELoss()


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute mean squared error between predictions and targets.

    Args:
        pred: Predicted tensor of any shape.
        target: Ground truth tensor matching pred's shape.

    Returns:
        Scalar tensor containing the MSE value.
    """
    return _mse(pred, target)
