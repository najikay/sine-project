"""Multi-layer perceptron model for sine-wave extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from sine_extraction.models.base import BaseModel
from sine_extraction.types import MLPConfig

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLPModel(BaseModel):
    """Fully-connected network that maps noisy windows to clean targets.

    Args:
        config: MLP architecture settings (hidden sizes, activation).
        window_size: Number of samples per input/output window.
    """

    def __init__(self, config: MLPConfig, window_size: int) -> None:
        super().__init__()
        activation_cls = _ACTIVATIONS.get(config.activation.lower(), nn.ReLU)
        self._window_size = window_size

        # Input is the mixed window (window_size) + 1-hot label (4)
        layers: list[nn.Module] = []
        in_size = window_size + 4
        for hidden in config.hidden_sizes:
            layers.append(nn.Linear(in_size, hidden))
            layers.append(activation_cls())
            in_size = hidden
        layers.append(nn.Linear(in_size, window_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor, label: Tensor) -> Tensor:
        """Run the forward pass through the MLP.

        Args:
            x: Input tensor of shape ``(batch, window_size)``.
            label: 1-hot label of shape ``(batch, 4)``.

        Returns:
            Output tensor of shape ``(batch, window_size)``.
        """
        inp = torch.cat([x, label], dim=-1)
        return self.network(inp)
