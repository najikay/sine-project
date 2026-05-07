"""Abstract base class shared by all sine-extraction model architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """Abstract base for MLP, RNN, and LSTM models.

    All concrete subclasses must implement :meth:`forward`.  The base class
    provides :meth:`count_parameters` and :meth:`save` so that training and
    evaluation code can treat every architecture uniformly.
    """

    @abstractmethod
    def forward(self, x: Tensor, label: Tensor) -> Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor of shape ``(batch, window_size)``.
            label: 1-hot frequency label of shape ``(batch, num_frequencies)``.

        Returns:
            Predicted tensor of shape ``(batch, window_size)``.
        """

    def count_parameters(self) -> int:
        """Return the number of trainable parameters.

        Returns:
            Total count of parameters that require gradients.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Path) -> None:
        """Persist the model's state dictionary to *path*.

        Args:
            path: Destination file path (typically a ``.pt`` file).
        """
        torch.save(self.state_dict(), path)
