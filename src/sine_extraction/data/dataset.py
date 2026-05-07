"""PyTorch Dataset wrapper for numpy sine-wave arrays."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SineDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Map-style dataset wrapping pre-generated numpy arrays.

    Args:
        X: Input windows, shape ``(N, window_size)``, dtype ``float32``.
        y: Target windows, shape ``(N, window_size)``, dtype ``float32``.
        freq_label: 1-hot frequency label, shape ``(num_frequencies,)``,
            shared by all samples (conditioned extraction).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, freq_label: Tensor) -> None:
        self.X: Tensor = torch.from_numpy(X).to(torch.float32)
        self.y: Tensor = torch.from_numpy(y).to(torch.float32)
        self.freq_label: Tensor = freq_label.to(torch.float32)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(mixed_window, frequency_label, target_window)`` tensors."""
        return self.X[idx], self.freq_label, self.y[idx]
