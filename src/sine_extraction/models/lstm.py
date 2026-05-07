"""LSTM model for sine-wave extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from sine_extraction.models.base import BaseModel
from sine_extraction.types import LSTMConfig


class LSTMModel(BaseModel):
    """Long Short-Term Memory network for sequence-to-sequence prediction.

    Each timestep receives 1 signal sample concatenated with the 4-element
    1-hot frequency label, giving an input size of 5 per timestep.

    Args:
        config: LSTM architecture settings (hidden_size, num_layers).
        window_size: Number of samples per input/output window (unused at
            construction but kept for API consistency with MLPModel).
    """

    def __init__(self, config: LSTMConfig, window_size: int) -> None:  # noqa: ARG002
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=5,  # 1 sample + 4 label bits
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        dirs = 2 if config.bidirectional else 1
        linear_in = dirs * config.hidden_size
        self.linear = nn.Linear(linear_in, 1)

    def forward(self, x: Tensor, label: Tensor) -> Tensor:
        """Run the forward pass through the LSTM.

        Args:
            x: Input tensor of shape ``(batch, window_size)``.
            label: 1-hot label of shape ``(batch, 4)``.

        Returns:
            Output tensor of shape ``(batch, window_size)``.
        """
        # (batch, window_size) -> (batch, window_size, 1)
        x_seq = x.unsqueeze(-1)
        # (batch, 4) -> (batch, window_size, 4)
        label_exp = label.unsqueeze(1).expand(-1, x_seq.size(1), -1)
        # (batch, window_size, 5)
        inp = torch.cat([x_seq, label_exp], dim=-1)
        out, _ = self.lstm(inp)
        # (batch, window_size, hidden) -> (batch, window_size, 1) -> (batch, ws)
        return self.linear(out).squeeze(-1)
