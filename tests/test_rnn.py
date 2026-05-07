"""Tests for RNNModel — written BEFORE implementation (TDD)."""

from __future__ import annotations

import pytest
import torch

from sine_extraction.types import RNNConfig


@pytest.fixture()
def rnn_cfg() -> RNNConfig:
    """Default RNNConfig (unidirectional to keep test dims simple)."""
    return RNNConfig(hidden_size=64, num_layers=2, bidirectional=False)


@pytest.fixture()
def window_size() -> int:
    """Window size matching the project default."""
    return 10


@pytest.fixture()
def label() -> torch.Tensor:
    """Batch of 1-hot frequency labels, shape (4, 4)."""
    lbl = torch.zeros(4, 4)
    lbl[:, 2] = 1.0  # 10 Hz at index 2
    return lbl


# ---------------------------------------------------------------------------
# instantiation
# ---------------------------------------------------------------------------


def test_rnn_instantiates(rnn_cfg: RNNConfig, window_size: int) -> None:
    from sine_extraction.models.rnn import RNNModel

    model = RNNModel(rnn_cfg, window_size)
    assert model is not None


# ---------------------------------------------------------------------------
# forward output shape
# ---------------------------------------------------------------------------


def test_rnn_forward_shape(
    rnn_cfg: RNNConfig, window_size: int, label: torch.Tensor
) -> None:
    from sine_extraction.models.rnn import RNNModel

    model = RNNModel(rnn_cfg, window_size)
    x = torch.randn(4, window_size)
    out = model(x, label)
    assert out.shape == (4, window_size)


# ---------------------------------------------------------------------------
# no NaN in output
# ---------------------------------------------------------------------------


def test_rnn_forward_no_nan(
    rnn_cfg: RNNConfig, window_size: int, label: torch.Tensor
) -> None:
    from sine_extraction.models.rnn import RNNModel

    model = RNNModel(rnn_cfg, window_size)
    x = torch.randn(4, window_size)
    out = model(x, label)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# count_parameters returns positive int
# ---------------------------------------------------------------------------


def test_rnn_count_parameters(rnn_cfg: RNNConfig, window_size: int) -> None:
    from sine_extraction.models.rnn import RNNModel

    model = RNNModel(rnn_cfg, window_size)
    n = model.count_parameters()
    assert isinstance(n, int)
    assert n > 0


# ---------------------------------------------------------------------------
# model is nn.Module
# ---------------------------------------------------------------------------


def test_rnn_is_nn_module(rnn_cfg: RNNConfig, window_size: int) -> None:
    from sine_extraction.models.rnn import RNNModel

    model = RNNModel(rnn_cfg, window_size)
    assert isinstance(model, torch.nn.Module)


# ---------------------------------------------------------------------------
# hidden_size and num_layers reflected in internal RNN
# ---------------------------------------------------------------------------


def test_rnn_internal_config(rnn_cfg: RNNConfig, window_size: int) -> None:
    from sine_extraction.models.rnn import RNNModel

    model = RNNModel(rnn_cfg, window_size)
    assert model.rnn.hidden_size == rnn_cfg.hidden_size
    assert model.rnn.num_layers == rnn_cfg.num_layers
