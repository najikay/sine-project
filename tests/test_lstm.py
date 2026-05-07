"""Tests for LSTMModel — written BEFORE implementation (TDD)."""

from __future__ import annotations

import pytest
import torch

from sine_extraction.types import LSTMConfig, RNNConfig


@pytest.fixture()
def lstm_cfg() -> LSTMConfig:
    """Default LSTMConfig (unidirectional to keep test dims simple)."""
    return LSTMConfig(hidden_size=64, num_layers=2, bidirectional=False)


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


def test_lstm_instantiates(lstm_cfg: LSTMConfig, window_size: int) -> None:
    from sine_extraction.models.lstm import LSTMModel

    model = LSTMModel(lstm_cfg, window_size)
    assert model is not None


# ---------------------------------------------------------------------------
# forward output shape
# ---------------------------------------------------------------------------


def test_lstm_forward_shape(
    lstm_cfg: LSTMConfig, window_size: int, label: torch.Tensor
) -> None:
    from sine_extraction.models.lstm import LSTMModel

    model = LSTMModel(lstm_cfg, window_size)
    x = torch.randn(4, window_size)
    out = model(x, label)
    assert out.shape == (4, window_size)


# ---------------------------------------------------------------------------
# no NaN in output
# ---------------------------------------------------------------------------


def test_lstm_forward_no_nan(
    lstm_cfg: LSTMConfig, window_size: int, label: torch.Tensor
) -> None:
    from sine_extraction.models.lstm import LSTMModel

    model = LSTMModel(lstm_cfg, window_size)
    x = torch.randn(4, window_size)
    out = model(x, label)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# count_parameters returns positive int
# ---------------------------------------------------------------------------


def test_lstm_count_parameters(lstm_cfg: LSTMConfig, window_size: int) -> None:
    from sine_extraction.models.lstm import LSTMModel

    model = LSTMModel(lstm_cfg, window_size)
    n = model.count_parameters()
    assert isinstance(n, int)
    assert n > 0


# ---------------------------------------------------------------------------
# model is nn.Module
# ---------------------------------------------------------------------------


def test_lstm_is_nn_module(lstm_cfg: LSTMConfig, window_size: int) -> None:
    from sine_extraction.models.lstm import LSTMModel

    model = LSTMModel(lstm_cfg, window_size)
    assert isinstance(model, torch.nn.Module)


# ---------------------------------------------------------------------------
# hidden_size and num_layers reflected in internal LSTM
# ---------------------------------------------------------------------------


def test_lstm_internal_config(lstm_cfg: LSTMConfig, window_size: int) -> None:
    from sine_extraction.models.lstm import LSTMModel

    model = LSTMModel(lstm_cfg, window_size)
    assert model.lstm.hidden_size == lstm_cfg.hidden_size
    assert model.lstm.num_layers == lstm_cfg.num_layers


# ---------------------------------------------------------------------------
# LSTM produces different output from RNN (sanity check)
# ---------------------------------------------------------------------------


def test_lstm_differs_from_rnn(
    lstm_cfg: LSTMConfig, window_size: int, label: torch.Tensor
) -> None:
    from sine_extraction.models.lstm import LSTMModel
    from sine_extraction.models.rnn import RNNModel

    torch.manual_seed(0)
    rnn_cfg = RNNConfig(
        hidden_size=lstm_cfg.hidden_size,
        num_layers=lstm_cfg.num_layers,
        bidirectional=False,
    )
    rnn_model = RNNModel(rnn_cfg, window_size)
    lstm_model = LSTMModel(lstm_cfg, window_size)

    x = torch.randn(4, window_size)
    rnn_out = rnn_model(x, label)
    lstm_out = lstm_model(x, label)
    assert not torch.allclose(rnn_out, lstm_out)
