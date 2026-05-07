"""Tests for MLPModel — written BEFORE implementation (TDD)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sine_extraction.types import MLPConfig


@pytest.fixture()
def mlp_cfg() -> MLPConfig:
    """Default MLPConfig with three hidden layers."""
    return MLPConfig(hidden_sizes=[64, 128, 64], activation="relu")


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


def test_mlp_instantiates(mlp_cfg: MLPConfig, window_size: int) -> None:
    from sine_extraction.models.mlp import MLPModel

    model = MLPModel(mlp_cfg, window_size)
    assert model is not None


# ---------------------------------------------------------------------------
# forward output shape
# ---------------------------------------------------------------------------


def test_mlp_forward_shape(
    mlp_cfg: MLPConfig, window_size: int, label: torch.Tensor
) -> None:
    from sine_extraction.models.mlp import MLPModel

    model = MLPModel(mlp_cfg, window_size)
    x = torch.randn(4, window_size)
    out = model(x, label)
    assert out.shape == (4, window_size)


# ---------------------------------------------------------------------------
# no NaN in output
# ---------------------------------------------------------------------------


def test_mlp_forward_no_nan(
    mlp_cfg: MLPConfig, window_size: int, label: torch.Tensor
) -> None:
    from sine_extraction.models.mlp import MLPModel

    model = MLPModel(mlp_cfg, window_size)
    x = torch.randn(4, window_size)
    out = model(x, label)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# count_parameters returns positive int
# ---------------------------------------------------------------------------


def test_mlp_count_parameters(mlp_cfg: MLPConfig, window_size: int) -> None:
    from sine_extraction.models.mlp import MLPModel

    model = MLPModel(mlp_cfg, window_size)
    n = model.count_parameters()
    assert isinstance(n, int)
    assert n > 0


# ---------------------------------------------------------------------------
# save writes a .pt file
# ---------------------------------------------------------------------------


def test_mlp_save(
    mlp_cfg: MLPConfig, window_size: int, tmp_path: Path
) -> None:
    from sine_extraction.models.mlp import MLPModel

    model = MLPModel(mlp_cfg, window_size)
    save_path = tmp_path / "mlp.pt"
    model.save(save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# hidden_sizes creates correct number of hidden layers
# ---------------------------------------------------------------------------


def test_mlp_hidden_layers(mlp_cfg: MLPConfig, window_size: int) -> None:
    from sine_extraction.models.mlp import MLPModel

    model = MLPModel(mlp_cfg, window_size)
    linear_count = sum(
        1 for m in model.modules() if isinstance(m, torch.nn.Linear)
    )
    assert linear_count == len(mlp_cfg.hidden_sizes) + 1


# ---------------------------------------------------------------------------
# model is nn.Module
# ---------------------------------------------------------------------------


def test_mlp_is_nn_module(mlp_cfg: MLPConfig, window_size: int) -> None:
    from sine_extraction.models.mlp import MLPModel

    model = MLPModel(mlp_cfg, window_size)
    assert isinstance(model, torch.nn.Module)
