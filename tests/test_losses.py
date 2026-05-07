"""Tests for training/losses.py — written BEFORE implementation (TDD)."""

from __future__ import annotations

import torch


def test_mse_loss_returns_scalar() -> None:
    """5.1.2 mse_loss returns a scalar tensor."""
    from sine_extraction.training.losses import mse_loss

    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    result = mse_loss(pred, target)
    assert result.ndim == 0  # scalar


def test_mse_loss_identical_inputs_is_zero() -> None:
    """5.1.3 mse_loss(x, x) == 0.0 for any x."""
    from sine_extraction.training.losses import mse_loss

    x = torch.randn(4, 10)
    result = mse_loss(x, x)
    assert result.item() == 0.0


def test_mse_loss_is_symmetric() -> None:
    """5.1.4 mse_loss is symmetric: mse_loss(a, b) == mse_loss(b, a)."""
    from sine_extraction.training.losses import mse_loss

    a = torch.randn(4, 10)
    b = torch.randn(4, 10)
    assert torch.isclose(mse_loss(a, b), mse_loss(b, a))


def test_mse_loss_known_value() -> None:
    """5.1.5 pred=[1.0], target=[0.0] -> MSE=1.0."""
    from sine_extraction.training.losses import mse_loss

    pred = torch.tensor([1.0])
    target = torch.tensor([0.0])
    result = mse_loss(pred, target)
    assert torch.isclose(result, torch.tensor(1.0))
