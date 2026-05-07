"""Tests for device auto-detection utilities (TDD — tests first)."""

from __future__ import annotations

import torch


def test_get_device_importable() -> None:
    """get_device can be imported from sine_extraction.device."""
    from sine_extraction.device import get_device

    assert callable(get_device)


def test_get_device_returns_torch_device() -> None:
    """get_device() returns a torch.device instance."""
    from sine_extraction.device import get_device

    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_is_cpu_or_cuda() -> None:
    """get_device() returns either 'cpu' or 'cuda' device."""
    from sine_extraction.device import get_device

    device = get_device()
    assert device.type in ("cpu", "cuda")


def test_get_device_cpu_when_no_cuda() -> None:
    """On a CPU-only machine, get_device() returns cpu device."""
    from sine_extraction.device import get_device

    device = get_device()
    if not torch.cuda.is_available():
        assert device.type == "cpu"
    else:
        assert device.type == "cuda"


def test_get_device_consistent() -> None:
    """get_device() returns the same device on repeated calls."""
    from sine_extraction.device import get_device

    d1 = get_device()
    d2 = get_device()
    assert d1.type == d2.type
