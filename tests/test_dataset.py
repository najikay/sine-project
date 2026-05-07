"""Tests for SineDataset — written BEFORE implementation (TDD)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture()
def arrays() -> tuple[np.ndarray, np.ndarray]:
    """Tiny (100, 10) float32 arrays for dataset tests."""
    rng = np.random.default_rng(0)
    X = rng.random((100, 10)).astype(np.float32)
    y = rng.random((100, 10)).astype(np.float32)
    return X, y


@pytest.fixture()
def freq_label() -> torch.Tensor:
    """1-hot label for 10 Hz (index 2 in [1,5,10,20])."""
    label = torch.zeros(4)
    label[2] = 1.0
    return label


# ---------------------------------------------------------------------------
# len(dataset) equals 100
# ---------------------------------------------------------------------------


def test_dataset_len(
    arrays: tuple[np.ndarray, np.ndarray],
    freq_label: torch.Tensor,
) -> None:
    from sine_extraction.data.dataset import SineDataset

    X, y = arrays
    ds = SineDataset(X, y, freq_label)
    assert len(ds) == 100


# ---------------------------------------------------------------------------
# dataset[0] returns a tuple of three tensors (x, label, y)
# ---------------------------------------------------------------------------


def test_dataset_getitem_returns_triple(
    arrays: tuple[np.ndarray, np.ndarray],
    freq_label: torch.Tensor,
) -> None:
    from sine_extraction.data.dataset import SineDataset

    X, y = arrays
    ds = SineDataset(X, y, freq_label)
    item = ds[0]
    assert isinstance(item, tuple)
    assert len(item) == 3
    for t in item:
        assert isinstance(t, torch.Tensor)


# ---------------------------------------------------------------------------
# first tensor (x window) has shape (10,)
# ---------------------------------------------------------------------------


def test_dataset_x_shape(
    arrays: tuple[np.ndarray, np.ndarray],
    freq_label: torch.Tensor,
) -> None:
    from sine_extraction.data.dataset import SineDataset

    X, y = arrays
    ds = SineDataset(X, y, freq_label)
    x_item, _label, _y = ds[0]
    assert x_item.shape == (10,)


# ---------------------------------------------------------------------------
# third tensor (target window) has shape (10,)
# ---------------------------------------------------------------------------


def test_dataset_y_shape(
    arrays: tuple[np.ndarray, np.ndarray],
    freq_label: torch.Tensor,
) -> None:
    from sine_extraction.data.dataset import SineDataset

    X, y = arrays
    ds = SineDataset(X, y, freq_label)
    _x, _label, y_item = ds[0]
    assert y_item.shape == (10,)


# ---------------------------------------------------------------------------
# label tensor has shape (4,)
# ---------------------------------------------------------------------------


def test_dataset_label_shape(
    arrays: tuple[np.ndarray, np.ndarray],
    freq_label: torch.Tensor,
) -> None:
    from sine_extraction.data.dataset import SineDataset

    X, y = arrays
    ds = SineDataset(X, y, freq_label)
    _x, label, _y = ds[0]
    assert label.shape == (4,)


# ---------------------------------------------------------------------------
# tensor dtype is torch.float32
# ---------------------------------------------------------------------------


def test_dataset_dtype(
    arrays: tuple[np.ndarray, np.ndarray],
    freq_label: torch.Tensor,
) -> None:
    from sine_extraction.data.dataset import SineDataset

    X, y = arrays
    ds = SineDataset(X, y, freq_label)
    x_item, label, y_item = ds[0]
    assert x_item.dtype == torch.float32
    assert label.dtype == torch.float32
    assert y_item.dtype == torch.float32


# ---------------------------------------------------------------------------
# values match input arrays
# ---------------------------------------------------------------------------


def test_dataset_values_match(
    arrays: tuple[np.ndarray, np.ndarray],
    freq_label: torch.Tensor,
) -> None:
    from sine_extraction.data.dataset import SineDataset

    X, y = arrays
    ds = SineDataset(X, y, freq_label)
    for i in range(len(ds)):
        x_item, _label, y_item = ds[i]
        np.testing.assert_allclose(x_item.numpy(), X[i], atol=1e-6)
        np.testing.assert_allclose(y_item.numpy(), y[i], atol=1e-6)
