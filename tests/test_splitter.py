"""Tests for make_dataloaders — written BEFORE implementation (TDD)."""

from __future__ import annotations

import numpy as np
import pytest

from sine_extraction.types import DataConfig, SignalConfig


@pytest.fixture()
def signal_cfg() -> SignalConfig:
    """Minimal SignalConfig for splitter tests."""
    return SignalConfig(
        frequencies=[1.0, 5.0, 10.0, 20.0],
        sample_rate=200,
        window_size=10,
        target_frequency=10.0,
        amplitude=1.0,
        amplitude_jitter_std=0.05,
        phase_jitter_std=0.1,
        noise_std=0.05,
    )


@pytest.fixture()
def data_cfg() -> DataConfig:
    """Minimal DataConfig for splitter tests."""
    return DataConfig(
        num_windows=1000,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=32,
        shuffle=True,
    )


@pytest.fixture()
def xy() -> tuple[np.ndarray, np.ndarray]:
    """1000 samples of shape (1000, 10) float32."""
    rng = np.random.default_rng(0)
    X = rng.random((1000, 10)).astype(np.float32)
    y = rng.random((1000, 10)).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# make_dataloaders returns 3-tuple
# ---------------------------------------------------------------------------


def test_make_dataloaders_returns_triple(
    xy: tuple[np.ndarray, np.ndarray],
    data_cfg: DataConfig,
    signal_cfg: SignalConfig,
) -> None:
    from sine_extraction.data.splitter import make_dataloaders

    X, y = xy
    result = make_dataloaders(X, y, data_cfg, signal_cfg)
    assert isinstance(result, tuple)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# train split ≈ 70 %
# ---------------------------------------------------------------------------


def test_train_split_size(
    xy: tuple[np.ndarray, np.ndarray],
    data_cfg: DataConfig,
    signal_cfg: SignalConfig,
) -> None:
    from sine_extraction.data.splitter import make_dataloaders

    X, y = xy
    train_loader, _, _ = make_dataloaders(X, y, data_cfg, signal_cfg)
    train_n = len(train_loader.dataset)  # type: ignore[arg-type]
    expected = int(1000 * 0.70)
    assert abs(train_n - expected) <= 1


# ---------------------------------------------------------------------------
# val split ≈ 15 %
# ---------------------------------------------------------------------------


def test_val_split_size(
    xy: tuple[np.ndarray, np.ndarray],
    data_cfg: DataConfig,
    signal_cfg: SignalConfig,
) -> None:
    from sine_extraction.data.splitter import make_dataloaders

    X, y = xy
    _, val_loader, _ = make_dataloaders(X, y, data_cfg, signal_cfg)
    val_n = len(val_loader.dataset)  # type: ignore[arg-type]
    expected = int(1000 * 0.15)
    assert abs(val_n - expected) <= 1


# ---------------------------------------------------------------------------
# test split ≈ 15 %
# ---------------------------------------------------------------------------


def test_test_split_size(
    xy: tuple[np.ndarray, np.ndarray],
    data_cfg: DataConfig,
    signal_cfg: SignalConfig,
) -> None:
    from sine_extraction.data.splitter import make_dataloaders

    X, y = xy
    _, _, test_loader = make_dataloaders(X, y, data_cfg, signal_cfg)
    test_n = len(test_loader.dataset)  # type: ignore[arg-type]
    expected = int(1000 * 0.15)
    assert abs(test_n - expected) <= 1


# ---------------------------------------------------------------------------
# split sizes sum to total
# ---------------------------------------------------------------------------


def test_split_sizes_sum_to_total(
    xy: tuple[np.ndarray, np.ndarray],
    data_cfg: DataConfig,
    signal_cfg: SignalConfig,
) -> None:
    from sine_extraction.data.splitter import make_dataloaders

    X, y = xy
    train_l, val_l, test_l = make_dataloaders(X, y, data_cfg, signal_cfg)
    total = (
        len(train_l.dataset)  # type: ignore[arg-type]
        + len(val_l.dataset)  # type: ignore[arg-type]
        + len(test_l.dataset)  # type: ignore[arg-type]
    )
    assert total == 1000


# ---------------------------------------------------------------------------
# each result is a DataLoader instance
# ---------------------------------------------------------------------------


def test_loaders_are_dataloaders(
    xy: tuple[np.ndarray, np.ndarray],
    data_cfg: DataConfig,
    signal_cfg: SignalConfig,
) -> None:
    from torch.utils.data import DataLoader

    from sine_extraction.data.splitter import make_dataloaders

    X, y = xy
    for loader in make_dataloaders(X, y, data_cfg, signal_cfg):
        assert isinstance(loader, DataLoader)


# ---------------------------------------------------------------------------
# loaders yield batches of correct shape
# ---------------------------------------------------------------------------


def test_loader_batch_shape(
    xy: tuple[np.ndarray, np.ndarray],
    data_cfg: DataConfig,
    signal_cfg: SignalConfig,
) -> None:
    from sine_extraction.data.splitter import make_dataloaders

    X, y = xy
    train_loader, _, _ = make_dataloaders(X, y, data_cfg, signal_cfg)
    xb, _label, yb = next(iter(train_loader))
    assert xb.shape[1] == 10
    assert yb.shape[1] == 10
    assert xb.shape[0] <= data_cfg.batch_size


# ---------------------------------------------------------------------------
# no sample in more than one split (index-level check via sizes)
# ---------------------------------------------------------------------------


def test_no_overlap_between_splits(
    xy: tuple[np.ndarray, np.ndarray],
    data_cfg: DataConfig,
    signal_cfg: SignalConfig,
) -> None:
    from torch.utils.data import Subset

    from sine_extraction.data.splitter import make_dataloaders

    X, y = xy
    train_l, val_l, test_l = make_dataloaders(X, y, data_cfg, signal_cfg)

    def indices(loader: object) -> set[int]:
        ds = loader.dataset  # type: ignore[union-attr]
        if isinstance(ds, Subset):
            return set(ds.indices)
        return set(range(len(ds)))  # type: ignore[arg-type]

    tr = indices(train_l)
    va = indices(val_l)
    te = indices(test_l)
    assert len(tr & va) == 0
    assert len(tr & te) == 0
    assert len(va & te) == 0
