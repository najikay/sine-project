"""Tests for evaluation/metrics.py — written BEFORE implementation (TDD)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from sine_extraction.data.dataset import SineDataset
from sine_extraction.models.mlp import MLPModel
from sine_extraction.types import MLPConfig


@pytest.fixture()
def window_size() -> int:
    """Window size for all metric tests."""
    return 10


@pytest.fixture()
def perfect_arrays(window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Identical pred and target arrays (perfect predictions)."""
    rng = np.random.default_rng(42)
    arr = rng.random((30, window_size)).astype(np.float32)
    return arr, arr


@pytest.fixture()
def offset_arrays(window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """pred = target + 1.0 (constant offset of 1.0)."""
    rng = np.random.default_rng(42)
    target = rng.random((30, window_size)).astype(np.float32)
    pred = target + 1.0
    return pred, target


@pytest.fixture()
def tiny_model(window_size: int) -> MLPModel:
    """Tiny MLP for evaluate_model tests."""
    cfg = MLPConfig(hidden_sizes=[8], activation="relu")
    return MLPModel(cfg, window_size)


@pytest.fixture()
def tiny_loader(window_size: int) -> DataLoader:
    """Small DataLoader with 20 samples."""
    rng = np.random.default_rng(0)
    X = rng.random((20, window_size)).astype(np.float32)
    y = rng.random((20, window_size)).astype(np.float32)
    freq_lbl = torch.zeros(4)
    freq_lbl[2] = 1.0
    ds = SineDataset(X, y, freq_lbl)
    return DataLoader(ds, batch_size=8)


# ---------------------------------------------------------------------------
# 6.1.2  compute_mse returns float
# ---------------------------------------------------------------------------


def test_compute_mse_returns_float(
    offset_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """compute_mse returns a Python float."""
    from sine_extraction.evaluation.metrics import compute_mse

    pred, target = offset_arrays
    result = compute_mse(pred, target)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 6.1.3  perfect predictions -> MSE = 0.0
# ---------------------------------------------------------------------------


def test_compute_mse_perfect(
    perfect_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """compute_mse returns 0.0 for identical pred and target."""
    from sine_extraction.evaluation.metrics import compute_mse

    pred, target = perfect_arrays
    assert compute_mse(pred, target) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6.1.4  known offset -> known MSE
# ---------------------------------------------------------------------------


def test_compute_mse_known_value(
    offset_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """pred = target + 1.0 -> MSE = 1.0."""
    from sine_extraction.evaluation.metrics import compute_mse

    pred, target = offset_arrays
    assert compute_mse(pred, target) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 6.1.5  compute_mae returns float
# ---------------------------------------------------------------------------


def test_compute_mae_returns_float(
    offset_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """compute_mae returns a Python float."""
    from sine_extraction.evaluation.metrics import compute_mae

    pred, target = offset_arrays
    result = compute_mae(pred, target)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 6.1.6  perfect predictions -> MAE = 0.0
# ---------------------------------------------------------------------------


def test_compute_mae_perfect(
    perfect_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """compute_mae returns 0.0 for identical pred and target."""
    from sine_extraction.evaluation.metrics import compute_mae

    pred, target = perfect_arrays
    assert compute_mae(pred, target) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6.1.7  compute_r2 returns float in range (-inf, 1.0]
# ---------------------------------------------------------------------------


def test_compute_r2_returns_float(
    offset_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """compute_r2 returns a Python float."""
    from sine_extraction.evaluation.metrics import compute_r2

    pred, target = offset_arrays
    result = compute_r2(pred, target)
    assert isinstance(result, float)
    assert result <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 6.1.8  perfect predictions -> R2 = 1.0
# ---------------------------------------------------------------------------


def test_compute_r2_perfect(
    perfect_arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    """compute_r2 returns 1.0 for perfect predictions."""
    from sine_extraction.evaluation.metrics import compute_r2

    pred, target = perfect_arrays
    assert compute_r2(pred, target) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 6.1.9  evaluate_model returns dict with correct keys
# ---------------------------------------------------------------------------


def test_evaluate_model_returns_dict_keys(
    tiny_model: MLPModel,
    tiny_loader: DataLoader,
) -> None:
    """evaluate_model returns dict with keys mse, mae, r2."""
    from sine_extraction.evaluation.metrics import evaluate_model

    result = evaluate_model(tiny_model, tiny_loader, torch.device("cpu"))
    assert "mse" in result
    assert "mae" in result
    assert "r2" in result


# ---------------------------------------------------------------------------
# 6.1.10  evaluate_model with identity-like model returns sensible metrics
# ---------------------------------------------------------------------------


def test_evaluate_model_sensible_metrics(
    tiny_model: MLPModel,
    tiny_loader: DataLoader,
) -> None:
    """evaluate_model returns finite numeric metrics."""
    from sine_extraction.evaluation.metrics import evaluate_model

    result = evaluate_model(tiny_model, tiny_loader, torch.device("cpu"))
    assert isinstance(result["mse"], float)
    assert isinstance(result["mae"], float)
    assert isinstance(result["r2"], float)
    assert result["mse"] >= 0.0
    assert result["mae"] >= 0.0


# ---------------------------------------------------------------------------
# Edge cases for compute_r2
# ---------------------------------------------------------------------------


def test_compute_r2_constant_target_perfect() -> None:
    """R² = 1.0 when target is constant and pred matches (SS_tot=0, SS_res=0)."""
    from sine_extraction.evaluation.metrics import compute_r2

    arr = np.ones((10, 5), dtype=np.float32)
    assert compute_r2(arr, arr) == pytest.approx(1.0, abs=1e-6)


def test_compute_r2_constant_target_wrong_pred() -> None:
    """R² = 0.0 when target is constant but pred differs (SS_tot=0, SS_res>0)."""
    from sine_extraction.evaluation.metrics import compute_r2

    target = np.ones((10, 5), dtype=np.float32)
    pred = np.zeros((10, 5), dtype=np.float32)
    assert compute_r2(pred, target) == pytest.approx(0.0, abs=1e-6)


def test_compute_r2_negative_for_bad_model() -> None:
    """R² is negative when predictions are worse than predicting the mean."""
    from sine_extraction.evaluation.metrics import compute_r2

    rng = np.random.default_rng(7)
    target = rng.random((50,)).astype(np.float32)
    pred = -target  # inverted: maximally wrong
    assert compute_r2(pred, target) < 0.0
