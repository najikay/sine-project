"""End-to-end integration tests: generate → train → evaluate → visualize."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sine_extraction.data.generator import SignalGenerator
from sine_extraction.data.splitter import make_dataloaders
from sine_extraction.evaluation.metrics import evaluate_model
from sine_extraction.models.mlp import MLPModel
from sine_extraction.types import (
    DataConfig,
    MLPConfig,
    SignalConfig,
)
from sine_extraction.visualization.plotter import ComparisonPlotter


@pytest.fixture()
def tiny_signal_cfg() -> SignalConfig:
    """Tiny SignalConfig for fast integration tests."""
    return SignalConfig(
        frequencies=[1.0, 5.0, 10.0, 20.0],
        sample_rate=200,
        window_size=10,
        target_frequency=10.0,
        amplitude=1.0,
        amplitude_jitter_std=0.05,
        phase_jitter_std=0.1,
        noise_std=0.1,
    )


@pytest.fixture()
def tiny_data_cfg() -> DataConfig:
    """Tiny DataConfig with 50 samples for integration tests."""
    return DataConfig(
        num_windows=50,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=8,
        shuffle=False,
    )


@pytest.fixture()
def generated_dataset(
    tiny_signal_cfg: SignalConfig,
    tiny_data_cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a tiny dataset."""
    gen = SignalGenerator(tiny_signal_cfg, seed=42)
    return gen.generate_dataset(tiny_data_cfg.num_windows)


def test_generate_returns_correct_shapes(
    tiny_signal_cfg: SignalConfig,
    tiny_data_cfg: DataConfig,
) -> None:
    """generate_dataset returns (X, y) with shape (N, window_size)."""
    gen = SignalGenerator(tiny_signal_cfg, seed=42)
    X, y = gen.generate_dataset(tiny_data_cfg.num_windows)
    n = tiny_data_cfg.num_windows
    ws = tiny_signal_cfg.window_size
    assert X.shape == (n, ws)
    assert y.shape == (n, ws)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_dataloaders_from_generated_data(
    generated_dataset: tuple[np.ndarray, np.ndarray],
    tiny_data_cfg: DataConfig,
    tiny_signal_cfg: SignalConfig,
) -> None:
    """make_dataloaders produces three DataLoaders summing to total."""
    X, y = generated_dataset
    train_loader, val_loader, test_loader = make_dataloaders(
        X, y, tiny_data_cfg, tiny_signal_cfg
    )
    total = len(X)
    train_n = sum(len(b[0]) for b in train_loader)
    val_n = sum(len(b[0]) for b in val_loader)
    test_n = sum(len(b[0]) for b in test_loader)
    assert train_n + val_n + test_n == total


def test_mlp_forward_on_generated_data(
    generated_dataset: tuple[np.ndarray, np.ndarray],
    tiny_data_cfg: DataConfig,
    tiny_signal_cfg: SignalConfig,
) -> None:
    """MLPModel forward pass on generated data returns correct shape."""
    X, y = generated_dataset
    train_loader, _, _ = make_dataloaders(X, y, tiny_data_cfg, tiny_signal_cfg)
    model = MLPModel(MLPConfig(hidden_sizes=[16], activation="relu"), window_size=10)
    model.eval()
    x_batch, label_batch, _ = next(iter(train_loader))
    with torch.no_grad():
        out = model(x_batch, label_batch)
    assert out.shape == x_batch.shape
    assert not torch.isnan(out).any()


def test_evaluate_model_returns_metrics(
    generated_dataset: tuple[np.ndarray, np.ndarray],
    tiny_data_cfg: DataConfig,
    tiny_signal_cfg: SignalConfig,
) -> None:
    """evaluate_model returns dict with mse, mae, r2 keys."""
    X, y = generated_dataset
    _, _, test_loader = make_dataloaders(X, y, tiny_data_cfg, tiny_signal_cfg)
    model = MLPModel(MLPConfig(hidden_sizes=[16], activation="relu"), window_size=10)
    device = torch.device("cpu")
    metrics = evaluate_model(model, test_loader, device)
    assert "mse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert isinstance(metrics["mse"], float)
    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["r2"], float)


def test_full_pipeline_generate_evaluate(
    tiny_signal_cfg: SignalConfig,
    tiny_data_cfg: DataConfig,
) -> None:
    """Full pipeline: generate → evaluate without training (untrained model)."""
    gen = SignalGenerator(tiny_signal_cfg, seed=0)
    X, y = gen.generate_dataset(tiny_data_cfg.num_windows)
    _, _, test_loader = make_dataloaders(X, y, tiny_data_cfg, tiny_signal_cfg)
    model = MLPModel(MLPConfig(hidden_sizes=[8], activation="relu"), window_size=10)
    metrics = evaluate_model(model, test_loader, torch.device("cpu"))
    assert metrics["mse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert metrics["r2"] <= 1.0


def test_plotter_saves_file_from_generated_data(
    generated_dataset: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    """ComparisonPlotter saves a PNG from generated data."""
    from sine_extraction.types import VisualizationConfig

    X, y = generated_dataset
    vis_cfg = VisualizationConfig(
        plots_dir=str(tmp_path / "plots"),
        interactive=False,
        num_samples_to_plot=1,
    )
    plotter = ComparisonPlotter(vis_cfg)
    fake_pred = X[0].copy()
    plotter.plot(
        noisy=X[0],
        pure=y[0],
        mlp_pred=fake_pred,
        rnn_pred=fake_pred,
        lstm_pred=fake_pred,
        sample_idx=0,
    )
    saved = plotter.save("integration_test.png")
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_reproducibility_same_seed(tiny_signal_cfg: SignalConfig) -> None:
    """Two generators with the same seed produce identical datasets."""
    gen1 = SignalGenerator(tiny_signal_cfg, seed=7)
    gen2 = SignalGenerator(tiny_signal_cfg, seed=7)
    X1, y1 = gen1.generate_dataset(20)
    X2, y2 = gen2.generate_dataset(20)
    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)


def test_reproducibility_different_seeds(tiny_signal_cfg: SignalConfig) -> None:
    """Two generators with different seeds produce different datasets."""
    gen1 = SignalGenerator(tiny_signal_cfg, seed=1)
    gen2 = SignalGenerator(tiny_signal_cfg, seed=2)
    X1, _ = gen1.generate_dataset(20)
    X2, _ = gen2.generate_dataset(20)
    assert not np.allclose(X1, X2)
