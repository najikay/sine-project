"""Shared pytest fixtures for the sine_extraction test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from sine_extraction.data.splitter import make_dataloaders
from sine_extraction.types import (
    AppConfig,
    DataConfig,
    LSTMConfig,
    MLPConfig,
    ModelConfig,
    RNNConfig,
    SignalConfig,
    TrainingConfig,
    VisualizationConfig,
)


@pytest.fixture()
def default_seed() -> int:
    """Return the canonical test seed."""
    return 42


@pytest.fixture()
def default_signal_config() -> SignalConfig:
    """Return a standard SignalConfig for tests."""
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
def default_app_config(tmp_path: Path) -> AppConfig:
    """Return a minimal AppConfig pointing to tmp_path for artifacts."""
    return AppConfig(
        seed=42,
        signal=SignalConfig(
            frequencies=[1.0, 5.0, 10.0, 20.0],
            sample_rate=200,
            window_size=10,
            target_frequency=10.0,
            amplitude=1.0,
            amplitude_jitter_std=0.05,
            phase_jitter_std=0.1,
            noise_std=0.1,
        ),
        data=DataConfig(
            num_windows=100,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=16,
            shuffle=False,
        ),
        model=ModelConfig(
            mlp=MLPConfig(hidden_sizes=[16], activation="relu"),
            rnn=RNNConfig(hidden_size=16, num_layers=1),
            lstm=LSTMConfig(hidden_size=16, num_layers=1),
        ),
        training=TrainingConfig(
            learning_rate=0.01,
            epochs=2,
            early_stopping_patience=2,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            results_dir=str(tmp_path / "results"),
        ),
        visualization=VisualizationConfig(
            plots_dir=str(tmp_path / "plots"),
            interactive=False,
            num_samples_to_plot=1,
        ),
    )


@pytest.fixture()
def tiny_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays of 100 samples with window_size=10."""
    rng = np.random.default_rng(0)
    X = rng.random((100, 10)).astype(np.float32)
    y = rng.random((100, 10)).astype(np.float32)
    return X, y


@pytest.fixture()
def tiny_loaders(
    tiny_dataset: tuple[np.ndarray, np.ndarray],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders from tiny_dataset."""
    X, y = tiny_dataset
    cfg = DataConfig(
        num_windows=100,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=16,
        shuffle=False,
    )
    return make_dataloaders(X, y, cfg)


@pytest.fixture()
def cpu_device() -> torch.device:
    """Return a CPU torch.device."""
    return torch.device("cpu")


@pytest.fixture()
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for checkpoints."""
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    return ckpt
