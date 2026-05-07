"""Tests for training/trainer.py — written BEFORE implementation (TDD)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from sine_extraction.data.dataset import SineDataset
from sine_extraction.models.mlp import MLPModel
from sine_extraction.types import MLPConfig, TrainingConfig


@pytest.fixture()
def window_size() -> int:
    """Window size used throughout trainer tests."""
    return 10


@pytest.fixture()
def tiny_mlp(window_size: int) -> MLPModel:
    """Small MLP for fast training in tests."""
    cfg = MLPConfig(hidden_sizes=[16], activation="relu")
    return MLPModel(cfg, window_size)


@pytest.fixture()
def tiny_loaders(window_size: int) -> tuple[DataLoader, DataLoader]:
    """Train and val DataLoaders with 20 samples each."""
    rng = np.random.default_rng(0)
    X = rng.random((20, window_size)).astype(np.float32)
    y = rng.random((20, window_size)).astype(np.float32)
    freq_lbl = torch.zeros(4)
    freq_lbl[2] = 1.0
    train_ds = SineDataset(X, y, freq_lbl)
    val_ds = SineDataset(X, y, freq_lbl)
    train_loader = DataLoader(train_ds, batch_size=8)
    val_loader = DataLoader(val_ds, batch_size=8)
    return train_loader, val_loader


@pytest.fixture()
def train_cfg(tmp_path: Path) -> TrainingConfig:
    """Minimal TrainingConfig for fast tests."""
    return TrainingConfig(
        learning_rate=0.01,
        epochs=3,
        early_stopping_patience=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
    )


# ---------------------------------------------------------------------------
# 5.3.4  instantiation
# ---------------------------------------------------------------------------


def test_trainer_instantiates(
    tiny_mlp: MLPModel,
    tiny_loaders: tuple[DataLoader, DataLoader],
    train_cfg: TrainingConfig,
) -> None:
    """Trainer(model, train, val, config, device) instantiates without error."""
    from sine_extraction.training.trainer import Trainer

    train_loader, val_loader = tiny_loaders
    device = torch.device("cpu")
    trainer = Trainer(tiny_mlp, train_loader, val_loader, train_cfg, device)
    assert trainer is not None


# ---------------------------------------------------------------------------
# 5.3.5  train() returns dict with correct keys
# ---------------------------------------------------------------------------


def test_trainer_returns_history(
    tiny_mlp: MLPModel,
    tiny_loaders: tuple[DataLoader, DataLoader],
    train_cfg: TrainingConfig,
) -> None:
    """train() returns dict with keys train_loss and val_loss."""
    from sine_extraction.training.trainer import Trainer

    train_loader, val_loader = tiny_loaders
    device = torch.device("cpu")
    trainer = Trainer(tiny_mlp, train_loader, val_loader, train_cfg, device)
    history = trainer.train()
    assert "train_loss" in history
    assert "val_loss" in history


# ---------------------------------------------------------------------------
# 5.3.6  train_loss list length <= epochs
# ---------------------------------------------------------------------------


def test_trainer_history_length(
    tiny_mlp: MLPModel,
    tiny_loaders: tuple[DataLoader, DataLoader],
    train_cfg: TrainingConfig,
) -> None:
    """train_loss list has length <= epochs."""
    from sine_extraction.training.trainer import Trainer

    train_loader, val_loader = tiny_loaders
    device = torch.device("cpu")
    trainer = Trainer(tiny_mlp, train_loader, val_loader, train_cfg, device)
    history = trainer.train()
    assert len(history["train_loss"]) <= train_cfg.epochs


# ---------------------------------------------------------------------------
# 5.3.7  final train_loss value is a float
# ---------------------------------------------------------------------------


def test_trainer_loss_is_float(
    tiny_mlp: MLPModel,
    tiny_loaders: tuple[DataLoader, DataLoader],
    train_cfg: TrainingConfig,
) -> None:
    """Final train_loss value is a Python float."""
    from sine_extraction.training.trainer import Trainer

    train_loader, val_loader = tiny_loaders
    device = torch.device("cpu")
    trainer = Trainer(tiny_mlp, train_loader, val_loader, train_cfg, device)
    history = trainer.train()
    assert isinstance(history["train_loss"][-1], float)


# ---------------------------------------------------------------------------
# 5.3.8  checkpoint file is created after training
# ---------------------------------------------------------------------------


def test_trainer_creates_checkpoint(
    tiny_mlp: MLPModel,
    tiny_loaders: tuple[DataLoader, DataLoader],
    train_cfg: TrainingConfig,
) -> None:
    """A checkpoint .pt file is created in checkpoint_dir after training."""
    from sine_extraction.training.trainer import Trainer

    train_loader, val_loader = tiny_loaders
    device = torch.device("cpu")
    trainer = Trainer(tiny_mlp, train_loader, val_loader, train_cfg, device)
    trainer.train()
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    pt_files = list(ckpt_dir.glob("*.pt"))
    assert len(pt_files) >= 1


# ---------------------------------------------------------------------------
# 5.3.9  early stopping halts training
# ---------------------------------------------------------------------------


def test_trainer_early_stopping(tmp_path: Path, window_size: int) -> None:
    """Early stopping halts training before all epochs when loss stagnates."""
    from sine_extraction.training.trainer import Trainer

    cfg = MLPConfig(hidden_sizes=[8], activation="relu")
    model = MLPModel(cfg, window_size)

    # Patience=1 means stop after 1 non-improving epoch
    train_cfg = TrainingConfig(
        learning_rate=0.0,  # zero lr → loss constant → triggers early stop
        epochs=10,
        early_stopping_patience=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        results_dir=str(tmp_path / "results"),
    )
    rng = np.random.default_rng(1)
    X = rng.random((16, window_size)).astype(np.float32)
    y = rng.random((16, window_size)).astype(np.float32)
    freq_lbl = torch.zeros(4)
    freq_lbl[2] = 1.0
    ds = SineDataset(X, y, freq_lbl)
    loader = DataLoader(ds, batch_size=8)
    trainer = Trainer(model, loader, loader, train_cfg, torch.device("cpu"))
    history = trainer.train()
    # Should stop before completing all 10 epochs
    assert len(history["train_loss"]) < 10


# ---------------------------------------------------------------------------
# 5.3.10  training on tiny data lowers loss
# ---------------------------------------------------------------------------


def test_trainer_loss_decreases(
    tiny_loaders: tuple[DataLoader, DataLoader],
    tmp_path: Path,
    window_size: int,
) -> None:
    """Training for several epochs should reduce training loss."""
    from sine_extraction.training.trainer import Trainer

    cfg = MLPConfig(hidden_sizes=[32, 32], activation="relu")
    model = MLPModel(cfg, window_size)
    train_cfg = TrainingConfig(
        learning_rate=0.01,
        epochs=20,
        early_stopping_patience=20,
        checkpoint_dir=str(tmp_path / "ckpts"),
        results_dir=str(tmp_path / "results"),
    )
    train_loader, val_loader = tiny_loaders
    trainer = Trainer(model, train_loader, val_loader, train_cfg, torch.device("cpu"))
    history = trainer.train()
    assert history["train_loss"][-1] < history["train_loss"][0]


# ---------------------------------------------------------------------------
# Gradient clipping does not break training
# ---------------------------------------------------------------------------


def test_trainer_grad_clip_does_not_crash(tmp_path: Path, window_size: int) -> None:
    """Training with grad_clip_max_norm=0.01 (very tight) completes without error."""
    from sine_extraction.training.trainer import Trainer

    cfg = MLPConfig(hidden_sizes=[16], activation="relu")
    model = MLPModel(cfg, window_size)
    train_cfg = TrainingConfig(
        learning_rate=0.01,
        epochs=3,
        early_stopping_patience=5,
        checkpoint_dir=str(tmp_path / "ckpts"),
        results_dir=str(tmp_path / "results"),
        grad_clip_max_norm=0.01,
    )
    rng = np.random.default_rng(42)
    X = rng.random((16, window_size)).astype(np.float32)
    y = rng.random((16, window_size)).astype(np.float32)
    freq_lbl = torch.zeros(4)
    freq_lbl[2] = 1.0
    ds = SineDataset(X, y, freq_lbl)
    loader = DataLoader(ds, batch_size=8)
    trainer = Trainer(model, loader, loader, train_cfg, torch.device("cpu"))
    history = trainer.train()
    assert len(history["train_loss"]) == 3


# ---------------------------------------------------------------------------
# checkpoint_manager saves file correctly
# ---------------------------------------------------------------------------


def test_checkpoint_manager_saves_file(tmp_path: Path, window_size: int) -> None:
    """save_checkpoint writes a .pt file at the expected path."""
    from sine_extraction.models.mlp import MLPModel
    from sine_extraction.training.checkpoint_manager import save_checkpoint
    from sine_extraction.types import MLPConfig

    cfg = MLPConfig(hidden_sizes=[8], activation="relu")
    model = MLPModel(cfg, window_size)
    path = save_checkpoint(model, tmp_path, epoch=1)
    assert path.exists()
    assert path.suffix == ".pt"
