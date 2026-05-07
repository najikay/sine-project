"""Split a dataset into train/val/test DataLoaders."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from sine_extraction.data.dataset import SineDataset
from sine_extraction.types import DataConfig, SignalConfig


def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    config: DataConfig,
    signal: SignalConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders from raw arrays.

    The dataset is split according to ``config.train_ratio``,
    ``config.val_ratio``, and ``config.test_ratio``.  Any remainder sample
    from rounding is appended to the training split.

    Args:
        X: Input array of shape ``(N, window_size)``.
        y: Target array of shape ``(N, window_size)``.
        config: Data configuration containing split ratios, batch size, etc.
        signal: Signal configuration used to build the 1-hot frequency label.

    Returns:
        Triple of ``(train_loader, val_loader, test_loader)``.
    """
    freq_idx = signal.frequencies.index(signal.target_frequency)
    freq_label = torch.zeros(len(signal.frequencies))
    freq_label[freq_idx] = 1.0

    dataset = SineDataset(X, y, freq_label)
    n = len(dataset)

    n_val = int(n * config.val_ratio)
    n_test = int(n * config.test_ratio)
    n_train = n - n_val - n_test  # remainder goes to train

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader
