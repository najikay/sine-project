"""Regression metrics and model evaluation utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from sine_extraction.models.base import BaseModel


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute mean squared error between pred and target.

    Args:
        pred: Predicted values array of any shape.
        target: Ground-truth array matching pred's shape.

    Returns:
        MSE as a Python float.
    """
    return float(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute mean absolute error between pred and target.

    Args:
        pred: Predicted values array of any shape.
        target: Ground-truth array matching pred's shape.

    Returns:
        MAE as a Python float.
    """
    return float(np.mean(np.abs(pred - target)))


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute the coefficient of determination (R²).

    Args:
        pred: Predicted values array of any shape.
        target: Ground-truth array matching pred's shape.

    Returns:
        R² as a Python float (≤ 1.0; can be negative for bad models).
    """
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def evaluate_model(
    model: BaseModel,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run inference on test_loader and compute regression metrics.

    Args:
        model: Trained model (any BaseModel subclass).
        test_loader: DataLoader over the test split.
        device: Torch device for inference.

    Returns:
        Dict with keys ``mse``, ``mae``, and ``r2``, each a Python float.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, label_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)
            pred = model(x_batch, label_batch).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return {
        "mse": compute_mse(preds, targets),
        "mae": compute_mae(preds, targets),
        "r2": compute_r2(preds, targets),
    }
