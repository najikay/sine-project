"""Checkpoint saving utilities for trained models."""

from __future__ import annotations

from pathlib import Path

from sine_extraction.constants import CHECKPOINT_FILENAME_TEMPLATE
from sine_extraction.models.base import BaseModel


def save_checkpoint(model: BaseModel, ckpt_dir: Path, epoch: int) -> Path:
    """Save model weights to a checkpoint file.

    Args:
        model: The model whose weights to persist.
        ckpt_dir: Directory where the checkpoint will be written.
        epoch: Current epoch number (1-indexed), embedded in the filename.

    Returns:
        Path to the saved checkpoint file.
    """
    model_name = type(model).__name__
    filename = CHECKPOINT_FILENAME_TEMPLATE.format(
        model_name=model_name, epoch=epoch
    )
    path = ckpt_dir / filename
    model.save(path)
    return path
