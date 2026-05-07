"""Training loop with early stopping and model checkpointing."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sine_extraction.models.base import BaseModel
from sine_extraction.training.checkpoint_manager import save_checkpoint
from sine_extraction.training.losses import mse_loss
from sine_extraction.types import TrainingConfig


class Trainer:
    """Manages the training loop for any BaseModel subclass.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        config: Training hyperparameters (lr, epochs, patience, dirs).
        device: Torch device to run training on.
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
            if config.use_scheduler
            else None
        )
        self._ckpt_dir = Path(config.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Remove stale checkpoints for this model class before a fresh run
        model_name = type(model).__name__
        for old_pt in self._ckpt_dir.glob(f"{model_name}*.pt"):
            old_pt.unlink()

    # ------------------------------------------------------------------
    # Private epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        """Run one training epoch and return mean loss.

        Returns:
            Mean training loss over all batches.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for x_batch, label_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x_batch, label_batch)
            loss = mse_loss(pred, y_batch)
            loss.backward()
            if self.config.grad_clip_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_max_norm
                )
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _val_epoch(self) -> float:
        """Run one validation epoch and return mean loss.

        Returns:
            Mean validation loss over all batches.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for x_batch, label_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self.model(x_batch, label_batch)
                total_loss += mse_loss(pred, y_batch).item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Public training entry point
    # ------------------------------------------------------------------

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop with early stopping.

        Returns:
            History dict with keys ``train_loss`` and ``val_loss``, each a
            list of per-epoch floats (may be shorter than ``epochs`` when
            early stopping triggers).
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(self.model, self._ckpt_dir, epoch)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break

        return history
