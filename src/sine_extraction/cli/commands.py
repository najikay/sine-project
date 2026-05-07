"""Click subcommands: generate, train, all (pipeline entry points)."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch

from sine_extraction.config_loader import load_config
from sine_extraction.data.generator import SignalGenerator
from sine_extraction.data.splitter import make_dataloaders
from sine_extraction.models.lstm import LSTMModel
from sine_extraction.models.mlp import MLPModel
from sine_extraction.models.rnn import RNNModel
from sine_extraction.training.trainer import Trainer

_DATA_DIR_OPT = click.option(
    "--data-dir",
    default="artifacts/data",
    show_default=True,
    help="Directory for X.npy / y.npy",
)


@click.command()
@_DATA_DIR_OPT
@click.pass_context
def cmd_generate(ctx: click.Context, data_dir: str) -> None:
    """Generate synthetic training data and save to DATA_DIR."""
    cfg = load_config(Path(ctx.obj["config"]))
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    gen = SignalGenerator(cfg.signal, seed=int(rng.integers(0, 2**31)))
    X, y = gen.generate_dataset(cfg.data.num_windows)

    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    click.echo(f"Saved {len(X)} windows to {out}")


@click.command()
@_DATA_DIR_OPT
@click.pass_context
def cmd_train(ctx: click.Context, data_dir: str) -> None:
    """Train MLP, RNN, and LSTM models on previously generated data."""
    cfg = load_config(Path(ctx.obj["config"]))
    data_path = Path(data_dir)
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")

    train_loader, val_loader, _ = make_dataloaders(X, y, cfg.data, cfg.signal)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws = cfg.signal.window_size

    model_instances = {
        "mlp": MLPModel(cfg.model.mlp, ws),
        "rnn": RNNModel(cfg.model.rnn, ws),
        "lstm": LSTMModel(cfg.model.lstm, ws),
    }
    for name, model in model_instances.items():
        trainer = Trainer(model, train_loader, val_loader, cfg.training, device)
        history = trainer.train()
        final = history["val_loss"][-1] if history["val_loss"] else float("nan")
        click.echo(
            f"{name}: {len(history['train_loss'])} epochs, val_loss={final:.6f}"
        )
