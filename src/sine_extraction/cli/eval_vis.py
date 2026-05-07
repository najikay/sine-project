"""Click subcommands: evaluate, visualize."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch

from sine_extraction.config_loader import load_config
from sine_extraction.data.splitter import make_dataloaders
from sine_extraction.evaluation.metrics import evaluate_model
from sine_extraction.models.lstm import LSTMModel
from sine_extraction.models.mlp import MLPModel
from sine_extraction.models.rnn import RNNModel
from sine_extraction.visualization.plotter import ComparisonPlotter

_DATA_DIR_OPT = click.option(
    "--data-dir",
    default="artifacts/data",
    show_default=True,
    help="Directory containing X.npy / y.npy",
)


def _load_models(
    cfg: object,
    device: torch.device,
) -> dict[str, object]:
    """Load MLP, RNN, LSTM from latest checkpoint if available."""
    ws = cfg.signal.window_size  # type: ignore[attr-defined]
    ckpt_dir = Path(cfg.training.checkpoint_dir)  # type: ignore[attr-defined]
    models: dict[str, object] = {
        "mlp": MLPModel(cfg.model.mlp, ws),  # type: ignore[attr-defined]
        "rnn": RNNModel(cfg.model.rnn, ws),  # type: ignore[attr-defined]
        "lstm": LSTMModel(cfg.model.lstm, ws),  # type: ignore[attr-defined]
    }
    for _name, model in models.items():
        cls_name = model.__class__.__name__  # type: ignore[union-attr]
        pts = sorted(ckpt_dir.glob(f"{cls_name}*.pt"))
        if pts:
            model.load_state_dict(  # type: ignore[union-attr]
                torch.load(pts[-1], map_location=device)
            )
        model.to(device)  # type: ignore[union-attr]
    return models


@click.command()
@_DATA_DIR_OPT
@click.pass_context
def cmd_evaluate(ctx: click.Context, data_dir: str) -> None:
    """Evaluate trained models and write metrics.json."""
    cfg = load_config(Path(ctx.obj["config"]))
    data_path = Path(data_dir)
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")

    _, _, test_loader = make_dataloaders(X, y, cfg.data, cfg.signal)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _load_models(cfg, device)

    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        metrics = evaluate_model(model, test_loader, device)  # type: ignore[arg-type]
        results[name] = metrics
        click.echo(f"{name}: {metrics}")

    results_dir = Path(cfg.training.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    click.echo(f"Metrics saved to {results_dir / 'metrics.json'}")


@click.command()
@_DATA_DIR_OPT
@click.pass_context
def cmd_visualize(ctx: click.Context, data_dir: str) -> None:
    """Generate and save comparison plots for each sample."""
    cfg = load_config(Path(ctx.obj["config"]))
    data_path = Path(data_dir)
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _load_models(cfg, device)
    for model in models.values():
        model.eval()  # type: ignore[union-attr]

    freq_idx = cfg.signal.frequencies.index(cfg.signal.target_frequency)
    freq_label = torch.zeros(len(cfg.signal.frequencies))
    freq_label[freq_idx] = 1.0

    plotter = ComparisonPlotter(cfg.visualization)
    n = min(cfg.visualization.num_samples_to_plot, len(X))
    for i in range(n):
        x_t = torch.tensor(X[i]).unsqueeze(0).to(device)
        label_t = freq_label.unsqueeze(0).to(device)
        with torch.no_grad():
            preds = {
                k: m(x_t, label_t).cpu().numpy()[0]  # type: ignore[operator]
                for k, m in models.items()
            }
        plotter.plot(
            noisy=X[i],
            pure=y[i],
            mlp_pred=preds["mlp"],
            rnn_pred=preds["rnn"],
            lstm_pred=preds["lstm"],
            sample_idx=i,
        )
        saved = plotter.save(f"comparison_sample_{i}.png")
        click.echo(f"Saved {saved}")
