"""Automated hyperparameter sweep for sine-wave extraction models.

Tests 8 combinations of architecture depth, bidirectionality, and LR
scheduler, then writes the winning configuration back to config/config.yaml.

Usage:
    PYTHONPATH=src uv run python sweep_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running without PYTHONPATH=src set externally.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sine_extraction.config_loader import load_config
from sine_extraction.data.generator import SignalGenerator
from sine_extraction.data.splitter import make_dataloaders
from sine_extraction.evaluation.metrics import evaluate_model
from sine_extraction.models.lstm import LSTMModel
from sine_extraction.models.mlp import MLPModel
from sine_extraction.models.rnn import RNNModel
from sine_extraction.training.trainer import Trainer
from sine_extraction.types import LSTMConfig, MLPConfig, RNNConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

VARIANT_CONFIGS: dict[str, dict] = {
    "A": {
        "mlp_hidden": [1024, 512, 256],
        "rnn_hidden": 128,
        "rnn_layers": 2,
        "lstm_hidden": 128,
        "lstm_layers": 2,
    },
    "B": {
        "mlp_hidden": [512, 1024, 512, 256],
        "rnn_hidden": 256,
        "rnn_layers": 3,
        "lstm_hidden": 256,
        "lstm_layers": 3,
    },
}

EXPERIMENTS: list[dict] = [
    {"name": "A_uni_flat",    "variant": "A", "bidirectional": False, "use_scheduler": False},
    {"name": "A_uni_plateau", "variant": "A", "bidirectional": False, "use_scheduler": True},
    {"name": "A_bi_flat",     "variant": "A", "bidirectional": True,  "use_scheduler": False},
    {"name": "A_bi_plateau",  "variant": "A", "bidirectional": True,  "use_scheduler": True},
    {"name": "B_uni_flat",    "variant": "B", "bidirectional": False, "use_scheduler": False},
    {"name": "B_uni_plateau", "variant": "B", "bidirectional": False, "use_scheduler": True},
    {"name": "B_bi_flat",     "variant": "B", "bidirectional": True,  "use_scheduler": False},
    {"name": "B_bi_plateau",  "variant": "B", "bidirectional": True,  "use_scheduler": True},
]

SWEEP_EPOCHS = 30
SWEEP_PATIENCE = 8
SWEEP_LR = 0.01
CONFIG_PATH = Path("config/config.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_best_checkpoint(model: torch.nn.Module, ckpt_dir: Path, device: torch.device) -> None:
    """Load the highest-epoch (= best val-loss) checkpoint into *model*."""
    pts = sorted(ckpt_dir.glob(f"{model.__class__.__name__}*.pt"))
    if pts:
        model.load_state_dict(torch.load(pts[-1], map_location=device))


def _run_experiment(
    exp: dict,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    window_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Train and evaluate all three models for one experiment config."""
    variant = VARIANT_CONFIGS[exp["variant"]]
    bidir = exp["bidirectional"]
    use_sched = exp["use_scheduler"]

    mlp_cfg = MLPConfig(hidden_sizes=variant["mlp_hidden"], activation="leaky_relu")
    rnn_cfg = RNNConfig(
        hidden_size=variant["rnn_hidden"],
        num_layers=variant["rnn_layers"],
        bidirectional=bidir,
    )
    lstm_cfg = LSTMConfig(
        hidden_size=variant["lstm_hidden"],
        num_layers=variant["lstm_layers"],
        bidirectional=bidir,
    )

    models = {
        "mlp":  MLPModel(mlp_cfg, window_size),
        "rnn":  RNNModel(rnn_cfg, window_size),
        "lstm": LSTMModel(lstm_cfg, window_size),
    }

    results: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        ckpt_dir = Path(f"artifacts/sweep/{exp['name']}/{model_name}")
        train_cfg = TrainingConfig(
            learning_rate=SWEEP_LR,
            epochs=SWEEP_EPOCHS,
            early_stopping_patience=SWEEP_PATIENCE,
            checkpoint_dir=str(ckpt_dir),
            results_dir="artifacts/results",
            use_scheduler=use_sched,
        )
        trainer = Trainer(model, train_loader, val_loader, train_cfg, device)
        trainer.train()

        # Restore best-checkpoint weights before evaluation.
        _load_best_checkpoint(model, ckpt_dir, device)
        results[model_name] = evaluate_model(model, test_loader, device)

    return results


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _print_table(all_results: dict[str, dict[str, dict[str, float]]]) -> None:
    header = (
        f"{'Experiment':<20} | {'MLP MSE':>8} {'MLP R²':>7} |"
        f" {'RNN MSE':>8} {'RNN R²':>7} |"
        f" {'LSTM MSE':>9} {'LSTM R²':>7} | {'Avg MSE':>8}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for exp_name, res in all_results.items():
        avg = (res["mlp"]["mse"] + res["rnn"]["mse"] + res["lstm"]["mse"]) / 3
        print(
            f"{exp_name:<20} | "
            f"{res['mlp']['mse']:>8.4f} {res['mlp']['r2']:>7.4f} | "
            f"{res['rnn']['mse']:>8.4f} {res['rnn']['r2']:>7.4f} | "
            f"{res['lstm']['mse']:>9.4f} {res['lstm']['r2']:>7.4f} | "
            f"{avg:>8.4f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Config update
# ---------------------------------------------------------------------------

def _write_config(winning_exp: dict) -> None:
    """Overwrite config/config.yaml with the winning hyperparameters."""
    variant = VARIANT_CONFIGS[winning_exp["variant"]]
    bidir = winning_exp["bidirectional"]
    use_sched = winning_exp["use_scheduler"]

    # Load existing config to preserve signal/data/visualization sections.
    with CONFIG_PATH.open() as fh:
        raw: dict = yaml.safe_load(fh)

    raw["model"]["mlp"]["hidden_sizes"] = variant["mlp_hidden"]
    raw["model"]["mlp"]["activation"] = "leaky_relu"

    raw["model"]["rnn"]["hidden_size"] = variant["rnn_hidden"]
    raw["model"]["rnn"]["num_layers"] = variant["rnn_layers"]
    raw["model"]["rnn"]["bidirectional"] = bidir

    raw["model"]["lstm"]["hidden_size"] = variant["lstm_hidden"]
    raw["model"]["lstm"]["num_layers"] = variant["lstm_layers"]
    raw["model"]["lstm"]["bidirectional"] = bidir

    raw["training"]["use_scheduler"] = use_sched

    with CONFIG_PATH.open("w") as fh:
        yaml.dump(raw, fh, default_flow_style=False, sort_keys=False)

    print(f"\nconfig/config.yaml updated with winning config: {winning_exp['name']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config(CONFIG_PATH)

    # Reproducible data generation.
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    print("Generating dataset...")
    gen = SignalGenerator(cfg.signal, seed=int(rng.integers(0, 2**31)))
    X, y = gen.generate_dataset(cfg.data.num_windows)

    # Build dataloaders once — shared across all experiments.
    train_loader, val_loader, test_loader = make_dataloaders(
        X, y, cfg.data, cfg.signal
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws = cfg.signal.window_size

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    total = len(EXPERIMENTS)
    for idx, exp in enumerate(EXPERIMENTS, 1):
        bidir_str = "bidir" if exp["bidirectional"] else "unidir"
        sched_str = "plateau" if exp["use_scheduler"] else "flat-lr"
        print(
            f"\n[{idx}/{total}] {exp['name']}  "
            f"(variant={exp['variant']}, {bidir_str}, {sched_str})"
        )
        all_results[exp["name"]] = _run_experiment(
            exp, train_loader, val_loader, test_loader, ws, device
        )
        res = all_results[exp["name"]]
        avg = (res["mlp"]["mse"] + res["rnn"]["mse"] + res["lstm"]["mse"]) / 3
        print(
            f"  MLP MSE={res['mlp']['mse']:.4f} R²={res['mlp']['r2']:.4f} | "
            f"RNN MSE={res['rnn']['mse']:.4f} R²={res['rnn']['r2']:.4f} | "
            f"LSTM MSE={res['lstm']['mse']:.4f} R²={res['lstm']['r2']:.4f} | "
            f"Avg MSE={avg:.4f}"
        )

    # ------------------------------------------------------------------
    # Select winner: lowest average MSE across all three models.
    # ------------------------------------------------------------------
    def avg_mse(exp_name: str) -> float:
        r = all_results[exp_name]
        return (r["mlp"]["mse"] + r["rnn"]["mse"] + r["lstm"]["mse"]) / 3

    winner_name = min(all_results, key=avg_mse)
    winner_exp = next(e for e in EXPERIMENTS if e["name"] == winner_name)
    winner_res = all_results[winner_name]

    # ------------------------------------------------------------------
    # Print results table.
    # ------------------------------------------------------------------
    print("\n\n=== SWEEP RESULTS ===")
    _print_table(all_results)

    print(f"\n*** WINNER: {winner_name} ***")
    print(f"  Variant     : {winner_exp['variant']}")
    print(f"  Bidirectional: {winner_exp['bidirectional']}")
    print(f"  Scheduler   : {'ReduceLROnPlateau' if winner_exp['use_scheduler'] else 'flat LR'}")
    print(f"  MLP  MSE={winner_res['mlp']['mse']:.4f}  R²={winner_res['mlp']['r2']:.4f}")
    print(f"  RNN  MSE={winner_res['rnn']['mse']:.4f}  R²={winner_res['rnn']['r2']:.4f}")
    print(f"  LSTM MSE={winner_res['lstm']['mse']:.4f}  R²={winner_res['lstm']['r2']:.4f}")
    print(f"  Avg  MSE={avg_mse(winner_name):.4f}")

    _write_config(winner_exp)


if __name__ == "__main__":
    main()
