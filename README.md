# Sine Wave Extraction — Assignment 01

> **Course:** AI Orchestration · SemB 2026 · **Group ID:** `NajAmjad`

Extract a target sine wave from a noisy mixture of multiple sine waves using three neural network architectures: MLP, RNN, and LSTM.

## Results Summary

| Model | MSE    | MAE    | R²     |
|-------|--------|--------|--------|
| MLP   | 0.0582 | 0.1811 | 0.8836 |
| LSTM  | 0.0568 | 0.1742 | **0.8863** |
| RNN   | 0.1828 | 0.3373 | 0.6343 |

## Code Quality

| Metric | Value |
|--------|-------|
| Ruff violations | **0** |
| Test coverage | **86%+** (≥85% required) |
| Max lines per file | **≤150** |
| Tests passing | **138 / 138** |

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
uv sync
```

## Usage

All commands are run via the CLI entry point:

```bash
PYTHONPATH=src uv run python -m sine_extraction --help
```

### Run the full pipeline

```bash
PYTHONPATH=src uv run python -m sine_extraction all
```

### Individual steps

```bash
# 1. Generate training data
PYTHONPATH=src uv run python -m sine_extraction generate --data-dir artifacts/data

# 2. Train all three models
PYTHONPATH=src uv run python -m sine_extraction train --data-dir artifacts/data

# 3. Evaluate trained models
PYTHONPATH=src uv run python -m sine_extraction evaluate --data-dir artifacts/data

# 4. Visualize predictions (6-panel PNG per sample)
PYTHONPATH=src uv run python -m sine_extraction visualize --data-dir artifacts/data
```

### Interactive Dashboard

After running the full pipeline, launch the Streamlit UI:

```bash
PYTHONPATH=src uv run streamlit run src/sine_extraction/app.py
```

The dashboard lets you scrub through any sample index and compare all three model predictions side-by-side with an overlay panel.

## Python API

```python
from sine_extraction import (
    load_config, SignalGenerator, make_dataloaders,
    MLPModel, Trainer, evaluate_model, ComparisonPlotter,
    set_all_seeds, get_device, save_metrics, load_metrics,
)

# Load configuration
config = load_config("config/config.yaml")

# Set reproducible seeds
set_all_seeds(config.seed)

# Generate data
gen = SignalGenerator(config.signal, seed=config.seed)
X, y = gen.generate_dataset(config.data.num_windows)

# Create data loaders
train_loader, val_loader, test_loader = make_dataloaders(
    X, y, config.data, config.signal
)

# Train a model
device = get_device()
model = MLPModel(config.model.mlp, config.signal.window_size)
trainer = Trainer(model, train_loader, val_loader, config.training, device)
history = trainer.train()

# Evaluate
metrics = evaluate_model(model, test_loader, device)
save_metrics(metrics, "artifacts/results/metrics.json")
```

## Project Structure

```
src/sine_extraction/
    app.py              # Streamlit interactive dashboard
    constants.py        # Project-wide constants
    types.py            # Typed dataclasses for all configs
    config_loader.py    # YAML config loader
    seeding.py          # Reproducibility utilities
    device.py           # Device auto-detection
    results.py          # Results persistence (JSON)
    data/               # Signal generation, dataset, splitting
    models/             # MLP, RNN, LSTM architectures
    training/           # Trainer with early stopping + gradient clipping
    evaluation/         # MSE, MAE, R² metrics
    visualization/      # 6-panel comparison plotter (+ overlay panel)
    cli/                # Click CLI subcommands
docs/
    PRD.md              # System-level product requirements
    PRD_MLP.md          # MLP architecture spec
    PRD_RNN.md          # RNN architecture spec
    PRD_LSTM.md         # LSTM architecture spec
    PLAN.md             # Architectural plan
    TODO.md             # Implementation checklist
```

## Testing

```bash
PYTHONPATH=src uv run pytest
```

Coverage report is generated automatically (≥ 85% required).

## Linting

```bash
uv run ruff check src/ tests/
```

## Mathematical Model

The system generates a noisy mixture of four sine waves at frequencies 1, 5, 10, and 20 Hz. Each component has random amplitude jitter and phase offset. White Gaussian noise is added. The task is to extract the clean 10 Hz sine wave.

**Signal model:**
- Input: `x(t) = Σ_f  A_f · sin(2π f t + φ_f) + ε(t)`
- Target: `y(t) = A · sin(2π · 10 · t + φ_10)`

where `A_f ~ N(A, σ_A)`, `φ_f ~ U(0, 2π)`, `ε(t) ~ N(0, σ_n)`.

| Parameter | Value |
|---|---|
| Frequencies | 1, 5, 10, 20 Hz |
| Sample rate | 200 Hz (Nyquist: 100 Hz) |
| Window size | 10 samples |
| Amplitude jitter σ_A | 0.05 |
| Phase jitter | U(0, 2π) per window |
| Noise std σ_n | 0.05 |

## Architecture Comparison

| Model | Input | Inductive Bias | Params (default) | Expected R² |
|---|---|---|---|---|
| MLP | `(B, 14)` flat | None — treats window as flat vector | ~673K | ~0.80–0.90 |
| RNN | `(B, 10, 5)` sequence | Temporal ordering + hidden state | ~394K | ~0.88–0.95 |
| LSTM | `(B, 10, 5)` sequence | Temporal ordering + gated memory | ~788K | ~0.92–0.97 |

Input is `window_size + 4` (= 14) for MLP (flat), and `(window_size, 1+4)` (= `(10, 5)`) per timestep for RNN/LSTM. The 4-dimensional 1-hot label encodes the target frequency for conditioned extraction.

## Gradient Clipping

RNN and LSTM models are susceptible to exploding gradients. The trainer applies `torch.nn.utils.clip_grad_norm_` with `max_norm=1.0` (configurable via `training.grad_clip_max_norm`) after every backward pass, ensuring stable convergence.

## Hyperparameter Sweep

The project ships with `sweep_experiments.py`, an automated grid search over 8 configurations combining:
- **Architecture variants** (shallower vs. deeper)
- **Bidirectionality** (unidirectional vs. bidirectional RNN/LSTM)
- **LR schedule** (flat vs. `ReduceLROnPlateau`)

```bash
PYTHONPATH=src uv run python sweep_experiments.py
```

The sweep trains all three models for each configuration, prints a ranked results table by average MSE, and writes the best configuration back to `config/config.yaml`.
