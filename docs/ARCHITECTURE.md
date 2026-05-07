# Architecture Overview

## Problem Statement

Given a mixed signal composed of sine waves at frequencies {1, 5, 10, 20} Hz plus Gaussian
noise, extract the target 10 Hz component. Each window of 10 time-samples is one data point.

## Signal Generation Pipeline

```
SineGenerator
  └─ Superpose sine waves (1, 5, 10, 20 Hz) with per-wave amplitude jitter + phase jitter
  └─ Add Gaussian noise (σ = 0.05)
  └─ Slide a window of size 10 across the signal
  └─ Label each window with a one-hot frequency vector  →  SineDataset
```

## Data Split

| Split | Ratio | Size (of 10,000 windows) |
|-------|-------|--------------------------|
| Train | 70%   | 7,000                    |
| Val   | 15%   | 1,500                    |
| Test  | 15%   | 1,500                    |

## Model Architectures

All three models receive `(x, label)` where `x ∈ ℝ^{W}` is the noisy window and
`label ∈ ℝ^{F}` is the one-hot frequency indicator. They output `ŷ ∈ ℝ^{W}`, the
reconstructed target-frequency window.

### MLP

```
x  ──► Flatten ──► [1024 → 512 → 256, LeakyReLU] ──► Linear(256, W) ──► ŷ
label ──► concat before first hidden layer
```

- Parameters: ~1.3M
- Best for: fast training, strong baseline on stationary signals

### RNN (Vanilla)

```
x  ──► reshape (W, 1) ──► Bidirectional GRU (hidden=256, layers=2)
label ──► concat to last hidden state ──► Linear ──► ŷ
```

- Parameters: ~700K
- Weakness: gradient vanishing limits long-range memory

### LSTM

```
x  ──► reshape (W, 1) ──► Bidirectional LSTM (hidden=128, layers=2)
label ──► concat to last hidden state ──► Linear ──► ŷ
```

- Parameters: ~900K
- Gating mechanism prevents gradient vanishing, outperforms RNN

## Training

| Setting               | Value          |
|-----------------------|----------------|
| Loss                  | MSE            |
| Optimizer             | Adam           |
| Learning rate         | 0.001          |
| LR schedule           | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping        | patience=15    |
| Gradient clipping     | max_norm=1.0   |
| Epochs (max)          | 50             |
| Batch size            | 64             |
| Seed                  | 42             |

## Module Map

```
src/sine_extraction/
├── config_loader.py        # YAML → AppConfig dataclasses
├── constants.py            # Shared string constants
├── device.py               # CPU/GPU selection
├── seeding.py              # Reproducibility (numpy + torch)
├── types.py                # All dataclass types
├── data/
│   ├── generator.py        # Signal synthesis
│   ├── dataset.py          # SineDataset (torch Dataset)
│   └── splitter.py         # Train/val/test split
├── models/
│   ├── base.py             # BaseModel (save/load interface)
│   ├── mlp.py              # MLPModel
│   ├── rnn.py              # RNNModel
│   └── lstm.py             # LSTMModel
├── training/
│   ├── trainer.py          # Training loop + early stopping
│   ├── checkpoint_manager.py # Checkpoint persistence
│   └── losses.py           # MSE loss wrapper
├── evaluation/
│   └── metrics.py          # MSE, MAE, R² + evaluate_model()
├── visualization/
│   └── plotter.py          # 6-panel comparison plots
├── cli/
│   ├── main.py             # CLI entry point
│   └── commands.py         # train / eval / all subcommands
└── app.py                  # Streamlit dashboard
```
