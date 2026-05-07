# Product Requirements Document (PRD)
## Project: Sine Wave Extraction System
### Version: 1.0.0 | Author: NajAmjad | Date: 2026-05-06

---

## 1. Executive Summary

The Sine Wave Extraction System is a machine-learning pipeline that isolates a single target pure sine wave from a noisy, multi-frequency mixed signal. The system trains three distinct neural-network architectures — a Fully Connected Network (MLP), a Recurrent Neural Network (RNN), and a Long Short-Term Memory network (LSTM) — and presents a visual comparison of the noisy input, the ground-truth pure signal, and each model's prediction.

---

## 2. Problem Statement

Real-world signals frequently contain additive noise, phase perturbations, and amplitude jitter layered across multiple frequency components. Isolating a target frequency component without access to traditional signal-processing filters (e.g., FFT bandpass) using learned models demonstrates the viability of neural networks for adaptive signal decomposition.

---

## 3. Signal Model

### 3.1 Mathematical Definition

The observed mixed signal at time **t** is defined as:

```
y(t) = (A ± σ_A) · sin(2π f t + φ + σ_φ) + ε(t)
```

Where:

| Symbol | Meaning | Value / Range |
|---|---|---|
| `A` | Nominal amplitude of the target frequency | 1.0 (configurable) |
| `σ_A` | Amplitude jitter (Gaussian noise on amplitude) | ~ N(0, 0.05) |
| `f` | Target frequency | One of {1, 5, 10, 20} Hz |
| `φ` | Fixed phase offset | ~ U(0, 2π) per sample sequence |
| `σ_φ` | Phase jitter | ~ N(0, 0.1 rad) |
| `ε(t)` | Additive white Gaussian noise | ~ N(0, 0.1) |

### 3.2 Mixed Signal Construction

The full observed signal is a superposition of all four frequency components plus noise:

```
x(t) = Σ_{f ∈ F} (A_f ± σ_A) · sin(2π f t + φ_f + σ_φ) + ε(t)
```

Where `F = {1, 5, 10, 20}` Hz.

The **target** (label) for each training sample is:

```
y_target(t) = A_target · sin(2π f_target t + φ_target)
```

i.e., the clean, noise-free, jitter-free version of the chosen target component.

### 3.3 Signal Parameters

| Parameter | Value |
|---|---|
| Known frequencies | 1 Hz, 5 Hz, 10 Hz, 20 Hz |
| Sample rate | 200 Hz |
| Context window (input length) | 10 samples |
| Output length | 10 samples (predicts the same window) |
| Nyquist frequency | 100 Hz |

---

## 4. Functional Requirements

### FR-01: Signal Generation
- The system SHALL generate synthetic mixed signals using the mathematical model in §3.
- Amplitude jitter, phase jitter, and additive noise SHALL be independently configurable via `config/`.
- The data generator SHALL produce deterministic outputs given a fixed random seed.
- The system SHALL support generating datasets of arbitrary size (number of windows).

### FR-02: Data Pipeline
- The system SHALL split generated data into train, validation, and test sets.
- Default split: 70% train / 15% validation / 15% test.
- The pipeline SHALL normalize input signals to the range [-1, 1].
- The pipeline SHALL support batching and shuffling via PyTorch DataLoader.

### FR-03: Model Architectures

#### FR-03a: Fully Connected Network (MLP)
- Input: flattened 10-sample noisy window (shape: [batch, 10]).
- Architecture: configurable number of hidden layers and units per layer.
- Activation: ReLU on hidden layers; linear output layer.
- Output: 10-sample predicted clean window (shape: [batch, 10]).

#### FR-03b: Recurrent Neural Network (RNN)
- Input: 10-sample noisy window as a time sequence (shape: [batch, 10, 1]).
- Architecture: one or more vanilla RNN layers followed by a linear projection.
- Output: 10-sample predicted clean window (shape: [batch, 10]).

#### FR-03c: Long Short-Term Memory (LSTM)
- Input: 10-sample noisy window as a time sequence (shape: [batch, 10, 1]).
- Architecture: one or more LSTM layers followed by a linear projection.
- Output: 10-sample predicted clean window (shape: [batch, 10]).

### FR-04: Training
- All models SHALL be trained using MSE loss.
- The system SHALL use the Adam optimizer with a configurable learning rate.
- Training SHALL support early stopping based on validation loss.
- The system SHALL log training and validation loss per epoch.
- Model checkpoints SHALL be saved to `artifacts/checkpoints/`.

### FR-05: Evaluation
- The system SHALL evaluate each model on the held-out test set.
- Reported metrics: MSE, MAE, R² (coefficient of determination).
- Evaluation results SHALL be saved to `artifacts/results/`.

### FR-06: Visualization UI
- The system SHALL produce a multi-panel plot comparing:
  1. The noisy mixed input signal.
  2. The ground-truth pure sine wave.
  3. The MLP prediction.
  4. The RNN prediction.
  5. The LSTM prediction.
- The UI SHALL be rendered using Matplotlib.
- The plot SHALL be saveable to `artifacts/plots/`.
- The system SHALL support an interactive mode (plt.show()) and a headless/save-only mode.

### FR-07: Configuration
- All hyperparameters (learning rate, batch size, epochs, hidden sizes, noise levels, etc.) SHALL reside in `config/config.yaml`.
- Secrets (if any) SHALL reside in `.env` and be accessed via `python-dotenv`.
- No hardcoded numerical values shall appear in source code; all constants SHALL be imported from config or a dedicated `constants.py`.

### FR-08: Entry Point
- The system SHALL expose a single CLI entry point: `uv run python -m sine_extraction`.
- The entry point SHALL accept subcommands: `generate`, `train`, `evaluate`, `visualize`, `all`.

---

## 5. Non-Functional Requirements

### NFR-01: Code Quality
- Ruff linter SHALL report 0 violations.
- All Python files SHALL be ≤ 150 lines (excluding blank lines and comments).
- Type hints SHALL be present on all public functions and class methods.

### NFR-02: Testing
- Test coverage SHALL exceed 85% as measured by `pytest-cov`.
- Tests SHALL be written before implementation (TDD).
- Unit tests SHALL cover: signal generation, dataset splitting, model forward passes, loss computation, metric calculation.

### NFR-03: Package Management
- `uv` SHALL be the sole package manager.
- `pip` SHALL NOT be used under any circumstances.
- The virtual environment SHALL be managed by `uv venv`.

### NFR-04: Project Structure
- All business logic SHALL reside in `src/sine_extraction/`.
- The package SHALL be accessed via a clean public API defined in `src/sine_extraction/__init__.py`.
- Utility, type, and constant modules SHALL be separate files within the package.

### NFR-05: Reproducibility
- All stochastic operations SHALL accept and use a configurable random seed.
- Seeds SHALL be set for Python `random`, `numpy`, and `torch`.

### NFR-06: Performance
- The full pipeline (`generate` + `train` + `evaluate` + `visualize`) SHALL complete in under 10 minutes on a modern laptop CPU.
- Model training SHALL support GPU acceleration if available (via `torch.device` auto-detection).

---

## 6. Constraints

- Language: Python 3.12+
- Package manager: `uv` exclusively
- Deep learning framework: PyTorch
- Plotting: Matplotlib
- Linter: Ruff (0 violations)
- Test framework: pytest + pytest-cov
- Configuration: YAML via PyYAML or OmegaConf
- No file may exceed 150 lines of Python code

---

## 7. Deliverables

| Deliverable | Path |
|---|---|
| Documentation | `docs/` |
| Source package | `src/sine_extraction/` |
| Configuration | `config/config.yaml` |
| Unit tests | `tests/` |
| Generated dataset | `artifacts/data/` |
| Model checkpoints | `artifacts/checkpoints/` |
| Evaluation results | `artifacts/results/` |
| Comparison plots | `artifacts/plots/` |

---

## 8. Acceptance Criteria

1. `uv run pytest --cov=sine_extraction --cov-report=term-missing` reports ≥ 85% coverage.
2. `uv run ruff check src/ tests/` reports 0 violations.
3. No Python source file exceeds 150 lines.
4. All three models train without error and produce predictions.
5. The visualization renders all five signal panels correctly.
6. Running `uv run python -m sine_extraction all` completes the full pipeline end-to-end.
