# Architectural Plan
## Project: Sine Wave Extraction System
### Version: 1.0.0 | Date: 2026-05-06

---

## 1. Repository Layout

```
najikay-sine-project/
├── .env                          # Secrets (random seed override, etc.)
├── .gitignore
├── .python-version               # Pinned Python version for uv
├── pyproject.toml                # Project metadata + uv dependencies (v1.0.0)
├── README.md
├── main.py                       # Root entry point — delegates to CLI
├── sweep_experiments.py          # Hyperparameter grid-search tool
│
├── config/
│   └── config.yaml               # All hyperparameters and signal params
│
├── docs/
│   ├── PRD.md                    # System-level product requirements
│   ├── PRD_MLP.md                # Per-algorithm PRD: MLP
│   ├── PRD_RNN.md                # Per-algorithm PRD: Vanilla RNN
│   ├── PRD_LSTM.md               # Per-algorithm PRD: LSTM
│   ├── PLAN.md
│   └── TODO.md
│
├── src/
│   └── sine_extraction/          # Main SDK package
│       ├── __init__.py           # Public API surface (__all__ defined)
│       ├── __main__.py           # Thin shim: calls cli()
│       ├── app.py                # Streamlit interactive dashboard
│       ├── constants.py          # All magic numbers / enums
│       ├── types.py              # Dataclasses, TypedDicts, type aliases
│       ├── config_loader.py      # Reads config/config.yaml + .env
│       ├── seeding.py            # set_all_seeds() for reproducibility
│       ├── device.py             # get_device() auto-detects GPU/CPU
│       ├── results.py            # save_metrics() / load_metrics()
│       │
│       ├── cli/                  # Click CLI sub-package
│       │   ├── __init__.py
│       │   ├── main.py           # @click.group cli, --config option, all cmd
│       │   ├── commands.py       # generate, train subcommands
│       │   └── eval_vis.py       # evaluate, visualize subcommands
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── generator.py      # SignalGenerator class
│       │   ├── dataset.py        # SineDataset (torch.utils.data.Dataset)
│       │   └── splitter.py       # train/val/test split + 1-hot label
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py           # Abstract BaseModel (nn.Module subclass)
│       │   ├── mlp.py            # MLPModel
│       │   ├── rnn.py            # RNNModel (bidirectional vanilla RNN)
│       │   └── lstm.py           # LSTMModel (bidirectional LSTM)
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py        # Trainer: train loop, early stopping, grad clip
│       │   └── losses.py         # mse_loss wrapper
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py        # MSE, MAE, R², evaluate_model()
│       │
│       └── visualization/
│           ├── __init__.py
│           └── plotter.py        # ComparisonPlotter: 6-panel figure
│
├── tests/
│   ├── conftest.py               # Shared fixtures (config, tiny dataset, etc.)
│   ├── test_config_loader.py
│   ├── test_constants.py
│   ├── test_generator.py
│   ├── test_dataset.py
│   ├── test_splitter.py
│   ├── test_mlp.py
│   ├── test_rnn.py
│   ├── test_lstm.py
│   ├── test_trainer.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_visualization.py
│   ├── test_seed.py
│   ├── test_device.py
│   ├── test_persistence.py
│   └── test_integration.py
│
└── artifacts/                    # gitignored generated outputs
    ├── data/                     # X.npy, y.npy
    ├── checkpoints/              # Model .pt files
    ├── results/                  # metrics.json, history JSONs
    └── plots/                    # comparison_sample_N.png
```

---

## 2. Public API (`src/sine_extraction/__init__.py`)

The package exposes a clean, flat API:

```python
from sine_extraction import (
    # Config
    load_config,
    AppConfig,
    # Data
    SignalGenerator,
    SineDataset,
    make_dataloaders,
    # Models
    MLPModel,
    RNNModel,
    LSTMModel,
    # Training
    Trainer,
    # Evaluation
    evaluate_model,
    # Visualization
    ComparisonPlotter,
)
```

All internal sub-module imports are hidden behind this surface. Callers never import from `sine_extraction.data.generator` directly.

---

## 3. Configuration Schema (`config/config.yaml`)

```yaml
seed: 42

signal:
  frequencies: [1, 5, 10, 20]      # Hz
  sample_rate: 200                  # Hz
  window_size: 10                   # samples
  target_frequency: 10              # Hz — which freq to extract
  amplitude: 1.0
  amplitude_jitter_std: 0.05
  phase_jitter_std: 0.1
  noise_std: 0.1

data:
  num_windows: 10000
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  batch_size: 64
  shuffle: true

model:
  mlp:
    hidden_sizes: [64, 128, 64]
    activation: relu
  rnn:
    hidden_size: 64
    num_layers: 2
  lstm:
    hidden_size: 64
    num_layers: 2

training:
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  checkpoint_dir: artifacts/checkpoints
  results_dir: artifacts/results

visualization:
  plots_dir: artifacts/plots
  interactive: false
  num_samples_to_plot: 3
```

---

## 4. Class Designs

### 4.1 `AppConfig` (types.py)

```python
@dataclass
class SignalConfig:
    frequencies: list[float]
    sample_rate: int
    window_size: int
    target_frequency: float
    amplitude: float
    amplitude_jitter_std: float
    phase_jitter_std: float
    noise_std: float

@dataclass
class DataConfig:
    num_windows: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    batch_size: int
    shuffle: bool

@dataclass
class MLPConfig:
    hidden_sizes: list[int]
    activation: str

@dataclass
class RNNConfig:
    hidden_size: int
    num_layers: int

@dataclass
class LSTMConfig:
    hidden_size: int
    num_layers: int

@dataclass
class ModelConfig:
    mlp: MLPConfig
    rnn: RNNConfig
    lstm: LSTMConfig

@dataclass
class TrainingConfig:
    learning_rate: float
    epochs: int
    early_stopping_patience: int
    checkpoint_dir: str
    results_dir: str

@dataclass
class VisualizationConfig:
    plots_dir: str
    interactive: bool
    num_samples_to_plot: int

@dataclass
class AppConfig:
    seed: int
    signal: SignalConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    visualization: VisualizationConfig
```

### 4.2 `SignalGenerator` (data/generator.py)

```python
class SignalGenerator:
    def __init__(self, config: SignalConfig, seed: int) -> None: ...

    def generate_pure_sine(
        self, frequency: float, num_windows: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (time_axis, clean_signal) for a single frequency."""

    def generate_mixed_window(self, start_sample: int) -> np.ndarray:
        """Returns one 10-sample noisy mixed window."""

    def generate_dataset(
        self, num_windows: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (X, y) where X is mixed windows, y is clean target windows."""
```

### 4.3 `SineDataset` (data/dataset.py)

```python
class SineDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]: ...
```

### 4.4 `BaseModel` (models/base.py)

```python
class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    def count_parameters(self) -> int: ...
    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path, config: Any) -> "BaseModel": ...
```

### 4.5 `MLPModel` (models/mlp.py)

```python
class MLPModel(BaseModel):
    def __init__(self, config: MLPConfig, window_size: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, window_size]
        # output: [batch, window_size]
```

### 4.6 `RNNModel` (models/rnn.py)

```python
class RNNModel(BaseModel):
    def __init__(self, config: RNNConfig, window_size: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, window_size, 1]  (unsqueeze inside forward)
        # output: [batch, window_size]
```

### 4.7 `LSTMModel` (models/lstm.py)

```python
class LSTMModel(BaseModel):
    def __init__(self, config: LSTMConfig, window_size: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, window_size, 1]
        # output: [batch, window_size]
```

### 4.8 `Trainer` (training/trainer.py)

```python
class Trainer:
    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
    ) -> None: ...

    def train(self) -> dict[str, list[float]]:
        """Returns history: {'train_loss': [...], 'val_loss': [...]}."""

    def _train_epoch(self) -> float: ...
    def _val_epoch(self) -> float: ...
    def _save_checkpoint(self, epoch: int, val_loss: float) -> None: ...
```

### 4.9 `evaluate_model` (evaluation/metrics.py)

```python
def evaluate_model(
    model: BaseModel,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Returns {'mse': ..., 'mae': ..., 'r2': ...}."""
```

### 4.10 `ComparisonPlotter` (visualization/plotter.py)

```python
class ComparisonPlotter:
    def __init__(self, config: VisualizationConfig) -> None: ...

    def plot(
        self,
        noisy: np.ndarray,
        pure: np.ndarray,
        mlp_pred: np.ndarray,
        rnn_pred: np.ndarray,
        lstm_pred: np.ndarray,
        sample_idx: int = 0,
    ) -> None:
        """Renders a 5-panel comparison figure."""

    def save(self, filename: str) -> Path: ...
```

---

## 5. CLI Design

The CLI is split across three files in `src/sine_extraction/cli/`:

| File | Contents |
|---|---|
| `cli/main.py` | `@click.group()` with `--config` option and `all` subcommand |
| `cli/commands.py` | `generate` and `train` subcommands |
| `cli/eval_vis.py` | `evaluate` and `visualize` subcommands |

```
PYTHONPATH=src uv run python -m sine_extraction <subcommand> [--config CONFIG_PATH]

Subcommands:
  generate    Generate and save synthetic dataset to artifacts/data/
  train       Train MLP, RNN, and LSTM models; save checkpoints
  evaluate    Load best checkpoints, compute MSE/MAE/R², save metrics.json
  visualize   Generate 6-panel comparison plots (PNG) for N samples
  all         Run the full pipeline: generate → train → evaluate → visualize
```

Interactive dashboard (requires artifacts to exist):

```
PYTHONPATH=src uv run streamlit run src/sine_extraction/app.py
```

---

## 6. Data Flow

```
config.yaml
    │
    ▼
load_config()  ──►  AppConfig
    │
    ▼
SignalGenerator.generate_dataset(num_windows)
    │  X: [N, 10]  (noisy mixed)
    │  y: [N, 10]  (clean target)
    ▼
make_dataloaders()  ──►  train_loader, val_loader, test_loader
    │
    ├──►  Trainer(MLPModel)  ──►  mlp_checkpoint.pt
    ├──►  Trainer(RNNModel)  ──►  rnn_checkpoint.pt
    └──►  Trainer(LSTMModel) ──►  lstm_checkpoint.pt
              │
              ▼
        evaluate_model()  ──►  results JSON
              │
              ▼
        ComparisonPlotter.plot()  ──►  comparison.png
```

---

## 7. Dependency Graph (pyproject.toml)

```toml
[project]
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "click>=8.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
]
```

---

## 8. Testing Strategy

| Test file | What it covers |
|---|---|
| `test_config_loader.py` | YAML parsing, .env override, missing key errors |
| `test_constants.py` | All constants accessible and correct types |
| `test_generator.py` | Signal math, shape, determinism under seed |
| `test_dataset.py` | `__len__`, `__getitem__`, tensor dtype |
| `test_splitter.py` | Exact split ratios, no data leakage |
| `test_mlp.py` | Forward pass shape, parameter count, save/load |
| `test_rnn.py` | Forward pass shape, hidden state handling |
| `test_lstm.py` | Forward pass shape, cell/hidden state handling |
| `test_trainer.py` | Loss decreases, checkpoint saved, early stopping triggers |
| `test_losses.py` | MSE values against known inputs |
| `test_metrics.py` | MAE, R² correctness against known inputs |
| `test_plotter.py` | Figure created, panels correct count, save path exists |

---

## 9. Linting & Formatting Rules (`pyproject.toml` ruff section)

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "ANN", "B", "SIM"]
ignore = []
```

---

## 10. Modularity Enforcement

Every Python file that risks exceeding 150 lines MUST be split:

| File | Strategy |
|---|---|
| `trainer.py` | Gradient-step logic kept inline; all files ≤ 150 lines ✓ |
| `generator.py` | Pure-tone and mix logic combined; ≤ 150 lines ✓ |
| `plotter.py` | 6-panel figure (added overlay panel); ≤ 150 lines ✓ |
| `cli/` | Already split into `main.py`, `commands.py`, `eval_vis.py` ✓ |
