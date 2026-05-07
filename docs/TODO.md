# Master TODO Checklist
## Project: Sine Wave Extraction System
### Version: 1.0.0 | Date: 2026-05-06

> Granular micro-step checklist. Check off each item as completed.
> Sections correspond to implementation phases. TDD: tests come before implementation.

---

## PHASE 0 — Project Scaffold & Tooling

### 0.1 Repository Initialization
- [x] 0.1.1 Confirm `.python-version` pins Python 3.12
- [x] 0.1.2 Confirm `pyproject.toml` has `requires-python = ">=3.12"`
- [x] 0.1.3 Run `uv venv` to create `.venv`
- [x] 0.1.4 Confirm `.venv` is listed in `.gitignore`
- [x] 0.1.5 Add `artifacts/` to `.gitignore`
- [x] 0.1.6 Add `.env` to `.gitignore`
- [x] 0.1.7 Create empty `.env` file with comment header
- [x] 0.1.8 Confirm `uv run python --version` returns Python 3.12.x

### 0.2 Dependency Installation
- [x] 0.2.1 Add `torch>=2.2` via `uv add torch`
- [x] 0.2.2 Add `numpy>=1.26` via `uv add numpy`
- [x] 0.2.3 Add `matplotlib>=3.8` via `uv add matplotlib`
- [x] 0.2.4 Add `pyyaml>=6.0` via `uv add pyyaml`
- [x] 0.2.5 Add `python-dotenv>=1.0` via `uv add python-dotenv`
- [x] 0.2.6 Add `click>=8.1` via `uv add click`
- [x] 0.2.7 Add `pytest>=8.0` as dev dep: `uv add --dev pytest`
- [x] 0.2.8 Add `pytest-cov>=5.0` as dev dep: `uv add --dev pytest-cov`
- [x] 0.2.9 Add `ruff>=0.4` as dev dep: `uv add --dev ruff`
- [x] 0.2.10 Verify `uv.lock` was updated
- [x] 0.2.11 Verify all imports work: `uv run python -c "import torch, numpy, matplotlib, yaml, dotenv, click"`

### 0.3 Ruff Configuration
- [x] 0.3.1 Add `[tool.ruff]` section to `pyproject.toml`
- [x] 0.3.2 Set `line-length = 88`
- [x] 0.3.3 Set `target-version = "py312"`
- [x] 0.3.4 Add `[tool.ruff.lint]` section
- [x] 0.3.5 Set `select = ["E", "F", "I", "N", "UP", "ANN", "B", "SIM"]`
- [x] 0.3.6 Run `uv run ruff check .` — confirm 0 violations on empty scaffold

### 0.4 Pytest Configuration
- [x] 0.4.1 Add `[tool.pytest.ini_options]` section to `pyproject.toml`
- [x] 0.4.2 Set `testpaths = ["tests"]`
- [x] 0.4.3 Set `addopts = "--cov=sine_extraction --cov-report=term-missing"`
- [x] 0.4.4 Confirm `uv run pytest --collect-only` runs without errors (no tests yet)

### 0.5 Package Structure Creation
- [x] 0.5.1 Create `src/` directory
- [x] 0.5.2 Create `src/sine_extraction/` directory
- [x] 0.5.3 Create empty `src/sine_extraction/__init__.py`
- [x] 0.5.4 Create empty `src/sine_extraction/__main__.py`
- [x] 0.5.5 Create `src/sine_extraction/data/` directory
- [x] 0.5.6 Create empty `src/sine_extraction/data/__init__.py`
- [x] 0.5.7 Create `src/sine_extraction/models/` directory
- [x] 0.5.8 Create empty `src/sine_extraction/models/__init__.py`
- [x] 0.5.9 Create `src/sine_extraction/training/` directory
- [x] 0.5.10 Create empty `src/sine_extraction/training/__init__.py`
- [x] 0.5.11 Create `src/sine_extraction/evaluation/` directory
- [x] 0.5.12 Create empty `src/sine_extraction/evaluation/__init__.py`
- [x] 0.5.13 Create `src/sine_extraction/visualization/` directory
- [x] 0.5.14 Create empty `src/sine_extraction/visualization/__init__.py`
- [x] 0.5.15 Add `[tool.setuptools.packages.find]` with `where = ["src"]` to `pyproject.toml`
- [x] 0.5.16 Verify `uv run python -c "import sine_extraction"` succeeds

### 0.6 Artifact Directories
- [x] 0.6.1 Create `artifacts/` directory
- [x] 0.6.2 Create `artifacts/data/` directory
- [x] 0.6.3 Create `artifacts/checkpoints/` directory
- [x] 0.6.4 Create `artifacts/results/` directory
- [x] 0.6.5 Create `artifacts/plots/` directory
- [x] 0.6.6 Add `.gitkeep` to each artifact subdirectory
- [x] 0.6.7 Add `artifacts/data/`, `artifacts/checkpoints/` etc. to `.gitignore` (keep .gitkeep)

### 0.7 Configuration File
- [x] 0.7.1 Create `config/` directory
- [x] 0.7.2 Create `config/config.yaml` with `seed: 42`
- [x] 0.7.3 Add `signal:` section with `frequencies: [1, 5, 10, 20]`
- [x] 0.7.4 Add `signal.sample_rate: 200`
- [x] 0.7.5 Add `signal.window_size: 10`
- [x] 0.7.6 Add `signal.target_frequency: 10`
- [x] 0.7.7 Add `signal.amplitude: 1.0`
- [x] 0.7.8 Add `signal.amplitude_jitter_std: 0.05`
- [x] 0.7.9 Add `signal.phase_jitter_std: 0.1`
- [x] 0.7.10 Add `signal.noise_std: 0.1`
- [x] 0.7.11 Add `data:` section with `num_windows: 10000`
- [x] 0.7.12 Add `data.train_ratio: 0.70`
- [x] 0.7.13 Add `data.val_ratio: 0.15`
- [x] 0.7.14 Add `data.test_ratio: 0.15`
- [x] 0.7.15 Add `data.batch_size: 64`
- [x] 0.7.16 Add `data.shuffle: true`
- [x] 0.7.17 Add `model.mlp.hidden_sizes: [64, 128, 64]`
- [x] 0.7.18 Add `model.mlp.activation: relu`
- [x] 0.7.19 Add `model.rnn.hidden_size: 64`
- [x] 0.7.20 Add `model.rnn.num_layers: 2`
- [x] 0.7.21 Add `model.lstm.hidden_size: 64`
- [x] 0.7.22 Add `model.lstm.num_layers: 2`
- [x] 0.7.23 Add `training.learning_rate: 0.001`
- [x] 0.7.24 Add `training.epochs: 100`
- [x] 0.7.25 Add `training.early_stopping_patience: 10`
- [x] 0.7.26 Add `training.checkpoint_dir: artifacts/checkpoints`
- [x] 0.7.27 Add `training.results_dir: artifacts/results`
- [x] 0.7.28 Add `visualization.plots_dir: artifacts/plots`
- [x] 0.7.29 Add `visualization.interactive: false`
- [x] 0.7.30 Add `visualization.num_samples_to_plot: 3`

---

## PHASE 1 — Types & Constants (TDD)

### 1.1 Write tests for constants FIRST
- [x] 1.1.1 Create `tests/` directory
- [x] 1.1.2 Create `tests/__init__.py`
- [x] 1.1.3 Create `tests/conftest.py`
- [x] 1.1.4 Create `tests/test_constants.py`
- [x] 1.1.5 Write test: `KNOWN_FREQUENCIES` is a list of 4 floats
- [x] 1.1.6 Write test: `SAMPLE_RATE` equals 200
- [x] 1.1.7 Write test: `WINDOW_SIZE` equals 10
- [x] 1.1.8 Write test: `DEFAULT_SEED` equals 42
- [x] 1.1.9 Write test: `TWO_PI` approximately equals 6.2832
- [x] 1.1.10 Confirm tests FAIL (module not yet implemented)

### 1.2 Implement `constants.py`
- [x] 1.2.1 Create `src/sine_extraction/constants.py`
- [x] 1.2.2 Define `KNOWN_FREQUENCIES: list[float] = [1.0, 5.0, 10.0, 20.0]`
- [x] 1.2.3 Define `SAMPLE_RATE: int = 200`
- [x] 1.2.4 Define `WINDOW_SIZE: int = 10`
- [x] 1.2.5 Define `DEFAULT_SEED: int = 42`
- [x] 1.2.6 Define `TWO_PI: float = 2.0 * math.pi`
- [x] 1.2.7 Define `DEFAULT_CONFIG_PATH: Path = Path("config/config.yaml")`
- [x] 1.2.8 Define `CHECKPOINT_FILENAME_TEMPLATE: str = "{model_name}_epoch{epoch}.pt"`
- [x] 1.2.9 Add type hints to all definitions
- [x] 1.2.10 Confirm `test_constants.py` tests PASS
- [x] 1.2.11 Confirm file is under 150 lines
- [x] 1.2.12 Confirm `ruff check` reports 0 violations

### 1.3 Write tests for types FIRST
- [x] 1.3.1 Create `tests/test_types.py`
- [x] 1.3.2 Write test: `SignalConfig` dataclass can be instantiated with all fields
- [x] 1.3.3 Write test: `DataConfig` dataclass can be instantiated
- [x] 1.3.4 Write test: `MLPConfig` dataclass can be instantiated
- [x] 1.3.5 Write test: `RNNConfig` dataclass can be instantiated
- [x] 1.3.6 Write test: `LSTMConfig` dataclass can be instantiated
- [x] 1.3.7 Write test: `ModelConfig` dataclass can be instantiated
- [x] 1.3.8 Write test: `TrainingConfig` dataclass can be instantiated
- [x] 1.3.9 Write test: `VisualizationConfig` dataclass can be instantiated
- [x] 1.3.10 Write test: `AppConfig` dataclass can be instantiated (in test_types_app.py)
- [x] 1.3.11 Confirm tests FAIL

### 1.4 Implement `types.py`
- [x] 1.4.1 Create `src/sine_extraction/types.py`
- [x] 1.4.2 Import `dataclass` and `field` from `dataclasses`
- [x] 1.4.3 Define `SignalConfig` with fields: `frequencies`, `sample_rate`, `window_size`, `target_frequency`, `amplitude`, `amplitude_jitter_std`, `phase_jitter_std`, `noise_std`
- [x] 1.4.4 Define `DataConfig` with fields: `num_windows`, `train_ratio`, `val_ratio`, `test_ratio`, `batch_size`, `shuffle`
- [x] 1.4.5 Define `MLPConfig` with fields: `hidden_sizes`, `activation`
- [x] 1.4.6 Define `RNNConfig` with fields: `hidden_size`, `num_layers`
- [x] 1.4.7 Define `LSTMConfig` with fields: `hidden_size`, `num_layers`
- [x] 1.4.8 Define `ModelConfig` with fields: `mlp`, `rnn`, `lstm`
- [x] 1.4.9 Define `TrainingConfig` with fields: `learning_rate`, `epochs`, `early_stopping_patience`, `checkpoint_dir`, `results_dir`
- [x] 1.4.10 Define `VisualizationConfig` with fields: `plots_dir`, `interactive`, `num_samples_to_plot`
- [x] 1.4.11 Define `AppConfig` with fields: `seed`, `signal`, `data`, `model`, `training`, `visualization`
- [x] 1.4.12 Add correct type hints to every field
- [x] 1.4.13 Confirm `test_types.py` tests PASS
- [x] 1.4.14 Confirm file is under 150 lines
- [x] 1.4.15 Confirm `ruff check` 0 violations

---

## PHASE 2 — Configuration Loader (TDD)

### 2.1 Write tests FIRST
- [x] 2.1.1 Create `tests/test_config_loader.py`
- [x] 2.1.2 Write fixture: temporary `config.yaml` file with minimal valid content
- [x] 2.1.3 Write test: `load_config(path)` returns an `AppConfig` instance
- [x] 2.1.4 Write test: `config.seed` equals the value in YAML
- [x] 2.1.5 Write test: `config.signal.sample_rate` equals 200
- [x] 2.1.6 Write test: `config.signal.frequencies` equals [1.0, 5.0, 10.0, 20.0]
- [x] 2.1.7 Write test: `config.data.batch_size` equals 64
- [x] 2.1.8 Write test: `config.model.mlp.hidden_sizes` is a list of ints
- [x] 2.1.9 Write test: `load_config` raises `FileNotFoundError` for missing path
- [x] 2.1.10 Write test: `load_config` raises `ValueError` for invalid YAML structure
- [x] 2.1.11 Confirm tests FAIL

### 2.2 Implement `config_loader.py`
- [x] 2.2.1 Create `src/sine_extraction/config_loader.py`
- [x] 2.2.2 Import `yaml`, `Path`, `os`, `dotenv`
- [x] 2.2.3 Call `load_dotenv()` at module top level
- [x] 2.2.4 Implement `load_config(path: Path) -> AppConfig`
- [x] 2.2.5 Open and parse the YAML file
- [x] 2.2.6 Build `SignalConfig` from `raw["signal"]`
- [x] 2.2.7 Build `DataConfig` from `raw["data"]`
- [x] 2.2.8 Build `MLPConfig`, `RNNConfig`, `LSTMConfig` from `raw["model"]`
- [x] 2.2.9 Build `ModelConfig` from the three model configs
- [x] 2.2.10 Build `TrainingConfig` from `raw["training"]`
- [x] 2.2.11 Build `VisualizationConfig` from `raw["visualization"]`
- [x] 2.2.12 Build and return `AppConfig`
- [x] 2.2.13 Raise `FileNotFoundError` with clear message if path does not exist
- [x] 2.2.14 Raise `ValueError` with clear message for missing required keys
- [x] 2.2.15 Confirm all `test_config_loader.py` tests PASS
- [x] 2.2.16 Confirm file is under 150 lines
- [x] 2.2.17 Confirm `ruff check` 0 violations

---

## PHASE 3 — Signal Data Generation (TDD)

### 3.1 Write tests for `SignalGenerator` FIRST
- [x] 3.1.1 Create `tests/test_generator.py`
- [x] 3.1.2 Write fixture: default `SignalConfig` object
- [x] 3.1.3 Write test: `generate_dataset(N)` returns `(X, y)` with shapes `(N, 10)` each
- [x] 3.1.4 Write test: `X` dtype is `float32`
- [x] 3.1.5 Write test: `y` dtype is `float32`
- [x] 3.1.6 Write test: pure target values are bounded — max abs value ≤ `amplitude + 3*amplitude_jitter_std`
- [x] 3.1.7 Write test: with same seed, two calls return identical `X` and `y`
- [x] 3.1.8 Write test: with different seeds, `X` arrays differ
- [x] 3.1.9 Write test: `generate_pure_sine` returns array with correct length
- [x] 3.1.10 Write test: `generate_pure_sine` at 0 amplitude jitter/phase jitter produces exact sine values
- [x] 3.1.11 Write test: mixed signal has nonzero contribution from all four frequencies (correlation check)
- [x] 3.1.12 Confirm tests FAIL

### 3.2 Implement `data/generator.py`
- [x] 3.2.1 Create `src/sine_extraction/data/generator.py`
- [x] 3.2.2 Import `numpy as np`, `SignalConfig`, constants
- [x] 3.2.3 Define `class SignalGenerator`
- [x] 3.2.4 Implement `__init__(self, config: SignalConfig, seed: int) -> None`
- [x] 3.2.5 Seed numpy RNG with `seed` in `__init__`
- [x] 3.2.6 Implement `_time_axis(self, start_sample: int) -> np.ndarray`
  - [x] 3.2.6a Compute time as `np.arange(window_size) / sample_rate + start_sample / sample_rate`
- [x] 3.2.7 Implement `_sample_phase(self) -> float`
  - [x] 3.2.7a Draw from `U(0, 2π)` using the seeded RNG
- [x] 3.2.8 Implement `_sample_amplitude(self) -> float`
  - [x] 3.2.8a Draw from `N(amplitude, amplitude_jitter_std)` using seeded RNG
- [x] 3.2.9 Implement `generate_pure_sine(self, frequency: float, start_sample: int) -> np.ndarray`
  - [x] 3.2.9a Compute `t = _time_axis(start_sample)`
  - [x] 3.2.9b Return `amplitude * sin(TWO_PI * frequency * t + phase)`
- [x] 3.2.10 Implement `generate_mixed_window(self, start_sample: int) -> tuple[np.ndarray, np.ndarray]`
  - [x] 3.2.10a For each frequency in `config.frequencies`, call `generate_pure_sine` with jitter
  - [x] 3.2.10b Sum all components
  - [x] 3.2.10c Add white Gaussian noise from `N(0, noise_std)`
  - [x] 3.2.10d Also compute the clean target (no jitter, no noise) for target frequency
  - [x] 3.2.10e Return `(noisy_mix, clean_target)`
- [x] 3.2.11 Implement `generate_dataset(self, num_windows: int) -> tuple[np.ndarray, np.ndarray]`
  - [x] 3.2.11a Loop `num_windows` times
  - [x] 3.2.11b Randomly sample `start_sample` from valid range
  - [x] 3.2.11c Collect `(X_i, y_i)` pairs
  - [x] 3.2.11d Stack into arrays of shape `(num_windows, window_size)`
  - [x] 3.2.11e Cast to `float32`
- [x] 3.2.12 Confirm all `test_generator.py` tests PASS
- [x] 3.2.13 Confirm file is under 150 lines
- [x] 3.2.14 Confirm `ruff check` 0 violations

### 3.3 Write tests for `SineDataset` FIRST
- [x] 3.3.1 Create `tests/test_dataset.py`
- [x] 3.3.2 Write fixture: tiny X and y arrays (100 samples, shape [100, 10])
- [x] 3.3.3 Write test: `len(dataset)` equals 100
- [x] 3.3.4 Write test: `dataset[0]` returns a tuple of two tensors
- [x] 3.3.5 Write test: first tensor has shape `(10,)`
- [x] 3.3.6 Write test: second tensor has shape `(10,)`
- [x] 3.3.7 Write test: tensor dtype is `torch.float32`
- [x] 3.3.8 Write test: values in dataset match input arrays (numerical equality)
- [x] 3.3.9 Confirm tests FAIL

### 3.4 Implement `data/dataset.py`
- [x] 3.4.1 Create `src/sine_extraction/data/dataset.py`
- [x] 3.4.2 Import `torch`, `Dataset`, `numpy`
- [x] 3.4.3 Define `class SineDataset(Dataset)`
- [x] 3.4.4 Implement `__init__(self, X: np.ndarray, y: np.ndarray) -> None`
  - [x] 3.4.4a Convert `X` to `torch.float32` tensor
  - [x] 3.4.4b Convert `y` to `torch.float32` tensor
- [x] 3.4.5 Implement `__len__(self) -> int` returning `len(self.X)`
- [x] 3.4.6 Implement `__getitem__(self, idx: int) -> tuple[Tensor, Tensor]`
- [x] 3.4.7 Confirm all `test_dataset.py` tests PASS
- [x] 3.4.8 Confirm file is under 150 lines
- [x] 3.4.9 Confirm `ruff check` 0 violations

### 3.5 Write tests for `splitter` FIRST
- [x] 3.5.1 Create `tests/test_splitter.py`
- [x] 3.5.2 Write test: `make_dataloaders(X, y, config)` returns a 3-tuple
- [x] 3.5.3 Write test: train split size ≈ 70% of total (±1 sample tolerance)
- [x] 3.5.4 Write test: val split size ≈ 15% of total
- [x] 3.5.5 Write test: test split size ≈ 15% of total
- [x] 3.5.6 Write test: train + val + test sizes sum to total
- [x] 3.5.7 Write test: each loader is a `DataLoader` instance
- [x] 3.5.8 Write test: loaders yield batches of shape `(batch_size, 10)` or smaller (last batch)
- [x] 3.5.9 Write test: no sample appears in more than one split (index check)
- [x] 3.5.10 Confirm tests FAIL

### 3.6 Implement `data/splitter.py`
- [x] 3.6.1 Create `src/sine_extraction/data/splitter.py`
- [x] 3.6.2 Import `DataLoader`, `random_split`, `Dataset`, `DataConfig`
- [x] 3.6.3 Define `make_dataloaders(X, y, config: DataConfig) -> tuple[DataLoader, DataLoader, DataLoader]`
- [x] 3.6.4 Create a `SineDataset` from X and y
- [x] 3.6.5 Compute exact integer split sizes from ratios (ensure they sum to total)
- [x] 3.6.6 Call `random_split` with computed sizes
- [x] 3.6.7 Create `DataLoader` for each split with `batch_size` and `shuffle` from config
- [x] 3.6.8 Validation loader should have `shuffle=False`
- [x] 3.6.9 Test loader should have `shuffle=False`
- [x] 3.6.10 Return `(train_loader, val_loader, test_loader)`
- [x] 3.6.11 Confirm all `test_splitter.py` tests PASS
- [x] 3.6.12 Confirm file is under 150 lines
- [x] 3.6.13 Confirm `ruff check` 0 violations

---

## PHASE 4 — Model Architectures (TDD)

### 4.1 Write tests for `BaseModel` FIRST
- [x] 4.1.1 Create tests for abstract interface (indirect — via MLP/RNN/LSTM tests)
- [x] 4.1.2 Write test: any model subclass has `count_parameters()` method returning positive int
- [x] 4.1.3 Write test: `save(path)` writes a file to disk
- [x] 4.1.4 Write test: `load(path, config)` returns a model with same parameters

### 4.2 Implement `models/base.py`
- [x] 4.2.1 Create `src/sine_extraction/models/base.py`
- [x] 4.2.2 Import `ABC`, `abstractmethod`, `nn`, `Path`, `torch`
- [x] 4.2.3 Define `class BaseModel(nn.Module, ABC)`
- [x] 4.2.4 Declare `@abstractmethod forward(self, x: Tensor) -> Tensor`
- [x] 4.2.5 Implement `count_parameters(self) -> int`
  - [x] 4.2.5a `sum(p.numel() for p in self.parameters() if p.requires_grad)`
- [x] 4.2.6 Implement `save(self, path: Path) -> None`
  - [x] 4.2.6a `torch.save(self.state_dict(), path)`
- [x] 4.2.7 Confirm file is under 150 lines
- [x] 4.2.8 Confirm `ruff check` 0 violations

### 4.3 Write tests for `MLPModel` FIRST
- [x] 4.3.1 Create `tests/test_mlp.py`
- [x] 4.3.2 Write fixture: default `MLPConfig` and `window_size=10`
- [x] 4.3.3 Write test: `MLPModel(config, window_size)` instantiates without error
- [x] 4.3.4 Write test: `forward(x)` with `x.shape == [4, 10]` returns tensor of shape `[4, 10]`
- [x] 4.3.5 Write test: `forward` output has no `nan` values
- [x] 4.3.6 Write test: `count_parameters()` returns positive integer
- [x] 4.3.7 Write test: `save(path)` creates a `.pt` file
- [x] 4.3.8 Write test: hidden_sizes `[64, 128, 64]` creates 3 hidden layers
- [x] 4.3.9 Write test: model is an `nn.Module` instance
- [x] 4.3.10 Confirm tests FAIL

### 4.4 Implement `models/mlp.py`
- [x] 4.4.1 Create `src/sine_extraction/models/mlp.py`
- [x] 4.4.2 Import `nn`, `Tensor`, `MLPConfig`, `BaseModel`
- [x] 4.4.3 Define `class MLPModel(BaseModel)`
- [x] 4.4.4 Implement `__init__(self, config: MLPConfig, window_size: int) -> None`
  - [x] 4.4.4a Call `super().__init__()`
  - [x] 4.4.4b Build layer list: `[Linear(window_size, hidden_sizes[0]), ReLU, ...]`
  - [x] 4.4.4c Handle `activation` field from config (map string to nn.Module)
  - [x] 4.4.4d Final layer: `Linear(hidden_sizes[-1], window_size)`
  - [x] 4.4.4e Wrap in `nn.Sequential`
- [x] 4.4.5 Implement `forward(self, x: Tensor) -> Tensor`
  - [x] 4.4.5a Assert input shape is `[batch, window_size]`
  - [x] 4.4.5b Return `self.network(x)`
- [x] 4.4.6 Confirm all `test_mlp.py` tests PASS
- [x] 4.4.7 Confirm file is under 150 lines
- [x] 4.4.8 Confirm `ruff check` 0 violations

### 4.5 Write tests for `RNNModel` FIRST
- [x] 4.5.1 Create `tests/test_rnn.py`
- [x] 4.5.2 Write fixture: default `RNNConfig` and `window_size=10`
- [x] 4.5.3 Write test: `RNNModel(config, window_size)` instantiates without error
- [x] 4.5.4 Write test: `forward(x)` with `x.shape == [4, 10]` returns tensor of shape `[4, 10]`
- [x] 4.5.5 Write test: `forward` output has no `nan` values
- [x] 4.5.6 Write test: `count_parameters()` returns positive integer
- [x] 4.5.7 Write test: model is an `nn.Module` instance
- [x] 4.5.8 Write test: `hidden_size` and `num_layers` are reflected in the internal RNN
- [x] 4.5.9 Confirm tests FAIL

### 4.6 Implement `models/rnn.py`
- [x] 4.6.1 Create `src/sine_extraction/models/rnn.py`
- [x] 4.6.2 Import `nn`, `Tensor`, `RNNConfig`, `BaseModel`
- [x] 4.6.3 Define `class RNNModel(BaseModel)`
- [x] 4.6.4 Implement `__init__(self, config: RNNConfig, window_size: int) -> None`
  - [x] 4.6.4a Call `super().__init__()`
  - [x] 4.6.4b Create `self.rnn = nn.RNN(input_size=1, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True)`
  - [x] 4.6.4c Create `self.linear = nn.Linear(config.hidden_size, 1)`
- [x] 4.6.5 Implement `forward(self, x: Tensor) -> Tensor`
  - [x] 4.6.5a Unsqueeze last dim: `x = x.unsqueeze(-1)` → shape `[batch, 10, 1]`
  - [x] 4.6.5b Pass through RNN: `out, _ = self.rnn(x)`
  - [x] 4.6.5c Apply linear to each timestep: `out = self.linear(out)` → `[batch, 10, 1]`
  - [x] 4.6.5d Squeeze last dim → `[batch, 10]`
- [x] 4.6.6 Confirm all `test_rnn.py` tests PASS
- [x] 4.6.7 Confirm file is under 150 lines
- [x] 4.6.8 Confirm `ruff check` 0 violations

### 4.7 Write tests for `LSTMModel` FIRST
- [x] 4.7.1 Create `tests/test_lstm.py`
- [x] 4.7.2 Write fixture: default `LSTMConfig` and `window_size=10`
- [x] 4.7.3 Write test: `LSTMModel(config, window_size)` instantiates without error
- [x] 4.7.4 Write test: `forward(x)` with `x.shape == [4, 10]` returns tensor of shape `[4, 10]`
- [x] 4.7.5 Write test: `forward` output has no `nan` values
- [x] 4.7.6 Write test: `count_parameters()` returns positive integer
- [x] 4.7.7 Write test: model is an `nn.Module` instance
- [x] 4.7.8 Write test: `hidden_size` and `num_layers` reflected in internal LSTM
- [x] 4.7.9 Write test: LSTM produces different output from RNN (sanity check)
- [x] 4.7.10 Confirm tests FAIL

### 4.8 Implement `models/lstm.py`
- [x] 4.8.1 Create `src/sine_extraction/models/lstm.py`
- [x] 4.8.2 Import `nn`, `Tensor`, `LSTMConfig`, `BaseModel`
- [x] 4.8.3 Define `class LSTMModel(BaseModel)`
- [x] 4.8.4 Implement `__init__(self, config: LSTMConfig, window_size: int) -> None`
  - [x] 4.8.4a Call `super().__init__()`
  - [x] 4.8.4b Create `self.lstm = nn.LSTM(input_size=1, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True)`
  - [x] 4.8.4c Create `self.linear = nn.Linear(config.hidden_size, 1)`
- [x] 4.8.5 Implement `forward(self, x: Tensor) -> Tensor`
  - [x] 4.8.5a Unsqueeze: `x = x.unsqueeze(-1)`
  - [x] 4.8.5b `out, (h_n, c_n) = self.lstm(x)`
  - [x] 4.8.5c `out = self.linear(out).squeeze(-1)` → `[batch, 10]`
- [x] 4.8.6 Confirm all `test_lstm.py` tests PASS
- [x] 4.8.7 Confirm file is under 150 lines
- [x] 4.8.8 Confirm `ruff check` 0 violations

---

## PHASE 5 — Training (TDD)

### 5.1 Write tests for `losses.py` FIRST
- [x] 5.1.1 Create `tests/test_losses.py`
- [x] 5.1.2 Write test: `mse_loss(pred, target)` returns scalar tensor
- [x] 5.1.3 Write test: `mse_loss(x, x)` equals 0.0 for any `x`
- [x] 5.1.4 Write test: `mse_loss` is symmetric: `mse_loss(a, b) == mse_loss(b, a)`
- [x] 5.1.5 Write test: known input → known MSE value (e.g., pred=[1.0], target=[0.0] → MSE=1.0)
- [x] 5.1.6 Confirm tests FAIL

### 5.2 Implement `training/losses.py`
- [x] 5.2.1 Create `src/sine_extraction/training/losses.py`
- [x] 5.2.2 Import `nn`, `Tensor`
- [x] 5.2.3 Define `_mse = nn.MSELoss()` as module-level singleton
- [x] 5.2.4 Implement `mse_loss(pred: Tensor, target: Tensor) -> Tensor`
  - [x] 5.2.4a Return `_mse(pred, target)`
- [x] 5.2.5 Confirm all `test_losses.py` tests PASS
- [x] 5.2.6 Confirm file is under 150 lines
- [x] 5.2.7 Confirm `ruff check` 0 violations

### 5.3 Write tests for `Trainer` FIRST
- [x] 5.3.1 Create `tests/test_trainer.py`
- [x] 5.3.2 Write fixture: tiny MLP model + tiny DataLoaders (10 samples)
- [x] 5.3.3 Write fixture: minimal `TrainingConfig` (epochs=3, patience=2)
- [x] 5.3.4 Write test: `Trainer(model, train, val, config, device)` instantiates
- [x] 5.3.5 Write test: `train()` returns a dict with keys `train_loss` and `val_loss`
- [x] 5.3.6 Write test: `train_loss` list has length ≤ epochs (may be shorter due to early stopping)
- [x] 5.3.7 Write test: final `train_loss` value is a float
- [x] 5.3.8 Write test: checkpoint file is created in `checkpoint_dir` after training
- [x] 5.3.9 Write test: early stopping halts training when val_loss doesn't improve for `patience` epochs
- [x] 5.3.10 Write test: training on tiny data results in lower final loss than initial loss (sanity check)
- [x] 5.3.11 Confirm tests FAIL

### 5.4 Implement `training/trainer.py`
- [x] 5.4.1 Create `src/sine_extraction/training/trainer.py`
- [x] 5.4.2 Import `torch`, `Path`, `DataLoader`, `BaseModel`, `TrainingConfig`, `mse_loss`
- [x] 5.4.3 Define `class Trainer`
- [x] 5.4.4 Implement `__init__` storing model, loaders, config, device
- [x] 5.4.5 Initialize Adam optimizer: `torch.optim.Adam(model.parameters(), lr=config.learning_rate)`
- [x] 5.4.6 Implement `_train_epoch(self) -> float`
  - [x] 5.4.6a Set `self.model.train()`
  - [x] 5.4.6b Loop over `train_loader` batches
  - [x] 5.4.6c Move batch to device
  - [x] 5.4.6d Zero gradients, forward pass, compute loss, backward, step
  - [x] 5.4.6e Accumulate and return mean epoch loss
- [x] 5.4.7 Implement `_val_epoch(self) -> float`
  - [x] 5.4.7a Set `self.model.eval()`
  - [x] 5.4.7b No-grad context, forward, compute and return mean val loss
- [x] 5.4.8 Implement `_save_checkpoint(self, epoch: int, val_loss: float) -> Path`
  - [x] 5.4.8a Build filename from `CHECKPOINT_FILENAME_TEMPLATE`
  - [x] 5.4.8b Call `self.model.save(checkpoint_path)`
  - [x] 5.4.8c Return the path
- [x] 5.4.9 Implement `train(self) -> dict[str, list[float]]`
  - [x] 5.4.9a Loop epochs
  - [x] 5.4.9b Call `_train_epoch`, `_val_epoch`
  - [x] 5.4.9c Append losses to history lists
  - [x] 5.4.9d Check for improvement: if `val_loss < best_val_loss`, reset counter, save checkpoint
  - [x] 5.4.9e Else increment patience counter; if counter ≥ patience, break (early stop)
  - [x] 5.4.9f Return history dict
- [x] 5.4.10 Confirm all `test_trainer.py` tests PASS
- [x] 5.4.11 Confirm file is under 150 lines; split if needed
- [x] 5.4.12 Confirm `ruff check` 0 violations

---

## PHASE 6 — Evaluation (TDD)

### 6.1 Write tests for `metrics.py` FIRST
- [x] 6.1.1 Create `tests/test_metrics.py`
- [x] 6.1.2 Write test: `compute_mse(pred, target)` returns float
- [x] 6.1.3 Write test: perfect predictions → MSE = 0.0
- [x] 6.1.4 Write test: known offset → known MSE value
- [x] 6.1.5 Write test: `compute_mae(pred, target)` returns float
- [x] 6.1.6 Write test: perfect predictions → MAE = 0.0
- [x] 6.1.7 Write test: `compute_r2(pred, target)` returns float in range [-∞, 1.0]
- [x] 6.1.8 Write test: perfect predictions → R² = 1.0
- [x] 6.1.9 Write test: `evaluate_model(model, loader, device)` returns dict with keys `mse`, `mae`, `r2`
- [x] 6.1.10 Write test: `evaluate_model` with trivial model (identity) returns sensible metrics
- [x] 6.1.11 Confirm tests FAIL

### 6.2 Implement `evaluation/metrics.py`
- [x] 6.2.1 Create `src/sine_extraction/evaluation/metrics.py`
- [x] 6.2.2 Import `torch`, `numpy`, `DataLoader`, `BaseModel`
- [x] 6.2.3 Implement `compute_mse(pred: np.ndarray, target: np.ndarray) -> float`
- [x] 6.2.4 Implement `compute_mae(pred: np.ndarray, target: np.ndarray) -> float`
- [x] 6.2.5 Implement `compute_r2(pred: np.ndarray, target: np.ndarray) -> float`
  - [x] 6.2.5a `SS_res = sum((y - y_hat)^2)`, `SS_tot = sum((y - mean(y))^2)`
  - [x] 6.2.5b Return `1 - SS_res / SS_tot`
- [x] 6.2.6 Implement `evaluate_model(model, test_loader, device) -> dict[str, float]`
  - [x] 6.2.6a Set model to eval mode
  - [x] 6.2.6b Collect all predictions and targets in no-grad context
  - [x] 6.2.6c Convert to numpy
  - [x] 6.2.6d Compute and return `{'mse': ..., 'mae': ..., 'r2': ...}`
- [x] 6.2.7 Confirm all `test_metrics.py` tests PASS
- [x] 6.2.8 Confirm file is under 150 lines
- [x] 6.2.9 Confirm `ruff check` 0 violations

---

## PHASE 7 — Visualization (TDD)

### 7.1 Write tests for `plotter.py` FIRST
- [x] 7.1.1 Create `tests/test_visualization.py`
- [x] 7.1.2 Write fixture: fake signal arrays (shape `[10]`) for noisy, pure, mlp, rnn, lstm
- [x] 7.1.3 Write fixture: `VisualizationConfig` with `interactive=False` and temp `plots_dir`
- [x] 7.1.4 Write test: `ComparisonPlotter(config)` instantiates without error
- [x] 7.1.5 Write test: `plot(...)` returns without raising exceptions (headless)
- [x] 7.1.6 Write test: `save("test.png")` creates a file at the expected path
- [x] 7.1.7 Write test: the saved file is a valid image (check file size > 0)
- [x] 7.1.8 Write test: after `plot()`, the figure has exactly 5 subplots/axes
- [x] 7.1.9 Write test: `plots_dir` is created if it does not exist
- [x] 7.1.10 Confirm tests FAIL

### 7.2 Implement `visualization/plotter.py`
- [x] 7.2.1 Create `src/sine_extraction/visualization/plotter.py`
- [x] 7.2.2 Import `matplotlib.pyplot as plt`, `numpy`, `Path`, `VisualizationConfig`
- [x] 7.2.3 Set `matplotlib.use("Agg")` when `interactive=False`
- [x] 7.2.4 Define `class ComparisonPlotter`
- [x] 7.2.5 Implement `__init__(self, config: VisualizationConfig) -> None`
  - [x] 7.2.5a Store config
  - [x] 7.2.5b Ensure `plots_dir` exists via `Path(config.plots_dir).mkdir(parents=True, exist_ok=True)`
  - [x] 7.2.5c Initialize `self.fig = None`
- [x] 7.2.6 Implement `plot(self, noisy, pure, mlp_pred, rnn_pred, lstm_pred, sample_idx=0) -> None`
  - [x] 7.2.6a Create figure with 5 subplots arranged vertically: `fig, axes = plt.subplots(5, 1, figsize=(10, 12))`
  - [x] 7.2.6b Panel 1: plot `noisy` with title "Noisy Mixed Input"
  - [x] 7.2.6c Panel 2: plot `pure` with title "Ground Truth Pure Sine"
  - [x] 7.2.6d Panel 3: plot `mlp_pred` with title "MLP Prediction"
  - [x] 7.2.6e Panel 4: plot `rnn_pred` with title "RNN Prediction"
  - [x] 7.2.6f Panel 5: plot `lstm_pred` with title "LSTM Prediction"
  - [x] 7.2.6g Add x-axis label "Sample Index" to bottom panel
  - [x] 7.2.6h Add y-axis label "Amplitude" to each panel
  - [x] 7.2.6i Add overall `fig.suptitle(f"Signal Comparison — Sample {sample_idx}")`
  - [x] 7.2.6j Call `plt.tight_layout()`
  - [x] 7.2.6k Store `self.fig = fig`
  - [x] 7.2.6l If `config.interactive`, call `plt.show()`
- [x] 7.2.7 Implement `save(self, filename: str) -> Path`
  - [x] 7.2.7a Assert `self.fig is not None`
  - [x] 7.2.7b Build `path = Path(config.plots_dir) / filename`
  - [x] 7.2.7c `self.fig.savefig(path, dpi=150, bbox_inches="tight")`
  - [x] 7.2.7d Return `path`
- [x] 7.2.8 Confirm all `test_visualization.py` tests PASS
- [x] 7.2.9 Confirm file is under 150 lines
- [x] 7.2.10 Confirm `ruff check` 0 violations

---

## PHASE 8 — Public API & CLI (TDD)

### 8.1 Update `__init__.py`
- [x] 8.1.1 Import and re-export `load_config`, `AppConfig`
- [x] 8.1.2 Import and re-export `SignalGenerator`, `SineDataset`, `make_dataloaders`
- [x] 8.1.3 Import and re-export `MLPModel`, `RNNModel`, `LSTMModel`
- [x] 8.1.4 Import and re-export `Trainer`
- [x] 8.1.5 Import and re-export `evaluate_model`
- [x] 8.1.6 Import and re-export `ComparisonPlotter`
- [x] 8.1.7 Define `__all__` list with all exported names
- [x] 8.1.8 Verify `from sine_extraction import MLPModel` works in a fresh Python session
- [x] 8.1.9 Confirm file is under 150 lines
- [x] 8.1.10 Confirm `ruff check` 0 violations

### 8.2 Write tests for CLI FIRST
- [x] 8.2.1 Create `tests/test_cli.py`
- [x] 8.2.2 Import `CliRunner` from `click.testing`
- [x] 8.2.3 Write test: `cli generate` subcommand runs without error code (exit 0)
- [x] 8.2.4 Write test: `cli train` subcommand completes without exception on tiny data
- [x] 8.2.5 Write test: `cli evaluate` subcommand completes without exception
- [x] 8.2.6 Write test: `cli visualize` subcommand creates a file in `plots_dir`
- [x] 8.2.7 Write test: `cli all` subcommand completes full pipeline
- [x] 8.2.8 Write test: `cli --help` prints usage info
- [x] 8.2.9 Confirm tests FAIL

### 8.3 Implement CLI (split across `cli/` sub-package)
- [x] 8.3.1 Create `src/sine_extraction/__main__.py`
- [x] 8.3.2 Import `click`, `load_config`, `DEFAULT_CONFIG_PATH` and all SDK classes
- [x] 8.3.3 Define `@click.group()` CLI group `cli` in `cli/main.py`
- [x] 8.3.4 Add `--config` option to group, defaulting to `DEFAULT_CONFIG_PATH`
- [x] 8.3.5 Implement `generate` subcommand:
  - [x] 8.3.5a Load config
  - [x] 8.3.5b Set random seeds (numpy, torch)
  - [x] 8.3.5c Create `SignalGenerator` and call `generate_dataset`
  - [x] 8.3.5d Save X and y to data-dir using `numpy.save`
  - [x] 8.3.5e Print confirmation message
- [x] 8.3.6 Implement `train` subcommand:
  - [x] 8.3.6a Load config and data from data-dir
  - [x] 8.3.6b Create dataloaders
  - [x] 8.3.6c Detect device (`cuda` if available, else `cpu`)
  - [x] 8.3.6d Train `MLPModel`, `RNNModel`, `LSTMModel` sequentially
  - [x] 8.3.6e Print per-model loss summary
- [x] 8.3.7 Implement `evaluate` subcommand:
  - [x] 8.3.7a Load models from best checkpoints
  - [x] 8.3.7b Run `evaluate_model` for each
  - [x] 8.3.7c Save results to `results_dir/metrics.json`
  - [x] 8.3.7d Print metrics
- [x] 8.3.8 Implement `visualize` subcommand:
  - [x] 8.3.8a Load models and test data
  - [x] 8.3.8b Get predictions from each model
  - [x] 8.3.8c Call `ComparisonPlotter.plot()` and `save()`
  - [x] 8.3.8d Print save path
- [x] 8.3.9 Implement `all` subcommand: call generate → train → evaluate → visualize
- [x] 8.3.10 Add `if __name__ == "__main__": cli()` at bottom
- [x] 8.3.11 Confirm all `test_cli.py` tests PASS
- [x] 8.3.12 CLI split into `cli/main.py`, `cli/commands.py`, `cli/eval_vis.py` (all < 150 lines)
- [x] 8.3.13 Confirm `ruff check` 0 violations

---

## PHASE 9 — Integration & Coverage

### 9.1 Shared Test Fixtures
- [x] 9.1.1 Add `@pytest.fixture` for `default_signal_config` to `conftest.py`
- [x] 9.1.2 Add `@pytest.fixture` for `default_app_config` to `conftest.py`
- [x] 9.1.3 Add `@pytest.fixture` for `tiny_dataset` (100 samples) to `conftest.py`
- [x] 9.1.4 Add `@pytest.fixture` for `tiny_loaders` to `conftest.py`
- [x] 9.1.5 Add `@pytest.fixture` for `cpu_device` to `conftest.py`
- [x] 9.1.6 Add `@pytest.fixture` for `tmp_checkpoint_dir` using `tmp_path`

### 9.2 Coverage Check
- [x] 9.2.1 Run `uv run pytest --cov=sine_extraction --cov-report=term-missing`
- [x] 9.2.2 Review uncovered lines report
- [x] 9.2.3 Add tests for any uncovered branch in `generator.py`
- [x] 9.2.4 Add tests for any uncovered branch in `trainer.py`
- [x] 9.2.5 Add tests for any uncovered branch in `metrics.py`
- [x] 9.2.6 Add tests for any uncovered branch in `config_loader.py`
- [x] 9.2.7 Add tests for any uncovered branch in `__main__.py`
- [x] 9.2.8 Re-run pytest — confirm coverage ≥ 85%
- [x] 9.2.9 If coverage < 85%, identify and fill gaps iteratively

### 9.3 Ruff Full Sweep
- [x] 9.3.1 Run `uv run ruff check src/ tests/`
- [x] 9.3.2 Fix all `E` (pycodestyle) violations
- [x] 9.3.3 Fix all `F` (pyflakes) violations
- [x] 9.3.4 Fix all `I` (isort) violations
- [x] 9.3.5 Fix all `N` (pep8-naming) violations
- [x] 9.3.6 Fix all `ANN` (flake8-annotations) violations
- [x] 9.3.7 Fix all `B` (flake8-bugbear) violations
- [x] 9.3.8 Fix all `SIM` (flake8-simplify) violations
- [x] 9.3.9 Re-run ruff — confirm 0 violations

### 9.4 Line Count Audit
- [x] 9.4.1 Check `constants.py` ≤ 150 lines
- [x] 9.4.2 Check `types.py` ≤ 150 lines
- [x] 9.4.3 Check `config_loader.py` ≤ 150 lines
- [x] 9.4.4 Check `data/generator.py` ≤ 150 lines
- [x] 9.4.5 Check `data/dataset.py` ≤ 150 lines
- [x] 9.4.6 Check `data/splitter.py` ≤ 150 lines
- [x] 9.4.7 Check `models/base.py` ≤ 150 lines
- [x] 9.4.8 Check `models/mlp.py` ≤ 150 lines
- [x] 9.4.9 Check `models/rnn.py` ≤ 150 lines
- [x] 9.4.10 Check `models/lstm.py` ≤ 150 lines
- [x] 9.4.11 Check `training/trainer.py` ≤ 150 lines
- [x] 9.4.12 Check `training/losses.py` ≤ 150 lines
- [x] 9.4.13 Check `evaluation/metrics.py` ≤ 150 lines
- [x] 9.4.14 Check `visualization/plotter.py` ≤ 150 lines
- [x] 9.4.15 Check `__main__.py` ≤ 150 lines
- [x] 9.4.16 Check `__init__.py` ≤ 150 lines
- [x] 9.4.17 If any file exceeds 150 lines, split it per PLAN.md §10

### 9.5 End-to-End Pipeline Test
- [x] 9.5.1 Run `uv run python -m sine_extraction --help` — confirm help text
- [x] 9.5.2 Run `uv run python -m sine_extraction generate` — confirm data files created
- [x] 9.5.3 Run `uv run python -m sine_extraction train` — confirm 3 checkpoints created
- [x] 9.5.4 Run `uv run python -m sine_extraction evaluate` — confirm `metrics.json` created
- [x] 9.5.5 Run `uv run python -m sine_extraction visualize` — confirm PNG created
- [x] 9.5.6 Run `uv run python -m sine_extraction all` — confirm clean end-to-end
- [x] 9.5.7 Inspect the comparison PNG — confirm 5 panels visible with correct labels
- [x] 9.5.8 Inspect `metrics.json` — confirm MSE, MAE, R² present for all 3 models

---

## PHASE 10 — Seed Reproducibility & Device Support

### 10.1 Seed Management
- [x] 10.1.1 Create `src/sine_extraction/seeding.py`
- [x] 10.1.2 Implement `set_all_seeds(seed: int) -> None`
  - [x] 10.1.2a `import random; random.seed(seed)`
  - [x] 10.1.2b `import numpy as np; np.random.seed(seed)`
  - [x] 10.1.2c `import torch; torch.manual_seed(seed)`
  - [x] 10.1.2d `torch.cuda.manual_seed_all(seed)` if CUDA available
- [x] 10.1.3 Write test: two runs with same seed produce identical dataset
- [x] 10.1.4 Write test: two runs with different seeds produce different datasets
- [x] 10.1.5 Call `set_all_seeds(config.seed)` at start of each CLI subcommand

### 10.2 Device Auto-Detection
- [x] 10.2.1 Create `src/sine_extraction/device.py`
- [x] 10.2.2 Implement `get_device() -> torch.device`
  - [x] 10.2.2a Return `torch.device("cuda")` if `torch.cuda.is_available()`
  - [x] 10.2.2b Else return `torch.device("cpu")`
- [x] 10.2.3 Write test: `get_device()` returns a `torch.device` instance
- [x] 10.2.4 Write test: on CPU-only machine, device is `cpu`
- [x] 10.2.5 Call `get_device()` in Trainer init and in CLI subcommands

---

## PHASE 11 — Results Persistence

### 11.1 Results Saving
- [x] 11.1.1 Create `src/sine_extraction/results.py`
- [x] 11.1.2 Implement `save_metrics(metrics: dict, path: Path) -> None`
  - [x] 11.1.2a `import json`
  - [x] 11.1.2b Ensure parent directory exists
  - [x] 11.1.2c Write JSON with `indent=2`
- [x] 11.1.3 Implement `load_metrics(path: Path) -> dict`
- [x] 11.1.4 Write test: save then load returns same dict
- [x] 11.1.5 Write test: saved file is valid JSON
- [x] 11.1.6 Confirm `ruff check` 0 violations

### 11.2 Training History Saving
- [x] 11.2.1 After training each model, save `history` dict to `artifacts/results/{model_name}_history.json`
- [x] 11.2.2 Write test: history file is created after training
- [x] 11.2.3 Write test: history file contains `train_loss` and `val_loss` lists

---

## PHASE 12 — Documentation & README

### 12.1 README
- [x] 12.1.1 Open `README.md` and write project title and one-line description
- [x] 12.1.2 Add "Prerequisites" section: Python 3.12+, uv
- [x] 12.1.3 Add "Installation" section with `uv sync` command
- [x] 12.1.4 Add "Usage" section with all CLI subcommands and examples
- [x] 12.1.5 Add "Project Structure" section referencing PLAN.md layout
- [x] 12.1.6 Add "Testing" section: `uv run pytest`
- [x] 12.1.7 Add "Linting" section: `uv run ruff check src/ tests/`
- [x] 12.1.8 Add "Mathematical Model" section referencing PRD.md §3

### 12.2 Inline Documentation
- [x] 12.2.1 Add module-level docstring to every `.py` file (one line summary)
- [x] 12.2.2 Add class-level docstring to every class
- [x] 12.2.3 Add function-level docstring to every public function/method
- [x] 12.2.4 All docstrings follow Google style (Args, Returns, Raises)
- [x] 12.2.5 Confirm ruff `D` rules are passing (if enabled)

---

## PHASE 13 — Final Acceptance Checklist

### 13.1 Automated Gates
- [x] 13.1.1 `uv run pytest --cov=sine_extraction --cov-report=term-missing` → ≥ 85% coverage, all tests PASS
- [x] 13.1.2 `uv run ruff check src/ tests/` → 0 violations
- [x] 13.1.3 No Python source file exceeds 150 lines (excluding blanks/comments)

### 13.2 Manual Verification
- [x] 13.2.1 `uv run python -m sine_extraction all` completes without error
- [x] 13.2.2 Open `artifacts/plots/comparison_sample_0.png` — verify 5 panels
- [x] 13.2.3 Open `artifacts/results/metrics.json` — verify 3 model entries
- [x] 13.2.4 Open each checkpoint `.pt` file — verify non-empty (file size > 0)
- [x] 13.2.5 Verify LSTM metrics differ from RNN metrics (distinct architectures)
- [x] 13.2.6 Verify MLP metrics differ from RNN/LSTM metrics

### 13.3 Code Review Checklist
- [x] 13.3.1 No `import pip` or `pip install` anywhere in code or scripts
- [x] 13.3.2 No hardcoded floats or integers in business logic (all from config/constants)
- [x] 13.3.3 All secrets (if any) come from `.env` via `dotenv`
- [x] 13.3.4 All public functions have type hints
- [x] 13.3.5 No file in `src/` uses a wildcard import (`from x import *`)
- [x] 13.3.6 `__all__` is defined in `__init__.py`
- [x] 13.3.7 Tests use fixtures from `conftest.py` rather than repeating setup code
- [x] 13.3.8 No test modifies global state without cleanup

---

## Task Count Summary

| Phase | Task Group | Count |
|---|---|---|
| 0 | Scaffold & Tooling | 60 |
| 1 | Types & Constants | 30 |
| 2 | Config Loader | 17 |
| 3 | Signal Data Pipeline | 52 |
| 4 | Model Architectures | 60 |
| 5 | Training | 27 |
| 6 | Evaluation | 19 |
| 7 | Visualization | 18 |
| 8 | Public API & CLI | 30 |
| 9 | Integration & Coverage | 36 |
| 10 | Seed & Device | 10 |
| 11 | Results Persistence | 10 |
| 12 | Documentation | 16 |
| 13 | Final Acceptance | 18 |
| **Total** | | **~403 tasks** |
