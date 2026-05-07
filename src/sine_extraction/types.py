"""Typed dataclasses for all configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SignalConfig:
    """Parameters that define the synthetic signal."""

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
    """Dataset generation and DataLoader parameters."""

    num_windows: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    batch_size: int
    shuffle: bool


@dataclass
class MLPConfig:
    """Fully connected network architecture settings."""

    hidden_sizes: list[int] = field(default_factory=lambda: [64, 128, 64])
    activation: str = "relu"


@dataclass
class RNNConfig:
    """Vanilla RNN architecture settings."""

    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = False


@dataclass
class LSTMConfig:
    """LSTM architecture settings."""

    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = False


@dataclass
class ModelConfig:
    """Container for all three model architecture configs."""

    mlp: MLPConfig
    rnn: RNNConfig
    lstm: LSTMConfig


@dataclass
class TrainingConfig:
    """Optimizer and training loop hyperparameters."""

    learning_rate: float
    epochs: int
    early_stopping_patience: int
    checkpoint_dir: str
    results_dir: str
    use_scheduler: bool = False
    grad_clip_max_norm: float = 1.0


@dataclass
class VisualizationConfig:
    """Plot output settings."""

    plots_dir: str
    interactive: bool
    num_samples_to_plot: int


@dataclass
class AppConfig:
    """Root configuration object — aggregates all sub-configs."""

    seed: int
    signal: SignalConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    visualization: VisualizationConfig
