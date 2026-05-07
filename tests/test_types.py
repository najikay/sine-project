"""Tests for sine_extraction.types — written before implementation (TDD)."""

from __future__ import annotations


def test_signal_config_instantiation() -> None:
    """SignalConfig dataclass must accept all documented fields."""
    from sine_extraction.types import SignalConfig

    cfg = SignalConfig(
        frequencies=[1.0, 5.0, 10.0, 20.0],
        sample_rate=200,
        window_size=10,
        target_frequency=10.0,
        amplitude=1.0,
        amplitude_jitter_std=0.05,
        phase_jitter_std=0.1,
        noise_std=0.1,
    )
    assert cfg.sample_rate == 200  # noqa: PLR2004
    assert cfg.window_size == 10  # noqa: PLR2004
    assert cfg.target_frequency == 10.0  # noqa: PLR2004


def test_data_config_instantiation() -> None:
    """DataConfig dataclass must accept all documented fields."""
    from sine_extraction.types import DataConfig

    cfg = DataConfig(
        num_windows=1000,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=64,
        shuffle=True,
    )
    assert cfg.num_windows == 1000  # noqa: PLR2004
    assert cfg.shuffle is True


def test_mlp_config_instantiation() -> None:
    """MLPConfig dataclass must accept hidden_sizes and activation."""
    from sine_extraction.types import MLPConfig

    cfg = MLPConfig(hidden_sizes=[64, 128, 64], activation="relu")
    assert cfg.activation == "relu"
    assert cfg.hidden_sizes == [64, 128, 64]


def test_rnn_config_instantiation() -> None:
    """RNNConfig dataclass must accept hidden_size and num_layers."""
    from sine_extraction.types import RNNConfig

    cfg = RNNConfig(hidden_size=64, num_layers=2)
    assert cfg.hidden_size == 64  # noqa: PLR2004
    assert cfg.num_layers == 2  # noqa: PLR2004


def test_lstm_config_instantiation() -> None:
    """LSTMConfig dataclass must accept hidden_size and num_layers."""
    from sine_extraction.types import LSTMConfig

    cfg = LSTMConfig(hidden_size=64, num_layers=2)
    assert cfg.hidden_size == 64  # noqa: PLR2004
    assert cfg.num_layers == 2  # noqa: PLR2004


def test_model_config_instantiation() -> None:
    """ModelConfig must compose MLPConfig, RNNConfig, LSTMConfig."""
    from sine_extraction.types import LSTMConfig, MLPConfig, ModelConfig, RNNConfig

    cfg = ModelConfig(
        mlp=MLPConfig(hidden_sizes=[64], activation="relu"),
        rnn=RNNConfig(hidden_size=32, num_layers=1),
        lstm=LSTMConfig(hidden_size=32, num_layers=1),
    )
    assert cfg.mlp.hidden_sizes == [64]
    assert cfg.rnn.hidden_size == 32  # noqa: PLR2004
    assert cfg.lstm.num_layers == 1


def test_training_config_instantiation() -> None:
    """TrainingConfig must accept all documented training hyperparameters."""
    from sine_extraction.types import TrainingConfig

    cfg = TrainingConfig(
        learning_rate=0.001,
        epochs=100,
        early_stopping_patience=10,
        checkpoint_dir="artifacts/checkpoints",
        results_dir="artifacts/results",
    )
    assert cfg.learning_rate == 0.001  # noqa: PLR2004
    assert cfg.epochs == 100  # noqa: PLR2004


def test_visualization_config_instantiation() -> None:
    """VisualizationConfig must accept all documented fields."""
    from sine_extraction.types import VisualizationConfig

    cfg = VisualizationConfig(
        plots_dir="artifacts/plots",
        interactive=False,
        num_samples_to_plot=3,
    )
    assert cfg.interactive is False
    assert cfg.num_samples_to_plot == 3  # noqa: PLR2004


# AppConfig composition test lives in test_types_app.py to respect 150-line limit
