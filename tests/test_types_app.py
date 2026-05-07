"""Tests for the AppConfig root composite — split from test_types.py (150-line rule)."""

from __future__ import annotations


def test_app_config_instantiation() -> None:
    """AppConfig must compose all sub-configs under a single root."""
    from sine_extraction.types import (
        AppConfig,
        DataConfig,
        LSTMConfig,
        MLPConfig,
        ModelConfig,
        RNNConfig,
        SignalConfig,
        TrainingConfig,
        VisualizationConfig,
    )

    app_cfg = AppConfig(
        seed=42,
        signal=SignalConfig(
            frequencies=[1.0, 5.0, 10.0, 20.0],
            sample_rate=200,
            window_size=10,
            target_frequency=10.0,
            amplitude=1.0,
            amplitude_jitter_std=0.05,
            phase_jitter_std=0.1,
            noise_std=0.1,
        ),
        data=DataConfig(
            num_windows=1000,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=64,
            shuffle=True,
        ),
        model=ModelConfig(
            mlp=MLPConfig(hidden_sizes=[64], activation="relu"),
            rnn=RNNConfig(hidden_size=32, num_layers=1),
            lstm=LSTMConfig(hidden_size=32, num_layers=1),
        ),
        training=TrainingConfig(
            learning_rate=0.001,
            epochs=10,
            early_stopping_patience=3,
            checkpoint_dir="artifacts/checkpoints",
            results_dir="artifacts/results",
        ),
        visualization=VisualizationConfig(
            plots_dir="artifacts/plots",
            interactive=False,
            num_samples_to_plot=3,
        ),
    )
    assert app_cfg.seed == 42  # noqa: PLR2004
    assert app_cfg.signal.sample_rate == 200  # noqa: PLR2004
    assert app_cfg.data.batch_size == 64  # noqa: PLR2004
    assert app_cfg.model.mlp.activation == "relu"
    assert app_cfg.training.early_stopping_patience == 3  # noqa: PLR2004
    assert app_cfg.visualization.interactive is False
