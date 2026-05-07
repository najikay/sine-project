"""Load and parse config.yaml into typed AppConfig dataclasses."""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

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

load_dotenv()

_REQUIRED_TOP_KEYS = ("seed", "signal", "data", "model", "training", "visualization")


def _require(mapping: dict, key: str, context: str) -> object:
    """Return mapping[key] or raise ValueError with a clear message."""
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {context}")
    return mapping[key]


def _build_signal(raw: dict) -> SignalConfig:
    s = raw
    return SignalConfig(
        frequencies=[float(f) for f in _require(s, "frequencies", "signal")],
        sample_rate=int(_require(s, "sample_rate", "signal")),
        window_size=int(_require(s, "window_size", "signal")),
        target_frequency=float(_require(s, "target_frequency", "signal")),
        amplitude=float(_require(s, "amplitude", "signal")),
        amplitude_jitter_std=float(_require(s, "amplitude_jitter_std", "signal")),
        phase_jitter_std=float(_require(s, "phase_jitter_std", "signal")),
        noise_std=float(_require(s, "noise_std", "signal")),
    )


def _build_data(raw: dict) -> DataConfig:
    d = raw
    return DataConfig(
        num_windows=int(_require(d, "num_windows", "data")),
        train_ratio=float(_require(d, "train_ratio", "data")),
        val_ratio=float(_require(d, "val_ratio", "data")),
        test_ratio=float(_require(d, "test_ratio", "data")),
        batch_size=int(_require(d, "batch_size", "data")),
        shuffle=bool(_require(d, "shuffle", "data")),
    )


def _build_model(raw: dict) -> ModelConfig:
    m = raw
    mlp_raw = _require(m, "mlp", "model")
    rnn_raw = _require(m, "rnn", "model")
    lstm_raw = _require(m, "lstm", "model")
    mlp = MLPConfig(
        hidden_sizes=[int(h) for h in mlp_raw["hidden_sizes"]],
        activation=str(mlp_raw["activation"]),
    )
    rnn = RNNConfig(
        hidden_size=int(rnn_raw["hidden_size"]),
        num_layers=int(rnn_raw["num_layers"]),
        bidirectional=bool(rnn_raw.get("bidirectional", False)),
        nonlinearity=str(rnn_raw.get("nonlinearity", "relu")),
    )
    lstm = LSTMConfig(
        hidden_size=int(lstm_raw["hidden_size"]),
        num_layers=int(lstm_raw["num_layers"]),
        bidirectional=bool(lstm_raw.get("bidirectional", False)),
        dropout=float(lstm_raw.get("dropout", 0.2)),
    )
    return ModelConfig(mlp=mlp, rnn=rnn, lstm=lstm)


def _build_training(raw: dict) -> TrainingConfig:
    t = raw
    return TrainingConfig(
        learning_rate=float(_require(t, "learning_rate", "training")),
        epochs=int(_require(t, "epochs", "training")),
        early_stopping_patience=int(
            _require(t, "early_stopping_patience", "training")
        ),
        checkpoint_dir=str(_require(t, "checkpoint_dir", "training")),
        results_dir=str(_require(t, "results_dir", "training")),
        use_scheduler=bool(t.get("use_scheduler", False)),
        grad_clip_max_norm=float(t.get("grad_clip_max_norm", 1.0)),
    )


def _build_visualization(raw: dict) -> VisualizationConfig:
    v = raw
    return VisualizationConfig(
        plots_dir=str(_require(v, "plots_dir", "visualization")),
        interactive=bool(_require(v, "interactive", "visualization")),
        num_samples_to_plot=int(_require(v, "num_samples_to_plot", "visualization")),
    )


def load_config(path: Path) -> AppConfig:
    """Parse *path* as YAML and return a fully typed :class:`AppConfig`.

    Args:
        path: Location of the YAML configuration file.

    Returns:
        Populated :class:`AppConfig` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required keys are absent from the YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as fh:
        raw: dict = yaml.safe_load(fh) or {}

    for key in _REQUIRED_TOP_KEYS:
        _require(raw, key, "top-level config")

    return AppConfig(
        seed=int(raw["seed"]),
        signal=_build_signal(raw["signal"]),
        data=_build_data(raw["data"]),
        model=_build_model(raw["model"]),
        training=_build_training(raw["training"]),
        visualization=_build_visualization(raw["visualization"]),
    )
