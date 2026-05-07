"""Tests for config_loader.load_config — written BEFORE implementation (TDD)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sine_extraction.types import AppConfig

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

MINIMAL_YAML: dict = {
    "seed": 42,
    "signal": {
        "frequencies": [1, 5, 10, 20],
        "sample_rate": 200,
        "window_size": 10,
        "target_frequency": 10,
        "amplitude": 1.0,
        "amplitude_jitter_std": 0.05,
        "phase_jitter_std": 0.1,
        "noise_std": 0.1,
    },
    "data": {
        "num_windows": 10000,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "batch_size": 64,
        "shuffle": True,
    },
    "model": {
        "mlp": {"hidden_sizes": [64, 128, 64], "activation": "relu"},
        "rnn": {"hidden_size": 64, "num_layers": 2},
        "lstm": {"hidden_size": 64, "num_layers": 2},
    },
    "training": {
        "learning_rate": 0.001,
        "epochs": 100,
        "early_stopping_patience": 10,
        "checkpoint_dir": "artifacts/checkpoints",
        "results_dir": "artifacts/results",
    },
    "visualization": {
        "plots_dir": "artifacts/plots",
        "interactive": False,
        "num_samples_to_plot": 3,
    },
}


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Write a minimal valid config.yaml to a temp directory."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(MINIMAL_YAML))
    return path


# ---------------------------------------------------------------------------
# 2.1.3  load_config returns AppConfig
# ---------------------------------------------------------------------------


def test_load_config_returns_app_config(config_file: Path) -> None:
    from sine_extraction.config_loader import load_config

    cfg = load_config(config_file)
    assert isinstance(cfg, AppConfig)


# ---------------------------------------------------------------------------
# 2.1.4  config.seed equals YAML value
# ---------------------------------------------------------------------------


def test_load_config_seed(config_file: Path) -> None:
    from sine_extraction.config_loader import load_config

    cfg = load_config(config_file)
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# 2.1.5  config.signal.sample_rate equals 200
# ---------------------------------------------------------------------------


def test_load_config_signal_sample_rate(config_file: Path) -> None:
    from sine_extraction.config_loader import load_config

    cfg = load_config(config_file)
    assert cfg.signal.sample_rate == 200


# ---------------------------------------------------------------------------
# 2.1.6  config.signal.frequencies equals [1.0, 5.0, 10.0, 20.0]
# ---------------------------------------------------------------------------


def test_load_config_signal_frequencies(config_file: Path) -> None:
    from sine_extraction.config_loader import load_config

    cfg = load_config(config_file)
    assert cfg.signal.frequencies == [1.0, 5.0, 10.0, 20.0]


# ---------------------------------------------------------------------------
# 2.1.7  config.data.batch_size equals 64
# ---------------------------------------------------------------------------


def test_load_config_data_batch_size(config_file: Path) -> None:
    from sine_extraction.config_loader import load_config

    cfg = load_config(config_file)
    assert cfg.data.batch_size == 64


# ---------------------------------------------------------------------------
# 2.1.8  config.model.mlp.hidden_sizes is a list of ints
# ---------------------------------------------------------------------------


def test_load_config_mlp_hidden_sizes(config_file: Path) -> None:
    from sine_extraction.config_loader import load_config

    cfg = load_config(config_file)
    assert isinstance(cfg.model.mlp.hidden_sizes, list)
    assert all(isinstance(h, int) for h in cfg.model.mlp.hidden_sizes)


# ---------------------------------------------------------------------------
# 2.1.9  FileNotFoundError for missing path
# ---------------------------------------------------------------------------


def test_load_config_missing_file(tmp_path: Path) -> None:
    from sine_extraction.config_loader import load_config

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# 2.1.10 ValueError for invalid YAML structure (missing required key)
# ---------------------------------------------------------------------------


def test_load_config_invalid_structure(tmp_path: Path) -> None:
    from sine_extraction.config_loader import load_config

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.dump({"seed": 42}))  # missing signal, data, …

    with pytest.raises(ValueError):
        load_config(bad_path)
