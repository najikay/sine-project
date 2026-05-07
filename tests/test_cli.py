"""Tests for the Click CLI defined in sine_extraction.__main__."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
from click.testing import CliRunner

from sine_extraction.__main__ import cli

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    """Return a Click test runner that mixes stdout/stderr."""
    return CliRunner()


@pytest.fixture()
def config_path(tmp_path: pytest.TempPathFactory) -> str:
    """Write a minimal config.yaml to a temp dir and return its path string."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """\
seed: 42
signal:
  frequencies: [1.0, 5.0, 10.0, 20.0]
  sample_rate: 200
  window_size: 10
  target_frequency: 10.0
  amplitude: 1.0
  amplitude_jitter_std: 0.05
  phase_jitter_std: 0.1
  noise_std: 0.1
data:
  num_windows: 50
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  batch_size: 16
  shuffle: true
model:
  mlp:
    hidden_sizes: [16, 16]
    activation: relu
  rnn:
    hidden_size: 16
    num_layers: 1
  lstm:
    hidden_size: 16
    num_layers: 1
training:
  learning_rate: 0.01
  epochs: 2
  early_stopping_patience: 2
  checkpoint_dir: {ckpt_dir}
  results_dir: {results_dir}
visualization:
  plots_dir: {plots_dir}
  interactive: false
  num_samples_to_plot: 1
""".format(
            ckpt_dir=str(tmp_path / "checkpoints"),
            results_dir=str(tmp_path / "results"),
            plots_dir=str(tmp_path / "plots"),
        )
    )
    return str(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_help_prints_usage(runner: CliRunner) -> None:
    """--help exits 0 and prints usage information."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_generate_subcommand_exits_zero(
    runner: CliRunner, config_path: str, tmp_path: pytest.TempPathFactory
) -> None:
    """generate subcommand completes without error."""
    result = runner.invoke(
        cli,
        ["--config", config_path, "generate", "--data-dir", str(tmp_path / "data")],
    )
    assert result.exit_code == 0, result.output


def test_generate_creates_data_files(
    runner: CliRunner, config_path: str, tmp_path: pytest.TempPathFactory
) -> None:
    """generate saves X.npy and y.npy to the data directory."""
    from pathlib import Path

    data_dir = tmp_path / "data"
    runner.invoke(
        cli,
        ["--config", config_path, "generate", "--data-dir", str(data_dir)],
    )
    assert (Path(data_dir) / "X.npy").exists()
    assert (Path(data_dir) / "y.npy").exists()


def test_train_subcommand_exits_zero(
    runner: CliRunner, config_path: str, tmp_path: pytest.TempPathFactory
) -> None:
    """train subcommand completes without error after generate."""
    data_dir = str(tmp_path / "data")
    runner.invoke(
        cli, ["--config", config_path, "generate", "--data-dir", data_dir]
    )
    result = runner.invoke(
        cli, ["--config", config_path, "train", "--data-dir", data_dir]
    )
    assert result.exit_code == 0, result.output


def test_evaluate_subcommand_exits_zero(
    runner: CliRunner, config_path: str, tmp_path: pytest.TempPathFactory
) -> None:
    """evaluate subcommand exits 0 after generate + train."""
    data_dir = str(tmp_path / "data")
    runner.invoke(
        cli, ["--config", config_path, "generate", "--data-dir", data_dir]
    )
    runner.invoke(
        cli, ["--config", config_path, "train", "--data-dir", data_dir]
    )
    result = runner.invoke(
        cli, ["--config", config_path, "evaluate", "--data-dir", data_dir]
    )
    assert result.exit_code == 0, result.output


def test_evaluate_creates_metrics_json(
    runner: CliRunner, config_path: str, tmp_path: pytest.TempPathFactory
) -> None:
    """evaluate writes metrics.json to results_dir."""
    import json

    data_dir = str(tmp_path / "data")
    runner.invoke(
        cli, ["--config", config_path, "generate", "--data-dir", data_dir]
    )
    runner.invoke(
        cli, ["--config", config_path, "train", "--data-dir", data_dir]
    )
    runner.invoke(
        cli, ["--config", config_path, "evaluate", "--data-dir", data_dir]
    )
    metrics_path = tmp_path / "results" / "metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert "mlp" in data or "MLP" in data or len(data) >= 1


def test_visualize_subcommand_exits_zero(
    runner: CliRunner, config_path: str, tmp_path: pytest.TempPathFactory
) -> None:
    """visualize subcommand exits 0 after generate + train."""
    data_dir = str(tmp_path / "data")
    runner.invoke(
        cli, ["--config", config_path, "generate", "--data-dir", data_dir]
    )
    runner.invoke(
        cli, ["--config", config_path, "train", "--data-dir", data_dir]
    )
    result = runner.invoke(
        cli, ["--config", config_path, "visualize", "--data-dir", data_dir]
    )
    assert result.exit_code == 0, result.output


def test_visualize_creates_png(
    runner: CliRunner, config_path: str, tmp_path: pytest.TempPathFactory
) -> None:
    """visualize saves at least one PNG file."""
    from pathlib import Path

    data_dir = str(tmp_path / "data")
    runner.invoke(
        cli, ["--config", config_path, "generate", "--data-dir", data_dir]
    )
    runner.invoke(
        cli, ["--config", config_path, "train", "--data-dir", data_dir]
    )
    runner.invoke(
        cli, ["--config", config_path, "visualize", "--data-dir", data_dir]
    )
    plots_dir = tmp_path / "plots"
    pngs = list(Path(plots_dir).glob("*.png"))
    assert len(pngs) >= 1


def test_all_subcommand_exits_zero(
    runner: CliRunner, config_path: str, tmp_path: pytest.TempPathFactory
) -> None:
    """all subcommand runs the full pipeline without error."""
    data_dir = str(tmp_path / "data")
    result = runner.invoke(
        cli, ["--config", config_path, "all", "--data-dir", data_dir]
    )
    assert result.exit_code == 0, result.output
