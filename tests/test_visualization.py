"""Tests for visualization/plotter.py — uses Agg backend to avoid GUI hangs."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Must be set before any other matplotlib import

from sine_extraction.types import VisualizationConfig  # noqa: E402
from sine_extraction.visualization.plotter import ComparisonPlotter  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vis_config(tmp_path: pytest.TempPathFactory) -> VisualizationConfig:
    """VisualizationConfig pointing at a temp directory."""
    return VisualizationConfig(
        plots_dir=str(tmp_path / "plots"),
        interactive=False,
        num_samples_to_plot=3,
    )


@pytest.fixture()
def signal_arrays() -> dict[str, np.ndarray]:
    """Five fake signal arrays of shape (10,)."""
    t = np.linspace(0, 1, 10, dtype=np.float32)
    return {
        "noisy": (np.sin(2 * np.pi * t) + 0.1 * np.random.randn(10)).astype(
            np.float32
        ),
        "pure": np.sin(2 * np.pi * t).astype(np.float32),
        "mlp_pred": (np.sin(2 * np.pi * t) * 0.9).astype(np.float32),
        "rnn_pred": (np.sin(2 * np.pi * t) * 0.85).astype(np.float32),
        "lstm_pred": (np.sin(2 * np.pi * t) * 0.95).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_comparison_plotter_instantiates(
    vis_config: VisualizationConfig,
) -> None:
    """ComparisonPlotter can be constructed without error."""
    plotter = ComparisonPlotter(vis_config)
    assert plotter is not None


def test_plots_dir_created_on_init(
    vis_config: VisualizationConfig,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """plots_dir is created during __init__ even if it did not exist."""
    from pathlib import Path

    ComparisonPlotter(vis_config)
    assert Path(vis_config.plots_dir).is_dir()


def test_plot_runs_without_exception(
    vis_config: VisualizationConfig,
    signal_arrays: dict[str, np.ndarray],
) -> None:
    """plot() completes without raising any exception."""
    plotter = ComparisonPlotter(vis_config)
    plotter.plot(
        noisy=signal_arrays["noisy"],
        pure=signal_arrays["pure"],
        mlp_pred=signal_arrays["mlp_pred"],
        rnn_pred=signal_arrays["rnn_pred"],
        lstm_pred=signal_arrays["lstm_pred"],
        sample_idx=0,
    )


def test_plot_stores_figure(
    vis_config: VisualizationConfig,
    signal_arrays: dict[str, np.ndarray],
) -> None:
    """After plot(), self.fig is a matplotlib Figure object."""
    import matplotlib.figure

    plotter = ComparisonPlotter(vis_config)
    plotter.plot(
        noisy=signal_arrays["noisy"],
        pure=signal_arrays["pure"],
        mlp_pred=signal_arrays["mlp_pred"],
        rnn_pred=signal_arrays["rnn_pred"],
        lstm_pred=signal_arrays["lstm_pred"],
    )
    assert isinstance(plotter.fig, matplotlib.figure.Figure)


def test_plot_has_six_axes(
    vis_config: VisualizationConfig,
    signal_arrays: dict[str, np.ndarray],
) -> None:
    """The figure produced by plot() has exactly 6 subplots."""
    plotter = ComparisonPlotter(vis_config)
    plotter.plot(
        noisy=signal_arrays["noisy"],
        pure=signal_arrays["pure"],
        mlp_pred=signal_arrays["mlp_pred"],
        rnn_pred=signal_arrays["rnn_pred"],
        lstm_pred=signal_arrays["lstm_pred"],
    )
    assert len(plotter.fig.axes) == 6


def test_save_creates_file(
    vis_config: VisualizationConfig,
    signal_arrays: dict[str, np.ndarray],
    tmp_path: pytest.TempPathFactory,
) -> None:
    """save() creates a non-empty PNG file at the expected path."""
    from pathlib import Path

    plotter = ComparisonPlotter(vis_config)
    plotter.plot(
        noisy=signal_arrays["noisy"],
        pure=signal_arrays["pure"],
        mlp_pred=signal_arrays["mlp_pred"],
        rnn_pred=signal_arrays["rnn_pred"],
        lstm_pred=signal_arrays["lstm_pred"],
    )
    saved = plotter.save("test_output.png")
    assert Path(saved).exists()
    assert Path(saved).stat().st_size > 0


def test_save_returns_path(
    vis_config: VisualizationConfig,
    signal_arrays: dict[str, np.ndarray],
) -> None:
    """save() returns a Path object pointing inside plots_dir."""
    from pathlib import Path

    plotter = ComparisonPlotter(vis_config)
    plotter.plot(
        noisy=signal_arrays["noisy"],
        pure=signal_arrays["pure"],
        mlp_pred=signal_arrays["mlp_pred"],
        rnn_pred=signal_arrays["rnn_pred"],
        lstm_pred=signal_arrays["lstm_pred"],
    )
    result = plotter.save("result.png")
    assert isinstance(result, Path)
    assert result.name == "result.png"
    assert str(vis_config.plots_dir) in str(result)
