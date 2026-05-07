"""Signal comparison plotter — renders 6-panel waveform figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sine_extraction.types import VisualizationConfig


class ComparisonPlotter:
    """Renders side-by-side comparison of noisy input, ground truth, and predictions.

    Args:
        config: Visualization settings (plots_dir, interactive flag, etc.).
    """

    def __init__(self, config: VisualizationConfig) -> None:
        self._config = config
        Path(config.plots_dir).mkdir(parents=True, exist_ok=True)
        if not config.interactive:
            matplotlib.use("Agg")
        self.fig: matplotlib.figure.Figure | None = None

    def plot(
        self,
        noisy: np.ndarray,
        pure: np.ndarray,
        mlp_pred: np.ndarray,
        rnn_pred: np.ndarray,
        lstm_pred: np.ndarray,
        sample_idx: int = 0,
    ) -> None:
        """Create a 6-panel comparison figure and store it in ``self.fig``.

        Panels 1-5 show individual signals.  Panel 6 is an overlay of all
        three model predictions against the ground-truth on the same axes.

        Args:
            noisy: Noisy mixed input signal, shape ``(window_size,)``.
            pure: Ground-truth clean sine, shape ``(window_size,)``.
            mlp_pred: MLP model prediction, shape ``(window_size,)``.
            rnn_pred: RNN model prediction, shape ``(window_size,)``.
            lstm_pred: LSTM model prediction, shape ``(window_size,)``.
            sample_idx: Sample index shown in the figure title.
        """
        fig, axes = plt.subplots(6, 1, figsize=(10, 14))

        panels = [
            (axes[0], noisy, "Noisy Mixed Input"),
            (axes[1], pure, "Ground Truth Pure Sine"),
            (axes[2], mlp_pred, "MLP Prediction"),
            (axes[3], rnn_pred, "RNN Prediction"),
            (axes[4], lstm_pred, "LSTM Prediction"),
        ]

        for ax, signal, title in panels:
            ax.plot(signal)
            ax.set_title(title)
            ax.set_ylabel("Amplitude")

        # Overlay panel: all predictions vs ground truth on one axes
        axes[5].plot(pure, label="Ground Truth", linewidth=2, color="black")
        axes[5].plot(mlp_pred, label="MLP", linestyle="--", alpha=0.8)
        axes[5].plot(rnn_pred, label="RNN", linestyle="-.", alpha=0.8)
        axes[5].plot(lstm_pred, label="LSTM", linestyle=":", alpha=0.8)
        axes[5].set_title("Overlay: All Predictions vs Ground Truth")
        axes[5].set_ylabel("Amplitude")
        axes[5].legend(loc="upper right")

        axes[-1].set_xlabel("Sample Index")
        fig.suptitle(f"Signal Comparison \u2014 Sample {sample_idx}")
        plt.tight_layout()

        self.fig = fig

        if self._config.interactive:
            plt.show()

    def save(self, filename: str) -> Path:
        """Save the current figure to ``plots_dir / filename``.

        Args:
            filename: Output file name (e.g. ``"comparison_0.png"``).

        Returns:
            Absolute path of the saved file.

        Raises:
            AssertionError: If :meth:`plot` has not been called yet.
        """
        assert self.fig is not None, "Call plot() before save()."
        path = Path(self._config.plots_dir) / filename
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        return path
