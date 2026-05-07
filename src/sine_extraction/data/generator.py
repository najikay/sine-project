"""Synthetic sine-wave signal generation with jitter and noise."""

from __future__ import annotations

import numpy as np

from sine_extraction.constants import TWO_PI
from sine_extraction.types import SignalConfig


class SignalGenerator:
    """Generate synthetic noisy-mixed windows and clean target sine waves.

    Args:
        config: Signal parameters (frequencies, sample_rate, etc.).
        seed: Integer seed for the internal numpy RNG.
    """

    def __init__(self, config: SignalConfig, seed: int) -> None:
        self._cfg = config
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _time_axis(self, start_sample: int) -> np.ndarray:
        """Return the time axis for one window starting at *start_sample*."""
        cfg = self._cfg
        return (
            np.arange(cfg.window_size, dtype=np.float64) / cfg.sample_rate
            + start_sample / cfg.sample_rate
        )

    def _sample_phase(self) -> float:
        """Draw a random phase uniformly from [0, 2π)."""
        return float(self._rng.uniform(0.0, TWO_PI))

    def _sample_amplitude(self) -> float:
        """Draw amplitude from N(amplitude, amplitude_jitter_std)."""
        cfg = self._cfg
        return float(self._rng.normal(cfg.amplitude, cfg.amplitude_jitter_std))

    # ------------------------------------------------------------------
    # Public generation methods
    # ------------------------------------------------------------------

    def generate_pure_sine(
        self,
        frequency: float,
        start_sample: int,
        *,
        amplitude: float | None = None,
        phase: float | None = None,
    ) -> np.ndarray:
        """Return a single sine wave window.

        Args:
            frequency: Sine frequency in Hz.
            start_sample: Index of the first sample in the window.
            amplitude: Amplitude override (uses config default when *None*).
            phase: Phase offset override (uses 0 when *None*).

        Returns:
            Float64 array of shape ``(window_size,)``.
        """
        amp = self._cfg.amplitude if amplitude is None else amplitude
        phi = 0.0 if phase is None else phase
        t = self._time_axis(start_sample)
        return (amp * np.sin(TWO_PI * frequency * t + phi)).astype(np.float32)

    def generate_mixed_window(
        self, start_sample: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate one noisy mixed window and its clean target.

        Args:
            start_sample: Window start index.

        Returns:
            Tuple of ``(noisy_mix, clean_target)`` each shape ``(window_size,)``.
        """
        cfg = self._cfg
        mix = np.zeros(cfg.window_size, dtype=np.float64)
        target_phase: float = 0.0

        t = self._time_axis(start_sample)
        for freq in cfg.frequencies:
            amp = self._sample_amplitude()
            phase = self._sample_phase()
            if freq == cfg.target_frequency:
                target_phase = phase  # reuse the same phase for the target
            mix += amp * np.sin(TWO_PI * freq * t + phase)

        # White Gaussian noise
        mix += self._rng.normal(0.0, cfg.noise_std, size=cfg.window_size)

        # Clean target: same t and phase as the target-frequency component in the mix
        clean = cfg.amplitude * np.sin(TWO_PI * cfg.target_frequency * t + target_phase)

        return mix.astype(np.float32), clean.astype(np.float32)

    def generate_dataset(
        self, num_windows: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate *num_windows* (noisy_mix, clean_target) pairs.

        Args:
            num_windows: Number of windows to generate.

        Returns:
            Tuple of arrays ``(X, y)`` each of shape
            ``(num_windows, window_size)`` and dtype ``float32``.
        """
        cfg = self._cfg
        # Max start sample so that the window fits within a 60-second signal
        max_start = max(1, cfg.sample_rate * 60 - cfg.window_size)

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        for _ in range(num_windows):
            start = int(self._rng.integers(0, max_start))
            x_i, y_i = self.generate_mixed_window(start)
            X_list.append(x_i)
            y_list.append(y_i)

        X = np.stack(X_list).astype(np.float32)
        y = np.stack(y_list).astype(np.float32)
        return X, y
