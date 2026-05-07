"""Tests for SignalGenerator — written BEFORE implementation (TDD)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sine_extraction.types import SignalConfig


@pytest.fixture()
def sig_cfg() -> SignalConfig:
    """Default SignalConfig matching config.yaml."""
    return SignalConfig(
        frequencies=[1.0, 5.0, 10.0, 20.0],
        sample_rate=200,
        window_size=10,
        target_frequency=10.0,
        amplitude=1.0,
        amplitude_jitter_std=0.05,
        phase_jitter_std=0.1,
        noise_std=0.1,
    )


# ---------------------------------------------------------------------------
# 3.1.3  generate_dataset returns (X, y) with correct shapes
# ---------------------------------------------------------------------------


def test_generate_dataset_shapes(sig_cfg: SignalConfig) -> None:
    from sine_extraction.data.generator import SignalGenerator

    gen = SignalGenerator(sig_cfg, seed=42)
    X, y = gen.generate_dataset(50)
    assert X.shape == (50, 10)
    assert y.shape == (50, 10)


# ---------------------------------------------------------------------------
# 3.1.4 / 3.1.5  dtype is float32
# ---------------------------------------------------------------------------


def test_generate_dataset_dtype(sig_cfg: SignalConfig) -> None:
    from sine_extraction.data.generator import SignalGenerator

    gen = SignalGenerator(sig_cfg, seed=42)
    X, y = gen.generate_dataset(20)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


# ---------------------------------------------------------------------------
# 3.1.6  target amplitude bounded
# ---------------------------------------------------------------------------


def test_target_amplitude_bounded(sig_cfg: SignalConfig) -> None:
    from sine_extraction.data.generator import SignalGenerator

    gen = SignalGenerator(sig_cfg, seed=0)
    _, y = gen.generate_dataset(500)
    bound = sig_cfg.amplitude + 3 * sig_cfg.amplitude_jitter_std
    assert float(np.max(np.abs(y))) <= bound + 1e-6


# ---------------------------------------------------------------------------
# 3.1.7  same seed → identical output
# ---------------------------------------------------------------------------


def test_reproducible_with_same_seed(sig_cfg: SignalConfig) -> None:
    from sine_extraction.data.generator import SignalGenerator

    X1, y1 = SignalGenerator(sig_cfg, seed=7).generate_dataset(30)
    X2, y2 = SignalGenerator(sig_cfg, seed=7).generate_dataset(30)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# 3.1.8  different seeds → different output
# ---------------------------------------------------------------------------


def test_different_seeds_differ(sig_cfg: SignalConfig) -> None:
    from sine_extraction.data.generator import SignalGenerator

    X1, _ = SignalGenerator(sig_cfg, seed=1).generate_dataset(30)
    X2, _ = SignalGenerator(sig_cfg, seed=2).generate_dataset(30)
    assert not np.array_equal(X1, X2)


# ---------------------------------------------------------------------------
# 3.1.9  generate_pure_sine returns correct length
# ---------------------------------------------------------------------------


def test_pure_sine_length(sig_cfg: SignalConfig) -> None:
    from sine_extraction.data.generator import SignalGenerator

    gen = SignalGenerator(sig_cfg, seed=42)
    sine = gen.generate_pure_sine(frequency=10.0, start_sample=0)
    assert len(sine) == sig_cfg.window_size


# ---------------------------------------------------------------------------
# 3.1.10 zero-jitter pure sine matches exact sine formula
# ---------------------------------------------------------------------------


def test_pure_sine_no_jitter(sig_cfg: SignalConfig) -> None:
    from sine_extraction.data.generator import SignalGenerator

    no_jitter_cfg = SignalConfig(
        frequencies=sig_cfg.frequencies,
        sample_rate=sig_cfg.sample_rate,
        window_size=sig_cfg.window_size,
        target_frequency=sig_cfg.target_frequency,
        amplitude=1.0,
        amplitude_jitter_std=0.0,
        phase_jitter_std=0.0,
        noise_std=0.0,
    )
    gen = SignalGenerator(no_jitter_cfg, seed=42)
    freq = 10.0
    start = 0
    result = gen.generate_pure_sine(frequency=freq, start_sample=start)

    t = np.arange(no_jitter_cfg.window_size) / no_jitter_cfg.sample_rate
    expected = np.sin(2.0 * math.pi * freq * t).astype(np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 3.1.11 mixed signal correlates with each frequency component
# ---------------------------------------------------------------------------


def test_mixed_signal_has_all_frequencies(sig_cfg: SignalConfig) -> None:
    """Verify each frequency contributes energy to the mixed signal.

    Uses FFT power spectrum on the flattened dataset because the time-domain
    correlation approach fails for 20 Hz: with window_size=10 and
    sample_rate=200 each window contains exactly one 20 Hz period, so random
    per-window phases cancel out in cross-correlation.  Checking that the
    power at each target frequency exceeds twice the mean spectral power is
    mathematically sound across all four frequencies.
    """
    from sine_extraction.data.generator import SignalGenerator

    gen = SignalGenerator(sig_cfg, seed=42)
    X, _ = gen.generate_dataset(200)
    mixed_flat = X.flatten()

    # FFT power spectrum of the flattened signal
    fft_vals = np.fft.rfft(mixed_flat)
    freqs = np.fft.rfftfreq(len(mixed_flat), d=1.0 / sig_cfg.sample_rate)
    power = np.abs(fft_vals) ** 2
    mean_power = float(np.mean(power))

    for freq in sig_cfg.frequencies:
        idx = int(np.argmin(np.abs(freqs - freq)))
        ratio = power[idx] / mean_power
        assert ratio > 2.0, (
            f"Frequency {freq} Hz not detected in mixed signal "
            f"(power ratio {ratio:.4f} <= 2.0)"
        )
