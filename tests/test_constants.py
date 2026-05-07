"""Tests for sine_extraction.constants — written before implementation (TDD)."""

from __future__ import annotations

import math

import pytest


def test_known_frequencies_is_list() -> None:
    """KNOWN_FREQUENCIES must be a list of exactly 4 floats."""
    from sine_extraction.constants import KNOWN_FREQUENCIES

    assert isinstance(KNOWN_FREQUENCIES, list)
    assert len(KNOWN_FREQUENCIES) == 4  # noqa: PLR2004
    assert all(isinstance(f, float) for f in KNOWN_FREQUENCIES)


def test_known_frequencies_values() -> None:
    """KNOWN_FREQUENCIES must contain exactly {1, 5, 10, 20} Hz."""
    from sine_extraction.constants import KNOWN_FREQUENCIES

    assert set(KNOWN_FREQUENCIES) == {1.0, 5.0, 10.0, 20.0}


def test_sample_rate() -> None:
    """SAMPLE_RATE must be 200."""
    from sine_extraction.constants import SAMPLE_RATE

    assert SAMPLE_RATE == 200  # noqa: PLR2004
    assert isinstance(SAMPLE_RATE, int)


def test_window_size() -> None:
    """WINDOW_SIZE must be 10."""
    from sine_extraction.constants import WINDOW_SIZE

    assert WINDOW_SIZE == 10  # noqa: PLR2004
    assert isinstance(WINDOW_SIZE, int)


def test_default_seed() -> None:
    """DEFAULT_SEED must be 42."""
    from sine_extraction.constants import DEFAULT_SEED

    assert DEFAULT_SEED == 42  # noqa: PLR2004
    assert isinstance(DEFAULT_SEED, int)


def test_two_pi() -> None:
    """TWO_PI must approximately equal 2 * pi."""
    from sine_extraction.constants import TWO_PI

    assert isinstance(TWO_PI, float)
    assert pytest.approx(2.0 * math.pi, rel=1e-9) == TWO_PI


def test_default_config_path_is_path() -> None:
    """DEFAULT_CONFIG_PATH must be a pathlib.Path pointing at config/config.yaml."""
    from pathlib import Path

    from sine_extraction.constants import DEFAULT_CONFIG_PATH

    assert isinstance(DEFAULT_CONFIG_PATH, Path)
    assert DEFAULT_CONFIG_PATH.parts[-1] == "config.yaml"


def test_checkpoint_filename_template_is_str() -> None:
    """CHECKPOINT_FILENAME_TEMPLATE must be a non-empty string."""
    from sine_extraction.constants import CHECKPOINT_FILENAME_TEMPLATE

    assert isinstance(CHECKPOINT_FILENAME_TEMPLATE, str)
    assert len(CHECKPOINT_FILENAME_TEMPLATE) > 0
    # Must be usable as a format template with model_name and epoch keys
    rendered = CHECKPOINT_FILENAME_TEMPLATE.format(model_name="mlp", epoch=3)
    assert "mlp" in rendered
    assert "3" in rendered
