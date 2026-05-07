"""Tests for seed reproducibility utilities (TDD — tests first)."""

from __future__ import annotations

import random
from unittest.mock import patch

import numpy as np
import pytest
import torch


def test_set_all_seeds_importable() -> None:
    """set_all_seeds can be imported from sine_extraction.seeding."""
    from sine_extraction.seeding import set_all_seeds

    assert callable(set_all_seeds)


def test_set_all_seeds_python_random() -> None:
    """set_all_seeds seeds Python random module for reproducibility."""
    from sine_extraction.seeding import set_all_seeds

    set_all_seeds(99)
    val1 = random.random()
    set_all_seeds(99)
    val2 = random.random()
    assert val1 == pytest.approx(val2)


def test_set_all_seeds_numpy() -> None:
    """set_all_seeds seeds NumPy for reproducibility."""
    from sine_extraction.seeding import set_all_seeds

    set_all_seeds(42)
    arr1 = np.random.rand(5)
    set_all_seeds(42)
    arr2 = np.random.rand(5)
    assert np.allclose(arr1, arr2)


def test_set_all_seeds_torch() -> None:
    """set_all_seeds seeds PyTorch for reproducibility."""
    from sine_extraction.seeding import set_all_seeds

    set_all_seeds(7)
    t1 = torch.rand(5)
    set_all_seeds(7)
    t2 = torch.rand(5)
    assert torch.allclose(t1, t2)


def test_different_seeds_produce_different_results() -> None:
    """Different seeds produce different NumPy random values."""
    from sine_extraction.seeding import set_all_seeds

    set_all_seeds(1)
    arr1 = np.random.rand(10)
    set_all_seeds(2)
    arr2 = np.random.rand(10)
    assert not np.allclose(arr1, arr2)


def test_set_all_seeds_returns_none() -> None:
    """set_all_seeds returns None."""
    from sine_extraction.seeding import set_all_seeds

    result = set_all_seeds(0)
    assert result is None


def test_set_all_seeds_calls_cuda_seed_when_available() -> None:
    """set_all_seeds calls cuda.manual_seed_all when CUDA is available."""
    from sine_extraction.seeding import set_all_seeds

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.manual_seed_all"
    ) as mock_cuda_seed:
        set_all_seeds(123)
    mock_cuda_seed.assert_called_with(123)
