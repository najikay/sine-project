"""Tests for results persistence (TDD — tests first)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_save_metrics_importable() -> None:
    """save_metrics can be imported from sine_extraction.results."""
    from sine_extraction.results import save_metrics

    assert callable(save_metrics)


def test_load_metrics_importable() -> None:
    """load_metrics can be imported from sine_extraction.results."""
    from sine_extraction.results import load_metrics

    assert callable(load_metrics)


def test_save_creates_file(tmp_path: Path) -> None:
    """save_metrics creates a file at the given path."""
    from sine_extraction.results import save_metrics

    metrics = {"mse": 0.1, "mae": 0.05, "r2": 0.95}
    out = tmp_path / "metrics.json"
    save_metrics(metrics, out)
    assert out.exists()


def test_saved_file_is_valid_json(tmp_path: Path) -> None:
    """The file written by save_metrics is valid JSON."""
    from sine_extraction.results import save_metrics

    metrics = {"mse": 0.1, "mae": 0.05, "r2": 0.95}
    out = tmp_path / "metrics.json"
    save_metrics(metrics, out)
    content = out.read_text()
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


def test_save_then_load_returns_same_dict(tmp_path: Path) -> None:
    """save_metrics then load_metrics returns the original dict."""
    from sine_extraction.results import load_metrics, save_metrics

    metrics = {"mse": 0.123, "mae": 0.456, "r2": 0.789}
    out = tmp_path / "results.json"
    save_metrics(metrics, out)
    loaded = load_metrics(out)
    assert loaded == metrics


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    """save_metrics creates parent directories if they don't exist."""
    from sine_extraction.results import save_metrics

    metrics = {"loss": 0.5}
    out = tmp_path / "nested" / "deep" / "metrics.json"
    save_metrics(metrics, out)
    assert out.exists()


def test_save_with_nested_dict(tmp_path: Path) -> None:
    """save_metrics handles nested dicts (model results)."""
    from sine_extraction.results import load_metrics, save_metrics

    metrics = {
        "mlp": {"mse": 0.1, "mae": 0.05, "r2": 0.95},
        "rnn": {"mse": 0.2, "mae": 0.1, "r2": 0.85},
    }
    out = tmp_path / "all_metrics.json"
    save_metrics(metrics, out)
    loaded = load_metrics(out)
    assert loaded["mlp"]["mse"] == pytest.approx(0.1)
    assert loaded["rnn"]["r2"] == pytest.approx(0.85)


def test_save_metrics_with_indent(tmp_path: Path) -> None:
    """save_metrics writes indented JSON (human-readable)."""
    from sine_extraction.results import save_metrics

    metrics = {"a": 1}
    out = tmp_path / "m.json"
    save_metrics(metrics, out)
    content = out.read_text()
    assert "\n" in content  # indented JSON has newlines


def test_history_save_load_roundtrip(tmp_path: Path) -> None:
    """save/load roundtrip for training history dicts."""
    from sine_extraction.results import load_metrics, save_metrics

    history = {"train_loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.7]}
    out = tmp_path / "history.json"
    save_metrics(history, out)
    loaded = load_metrics(out)
    assert loaded["train_loss"] == [1.0, 0.8, 0.6]
    assert loaded["val_loss"] == [1.1, 0.9, 0.7]


def test_load_metrics_raises_for_missing_file(tmp_path: Path) -> None:
    """load_metrics raises FileNotFoundError for missing path."""
    from sine_extraction.results import load_metrics

    with pytest.raises(FileNotFoundError):
        load_metrics(tmp_path / "nonexistent.json")
