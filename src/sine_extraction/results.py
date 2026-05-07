"""Experiment results persistence — save and load JSON metrics."""

from __future__ import annotations

import json
from pathlib import Path


def save_metrics(metrics: dict, path: Path) -> None:
    """Serialise *metrics* to an indented JSON file at *path*.

    Parent directories are created automatically.

    Args:
        metrics: Dictionary of metric names to values (may be nested).
        path: Destination file path (will be created or overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def load_metrics(path: Path) -> dict:
    """Load metrics from a JSON file previously written by :func:`save_metrics`.

    Args:
        path: Path to a JSON file.

    Returns:
        Dictionary of metrics as stored in the file.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text())
