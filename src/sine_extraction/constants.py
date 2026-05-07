"""Project-wide constants — no hardcoded values anywhere else."""

from __future__ import annotations

import math
from pathlib import Path

# Signal parameters
KNOWN_FREQUENCIES: list[float] = [1.0, 5.0, 10.0, 20.0]
SAMPLE_RATE: int = 200
WINDOW_SIZE: int = 10

# Reproducibility
DEFAULT_SEED: int = 42

# Math
TWO_PI: float = 2.0 * math.pi

# Paths
DEFAULT_CONFIG_PATH: Path = Path("config/config.yaml")

# Checkpoint naming
CHECKPOINT_FILENAME_TEMPLATE: str = "{model_name}_epoch{epoch:04d}.pt"
