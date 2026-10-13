"""Minimal YAML config loading — plain nested dicts, no framework."""
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a plain nested dict.

    Raises FileNotFoundError with a clear message if the path is wrong, rather
    than letting a cryptic yaml/OS error surface — configs are the first thing
    a user points at a wrong path.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config file is empty: {config_path.resolve()}")
    return config


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs for reproducibility (CPU-only; the project
    has no CUDA device, see docs/PHASE_1_AUDIT.md, so no cudnn flags needed)."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
