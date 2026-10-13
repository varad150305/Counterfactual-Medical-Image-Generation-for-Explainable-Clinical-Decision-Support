"""Minimal YAML config loading — plain nested dicts, no framework.

Supports two small additions on top of plain YAML, both needed by the shared
experiment configs (configs/baseline.yaml -> disease_consistency.yaml ->
anatomy_consistency.yaml): `extends:` (deep-merge a parent config) and dotted
command-line overrides (`--set editing.num_ddim_steps=4`).
"""
import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into a copy of `base` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str, _seen: Optional[tuple] = None) -> Dict[str, Any]:
    """Load a YAML config file into a plain nested dict, resolving `extends`.

    `extends: other.yaml` (path relative to this file) deep-merges this file
    over the parent. Raises FileNotFoundError with a clear message if a path is
    wrong, and ValueError on empty files or `extends` cycles.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    resolved = str(config_path.resolve())
    seen = (_seen or ()) + (resolved,)
    if resolved in (_seen or ()):
        raise ValueError(f"Circular `extends` chain: {' -> '.join(seen)}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config file is empty: {config_path.resolve()}")

    parent = config.pop("extends", None)
    if parent:
        parent_cfg = load_config(str(config_path.parent / parent), _seen=seen)
        config = deep_update(parent_cfg, config)
    return config


def _parse_override_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    # PyYAML (YAML 1.1) reads "1e-5" as a *string*; coerce numeric-looking strings.
    if isinstance(value, str) and _FLOAT_RE.match(value.strip()):
        return float(value)
    return value


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply `dotted.key=value` overrides to a copy of `config` (values parsed as YAML)."""
    out = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like dotted.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override {key!r}: {part!r} is not a mapping")
        node[parts[-1]] = _parse_override_value(raw)
    return out


def resolve_device(requested: str = "cpu"):
    """'cpu' -> cpu; 'cuda' -> cuda or a clear error (never a silent CPU
    fallback: an hours-long GPU job quietly running on CPU is worse than a
    failure); 'auto' -> cuda if available else cpu."""
    import torch

    requested = (requested or "cpu").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={requested!r} was requested but torch.cuda.is_available() is False "
                f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). Enable a GPU "
                f"session / check the driver, or set device to 'cpu' or 'auto'."
            )
        return torch.device(requested)
    raise ValueError(f"Unknown device {requested!r}; expected 'cpu', 'cuda' or 'auto'")


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs for reproducibility (CPU-only; the project
    has no CUDA device, see docs/PHASE_1_AUDIT.md, so no cudnn flags needed)."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
