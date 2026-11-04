"""Configuration loading helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a dictionary at its root.")
    return config


def apply_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Apply non-None command-line overrides without mutating the original."""
    updated = deepcopy(config)
    mapping = {
        "data_root": ("data", "root"),
        "image_size": ("data", "image_size"),
        "architecture": ("model", "architecture"),
        "encoder": ("model", "encoder"),
        "encoder_weights": ("model", "encoder_weights"),
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "learning_rate": ("training", "learning_rate"),
        "output_dir": ("training", "output_dir"),
    }
    for key, value in overrides.items():
        if value is None or key not in mapping:
            continue
        section, field = mapping[key]
        updated.setdefault(section, {})[field] = value
    return updated
