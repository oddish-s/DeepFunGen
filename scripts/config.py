"""Lightweight YAML configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


__all__ = ["load_config"]
