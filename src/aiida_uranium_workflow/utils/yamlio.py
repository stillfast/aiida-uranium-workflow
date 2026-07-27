"""Low-level JSON/YAML readers used across workchains.

These helpers are kept tiny and AiiDA-free so any module can import them
without paying for aiida-core / pyxtal dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file, requiring the root to be a mapping."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root of {path} must be a mapping, got {type(data)}")
    return data


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
