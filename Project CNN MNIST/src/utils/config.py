"""
Configuration utilities.

Supports:
- YAML loading
- YAML saving
- Recursive dictionary merging
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to yaml file.

    Returns
    -------
    dict
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(
    config: dict[str, Any],
    path: str | Path,
) -> None:
    """
    Save configuration to YAML.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def merge_configs(
    *configs: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge multiple configs.

    Later configs override earlier configs.
    """

    merged: dict[str, Any] = {}

    for config in configs:
        merged = _recursive_merge(
            merged,
            config,
        )

    return merged


def _recursive_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:

    result = base.copy()

    for key, value in override.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _recursive_merge(
                result[key],
                value,
            )
        else:
            result[key] = value

    return result
