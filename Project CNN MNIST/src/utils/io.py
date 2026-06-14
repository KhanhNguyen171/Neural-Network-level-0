"""
Input / Output utilities.
"""

from pathlib import Path
from typing import Any
import json
import pickle

import yaml


def save_json(
    data: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(
    data: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def load_yaml(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def save_pickle(
    obj: Any,
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "wb") as f:
        pickle.dump(
            obj,
            f,
        )


def load_pickle(
    path: str | Path,
) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        return pickle.load(f)