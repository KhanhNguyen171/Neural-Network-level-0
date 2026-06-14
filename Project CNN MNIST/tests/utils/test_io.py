import tempfile
from pathlib import Path

import pytest

from src.utils.io import (
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    save_pickle,
    load_pickle,
)

# pytest tests/utils/test_io.py -v


def test_save_and_load_json():
    data = {
        "name": "mnist",
        "epochs": 10,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.json"

        save_json(
            data,
            path,
        )

        loaded = load_json(path)

        assert loaded == data


def test_json_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_json("missing.json")


def test_save_and_load_yaml():
    data = {
        "training": {
            "epochs": 20,
            "lr": 0.001,
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.yaml"

        save_yaml(
            data,
            path,
        )

        loaded = load_yaml(path)

        assert loaded == data


def test_yaml_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_yaml("missing.yaml")


def test_save_and_load_pickle():
    data = {
        "loss": [1.0, 0.5, 0.1],
        "accuracy": [0.8, 0.9],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "history.pkl"

        save_pickle(
            data,
            path,
        )

        loaded = load_pickle(path)

        assert loaded == data


def test_pickle_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_pickle("missing.pkl")


def test_nested_directory_creation():
    data = {
        "test": True,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = (
            Path(tmpdir)
            / "a"
            / "b"
            / "c"
            / "file.json"
        )

        save_json(
            data,
            path,
        )

        assert path.exists()

        loaded = load_json(path)

        assert loaded == data


def test_pickle_complex_object():
    obj = [
        {"a": 1},
        {"b": 2},
        [1, 2, 3],
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "obj.pkl"

        save_pickle(
            obj,
            path,
        )

        loaded = load_pickle(path)

        assert loaded == obj