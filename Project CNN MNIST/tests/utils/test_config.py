import tempfile
from pathlib import Path

import pytest

from src.utils.config import (
    load_config,
    save_config,
    merge_configs,
)

# pytest tests/utils/test_config.py -v


def test_load_config():
    config_data = {
        "model": {
            "name": "mnist_cnn"
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.yaml"

        save_config(
            config_data,
            path,
        )

        loaded = load_config(path)

        assert loaded == config_data


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_save_config():
    config = {
        "training": {
            "epochs": 10
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.yaml"

        save_config(
            config,
            path,
        )

        assert path.exists()

        loaded = load_config(path)

        assert loaded == config


def test_merge_configs_simple():
    cfg1 = {
        "lr": 0.001,
        "batch_size": 32,
    }

    cfg2 = {
        "batch_size": 64,
    }

    merged = merge_configs(
        cfg1,
        cfg2,
    )

    assert merged["lr"] == 0.001
    assert merged["batch_size"] == 64


def test_merge_configs_nested():
    cfg1 = {
        "training": {
            "epochs": 10,
            "lr": 0.001,
        }
    }

    cfg2 = {
        "training": {
            "lr": 0.01,
        }
    }

    merged = merge_configs(
        cfg1,
        cfg2,
    )

    assert merged["training"]["epochs"] == 10
    assert merged["training"]["lr"] == 0.01


def test_merge_multiple_configs():
    cfg1 = {"a": 1}
    cfg2 = {"b": 2}
    cfg3 = {"a": 3}

    merged = merge_configs(
        cfg1,
        cfg2,
        cfg3,
    )

    assert merged == {
        "a": 3,
        "b": 2,
    }