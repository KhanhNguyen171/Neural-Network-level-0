# tests/training/test_history.py

import json

import pytest

from src.training.history import TrainingHistory

# pytest tests/training/test_history.py -v

# Initialization


def test_history_initialization():
    history = TrainingHistory()

    assert history.history == {}
    assert len(history) == 0


def test_history_contains_empty():
    history = TrainingHistory()

    assert "loss" not in history


# Update


def test_update_single_metric():
    history = TrainingHistory()

    history.update(loss=0.5)

    assert history.history["loss"] == [0.5]


def test_update_multiple_metrics():
    history = TrainingHistory()

    history.update(
        train_loss=0.5,
        val_loss=0.4,
        accuracy=0.9,
    )

    assert history.history["train_loss"] == [0.5]
    assert history.history["val_loss"] == [0.4]
    assert history.history["accuracy"] == [0.9]


def test_update_multiple_epochs():
    history = TrainingHistory()

    history.update(loss=1.0)
    history.update(loss=0.8)
    history.update(loss=0.6)

    assert history.history["loss"] == [
        1.0,
        0.8,
        0.6,
    ]


def test_update_converts_to_float():
    history = TrainingHistory()

    history.update(loss=1)

    assert isinstance(
        history.history["loss"][0],
        float,
    )


# Get


def test_get_existing_metric():
    history = TrainingHistory()

    history.update(loss=0.5)

    values = history.get("loss")

    assert values == [0.5]


def test_get_missing_metric():
    history = TrainingHistory()

    assert history.get("unknown") == []


# Latest


def test_latest_metric():
    history = TrainingHistory()

    history.update(loss=1.0)
    history.update(loss=0.5)

    assert history.latest("loss") == 0.5


def test_latest_missing_metric_raises():
    history = TrainingHistory()

    with pytest.raises(ValueError):
        history.latest("loss")


# Best


def test_best_max():
    history = TrainingHistory()

    history.update(acc=0.7)
    history.update(acc=0.8)
    history.update(acc=0.9)

    assert history.best("acc") == 0.9


def test_best_min():
    history = TrainingHistory()

    history.update(loss=1.0)
    history.update(loss=0.5)
    history.update(loss=0.8)

    assert history.best("loss", mode="min") == 0.5


def test_best_invalid_mode():
    history = TrainingHistory()

    history.update(loss=1.0)

    with pytest.raises(ValueError):
        history.best("loss", mode="invalid")


def test_best_missing_metric():
    history = TrainingHistory()

    with pytest.raises(ValueError):
        history.best("loss")


# Best Epoch


def test_best_epoch_max():
    history = TrainingHistory()

    history.update(acc=0.6)
    history.update(acc=0.8)
    history.update(acc=0.7)

    assert history.best_epoch("acc") == 2


def test_best_epoch_min():
    history = TrainingHistory()

    history.update(loss=0.8)
    history.update(loss=0.3)
    history.update(loss=0.5)

    assert history.best_epoch(
        "loss",
        mode="min",
    ) == 2


def test_best_epoch_invalid_mode():
    history = TrainingHistory()

    history.update(loss=0.5)

    with pytest.raises(ValueError):
        history.best_epoch(
            "loss",
            mode="bad",
        )


def test_best_epoch_missing_metric():
    history = TrainingHistory()

    with pytest.raises(ValueError):
        history.best_epoch("loss")


# Num Epochs / Length


def test_num_epochs_empty():
    history = TrainingHistory()

    assert history.num_epochs() == 0


def test_num_epochs_non_empty():
    history = TrainingHistory()

    history.update(loss=1.0)
    history.update(loss=0.8)

    assert history.num_epochs() == 2


def test_len_matches_num_epochs():
    history = TrainingHistory()

    history.update(loss=1.0)
    history.update(loss=0.8)
    history.update(loss=0.5)

    assert len(history) == 3


# Contains


def test_contains_metric():
    history = TrainingHistory()

    history.update(loss=0.5)

    assert "loss" in history


def test_contains_missing_metric():
    history = TrainingHistory()

    assert "accuracy" not in history


# State Dict


def test_state_dict():
    history = TrainingHistory()

    history.update(loss=0.5)

    state = history.state_dict()

    assert "history" in state
    assert state["history"]["loss"] == [0.5]


def test_load_state_dict():
    history = TrainingHistory()

    state = {
        "history": {
            "loss": [1.0, 0.5]
        }
    }

    history.load_state_dict(state)

    assert history.history["loss"] == [
        1.0,
        0.5,
    ]


# Save / Load JSON


def test_save_history(tmp_path):
    history = TrainingHistory()

    history.update(loss=0.5)

    file_path = tmp_path / "history.json"

    history.save(str(file_path))

    assert file_path.exists()

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    assert data["loss"] == [0.5]


def test_load_history(tmp_path):
    file_path = tmp_path / "history.json"

    data = {
        "loss": [1.0, 0.5],
        "accuracy": [0.8, 0.9],
    }

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data, f)

    history = TrainingHistory()

    history.load(str(file_path))

    assert history.history == data


# Clear


def test_clear_history():
    history = TrainingHistory()

    history.update(loss=0.5)
    history.update(acc=0.9)

    history.clear()

    assert history.history == {}
    assert len(history) == 0


# Repr


def test_repr_empty():
    history = TrainingHistory()

    rep = repr(history)

    assert "TrainingHistory" in rep
    assert "epochs=0" in rep


def test_repr_non_empty():
    history = TrainingHistory()

    history.update(loss=0.5)

    rep = repr(history)

    assert "TrainingHistory" in rep
    assert "loss" in rep