# tests/training/test_early_stopping.py

import pytest

from src.training.early_stopping import EarlyStopping

# pytest tests/training/test_early_stopping.py -v

# Initialization


def test_initialization_defaults():
    early_stopping = EarlyStopping()

    assert early_stopping.monitor == "val_loss"
    assert early_stopping.patience == 5
    assert early_stopping.mode == "min"
    assert early_stopping.min_delta == 0.0

    assert early_stopping.best_score is None
    assert early_stopping.best_epoch is None
    assert early_stopping.counter == 0
    assert not early_stopping.should_stop


def test_initialization_custom_values():
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        mode="max",
        min_delta=0.01,
    )

    assert early_stopping.monitor == "val_accuracy"
    assert early_stopping.patience == 10
    assert early_stopping.mode == "max"
    assert early_stopping.min_delta == 0.01


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        EarlyStopping(mode="invalid")


def test_negative_patience_raises():
    with pytest.raises(ValueError):
        EarlyStopping(patience=-1)


# Improvement Logic (min mode)


def test_first_step_is_improvement():
    early_stopping = EarlyStopping()

    stop = early_stopping.step(
        current=1.0,
        epoch=1,
    )

    assert stop is False
    assert early_stopping.best_score == 1.0
    assert early_stopping.best_epoch == 1
    assert early_stopping.counter == 0


def test_min_mode_improvement():
    early_stopping = EarlyStopping(
        mode="min"
    )

    early_stopping.step(
        current=1.0,
        epoch=1,
    )

    stop = early_stopping.step(
        current=0.8,
        epoch=2,
    )

    assert stop is False
    assert early_stopping.best_score == 0.8
    assert early_stopping.best_epoch == 2
    assert early_stopping.counter == 0


def test_min_mode_no_improvement():
    early_stopping = EarlyStopping(
        mode="min",
        patience=3,
    )

    early_stopping.step(
        current=0.5,
        epoch=1,
    )

    early_stopping.step(
        current=0.6,
        epoch=2,
    )

    assert early_stopping.counter == 1
    assert not early_stopping.should_stop


# Improvement Logic (max mode)


def test_max_mode_improvement():
    early_stopping = EarlyStopping(
        mode="max"
    )

    early_stopping.step(
        current=0.80,
        epoch=1,
    )

    early_stopping.step(
        current=0.90,
        epoch=2,
    )

    assert early_stopping.best_score == 0.90
    assert early_stopping.best_epoch == 2
    assert early_stopping.counter == 0


def test_max_mode_no_improvement():
    early_stopping = EarlyStopping(
        mode="max",
        patience=3,
    )

    early_stopping.step(
        current=0.90,
        epoch=1,
    )

    early_stopping.step(
        current=0.85,
        epoch=2,
    )

    assert early_stopping.counter == 1
    assert not early_stopping.should_stop


# Patience


def test_stop_after_patience_reached():
    early_stopping = EarlyStopping(
        patience=3,
        mode="min",
    )

    early_stopping.step(1.0, 1)

    early_stopping.step(1.1, 2)
    early_stopping.step(1.2, 3)

    stop = early_stopping.step(
        1.3,
        4,
    )

    assert stop
    assert early_stopping.should_stop


def test_counter_resets_after_improvement():
    early_stopping = EarlyStopping(
        patience=3,
        mode="min",
    )

    early_stopping.step(1.0, 1)

    early_stopping.step(1.1, 2)
    early_stopping.step(1.2, 3)

    assert early_stopping.counter == 2

    early_stopping.step(
        0.8,
        4,
    )

    assert early_stopping.counter == 0
    assert early_stopping.best_score == 0.8


# min_delta


def test_min_delta_min_mode():
    early_stopping = EarlyStopping(
        mode="min",
        min_delta=0.1,
    )

    early_stopping.step(
        1.0,
        1,
    )

    early_stopping.step(
        0.95,
        2,
    )

    assert early_stopping.counter == 1
    assert early_stopping.best_score == 1.0


def test_min_delta_max_mode():
    early_stopping = EarlyStopping(
        mode="max",
        min_delta=0.05,
    )

    early_stopping.step(
        0.90,
        1,
    )

    early_stopping.step(
        0.93,
        2,
    )

    assert early_stopping.counter == 1
    assert early_stopping.best_score == 0.90


# Reset


def test_reset():
    early_stopping = EarlyStopping()

    early_stopping.step(
        1.0,
        1,
    )

    early_stopping.step(
        1.1,
        2,
    )

    early_stopping.reset()

    assert early_stopping.best_score is None
    assert early_stopping.best_epoch is None
    assert early_stopping.counter == 0
    assert not early_stopping.should_stop


# Properties


def test_stopped_property():
    early_stopping = EarlyStopping(
        patience=1,
    )

    early_stopping.step(
        1.0,
        1,
    )

    early_stopping.step(
        1.1,
        2,
    )

    assert early_stopping.stopped


def test_num_bad_epochs_property():
    early_stopping = EarlyStopping(
        patience=5,
    )

    early_stopping.step(
        1.0,
        1,
    )

    early_stopping.step(
        1.1,
        2,
    )

    early_stopping.step(
        1.2,
        3,
    )

    assert (
        early_stopping.num_bad_epochs
        == 2
    )


# State Dict


def test_state_dict():
    early_stopping = EarlyStopping(
        patience=3,
        mode="min",
    )

    early_stopping.step(
        1.0,
        1,
    )

    state = early_stopping.state_dict()

    assert state["patience"] == 3
    assert state["mode"] == "min"
    assert state["best_score"] == 1.0
    assert state["best_epoch"] == 1


def test_load_state_dict():
    early_stopping = EarlyStopping()

    state = {
        "monitor": "val_loss",
        "patience": 5,
        "mode": "min",
        "min_delta": 0.0,
        "best_score": 0.5,
        "best_epoch": 10,
        "counter": 2,
        "should_stop": False,
    }

    early_stopping.load_state_dict(
        state
    )

    assert early_stopping.best_score == 0.5
    assert early_stopping.best_epoch == 10
    assert early_stopping.counter == 2


# Repr


def test_repr():
    early_stopping = EarlyStopping()

    rep = repr(early_stopping)

    assert "EarlyStopping" in rep
    assert "patience=5" in rep
    assert "mode='min'" in rep