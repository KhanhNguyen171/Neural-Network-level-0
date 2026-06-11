# tests/training/test_checkpoint.py

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.training.checkpoint import ModelCheckpoint
from src.training.history import TrainingHistory

# pytest tests/training/test_checkpoint.py -v

# Helpers


def create_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def create_optimizer(model):
    return optim.Adam(
        model.parameters(),
        lr=1e-3,
    )


class DummyScheduler:
    def __init__(self):
        self.state = {
            "step": 0
        }

    def state_dict(self):
        return self.state

    def load_state_dict(
        self,
        state_dict,
    ):
        self.state = state_dict


# Initialization


def test_checkpoint_initialization(
    tmp_path,
):
    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    assert ckpt.path.exists()
    assert ckpt.path.is_dir()


def test_checkpoint_repr(
    tmp_path,
):
    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    rep = repr(ckpt)

    assert "ModelCheckpoint" in rep


# Save


def test_save_checkpoint(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        epoch=1,
    )

    assert filepath.exists()
    assert filepath.name == "checkpoint.pt"


def test_save_custom_filename(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        filename="custom.pt",
    )

    assert filepath.exists()
    assert filepath.name == "custom.pt"


def test_save_with_metric(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        metric=0.95,
    )

    checkpoint = torch.load(
        filepath,
        weights_only=False,
    )

    assert checkpoint["metric"] == 0.95


def test_save_with_extra_data(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        extra={
            "experiment": "mnist"
        },
    )

    checkpoint = torch.load(
        filepath,
        weights_only=False,
    )

    assert checkpoint["extra"][
        "experiment"
    ] == "mnist"


# Optimizer


def test_save_optimizer_state(
    tmp_path,
):
    model = create_model()

    optimizer = create_optimizer(
        model
    )

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        optimizer=optimizer,
    )

    checkpoint = torch.load(
        filepath,
        weights_only=False,
    )

    assert (
        "optimizer_state_dict"
        in checkpoint
    )


# Scheduler


def test_save_scheduler_state(
    tmp_path,
):
    model = create_model()

    scheduler = DummyScheduler()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        scheduler=scheduler,
    )

    checkpoint = torch.load(
        filepath,
        weights_only=False,
    )

    assert (
        "scheduler_state_dict"
        in checkpoint
    )


# History


def test_save_history(
    tmp_path,
):
    model = create_model()

    history = TrainingHistory()

    history.update(
        loss=0.5
    )

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        history=history,
    )

    checkpoint = torch.load(
        filepath,
        weights_only=False,
    )

    assert "history" in checkpoint


# Load


def test_load_checkpoint(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        epoch=5,
    )

    new_model = create_model()

    checkpoint = ckpt.load(
        filepath=filepath,
        model=new_model,
    )

    assert checkpoint["epoch"] == 5


def test_load_optimizer_state(
    tmp_path,
):
    model = create_model()

    optimizer = create_optimizer(
        model
    )

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        optimizer=optimizer,
    )

    new_model = create_model()

    new_optimizer = create_optimizer(
        new_model
    )

    ckpt.load(
        filepath=filepath,
        model=new_model,
        optimizer=new_optimizer,
    )

    assert (
        len(
            new_optimizer.state_dict()
        )
        > 0
    )


def test_load_scheduler_state(
    tmp_path,
):
    model = create_model()

    scheduler = DummyScheduler()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        scheduler=scheduler,
    )

    new_scheduler = DummyScheduler()

    new_scheduler.state = {}

    ckpt.load(
        filepath=filepath,
        model=create_model(),
        scheduler=new_scheduler,
    )

    assert (
        new_scheduler.state
        == {"step": 0}
    )


def test_load_history_state(
    tmp_path,
):
    model = create_model()

    history = TrainingHistory()

    history.update(
        loss=0.5
    )

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save(
        model=model,
        history=history,
    )

    new_history = (
        TrainingHistory()
    )

    ckpt.load(
        filepath=filepath,
        model=create_model(),
        history=new_history,
    )

    assert (
        new_history.history
        == history.history
    )


# Exists


def test_exists_true(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    ckpt.save(
        model=model,
        filename="test.pt",
    )

    assert ckpt.exists(
        "test.pt"
    )


def test_exists_false(
    tmp_path,
):
    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    assert not ckpt.exists(
        "missing.pt"
    )


# Delete


def test_delete_checkpoint(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    ckpt.save(
        model=model,
        filename="temp.pt",
    )

    assert ckpt.exists(
        "temp.pt"
    )

    ckpt.delete(
        "temp.pt"
    )

    assert not ckpt.exists(
        "temp.pt"
    )


def test_delete_missing_file(
    tmp_path,
):
    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    ckpt.delete(
        "not_found.pt"
    )


# Latest


def test_latest_checkpoint(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    ckpt.save(
        model=model,
        filename="a.pt",
    )

    ckpt.save(
        model=model,
        filename="b.pt",
    )

    latest = ckpt.latest()

    assert latest is not None
    assert latest.exists()


def test_latest_empty_directory(
    tmp_path,
):
    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    assert (
        ckpt.latest()
        is None
    )


# List Checkpoints


def test_list_checkpoints(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    ckpt.save(
        model=model,
        filename="a.pt",
    )

    ckpt.save(
        model=model,
        filename="b.pt",
    )

    files = (
        ckpt.list_checkpoints()
    )

    assert len(files) == 2

    assert all(
        isinstance(
            f,
            Path,
        )
        for f in files
    )


# Save Best


def test_save_best(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save_best(
        model=model,
        metric=0.98,
    )

    assert filepath.exists()
    assert (
        filepath.name
        == "best.pt"
    )


# Save Last


def test_save_last(
    tmp_path,
):
    model = create_model()

    ckpt = ModelCheckpoint(
        str(tmp_path)
    )

    filepath = ckpt.save_last(
        model=model,
        epoch=10,
    )

    assert filepath.exists()
    assert (
        filepath.name
        == "last.pt"
    )