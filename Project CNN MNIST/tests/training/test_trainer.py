# tests/training/test_trainer.py

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.callbacks import CallbackList
from src.training.checkpoint import ModelCheckpoint
from src.training.early_stopping import EarlyStopping
from src.training.history import TrainingHistory
from src.training.trainer import Trainer

# pytest tests/training/test_trainer.py -v

# Helpers

class TinyNet(nn.Module):
    def __init__(self, in_features=20, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class RecordingCallback(CallbackList):
    def __init__(self):
        self.events = []

    def on_train_begin(self, trainer):
        self.events.append("train_begin")

    def on_train_end(self, trainer):
        self.events.append("train_end")

    def on_epoch_begin(self, trainer, epoch: int):
        self.events.append(f"epoch_begin_{epoch}")

    def on_epoch_end(self, trainer, epoch: int, logs=None):
        self.events.append(f"epoch_end_{epoch}")

    def on_batch_begin(self, trainer, batch_idx: int):
        self.events.append("batch_begin")

    def on_batch_end(self, trainer, batch_idx: int, logs=None):
        self.events.append("batch_end")


class DummyScheduler:
    def __init__(self):
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = state["steps"]


def create_model():
    return TinyNet()


def create_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)


def create_train_loader(
    num_samples=128,
    in_features=20,
    num_classes=10,
    batch_size=16,
):
    x = torch.randn(num_samples, in_features)
    y = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def create_val_loader(
    num_samples=64,
    in_features=20,
    num_classes=10,
    batch_size=16,
):
    x = torch.randn(num_samples, in_features)
    y = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def create_trainer(
    callbacks=None,
    checkpoint=None,
    scheduler=None,
    early_stopping=None,
):
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model)

    return Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        checkpoint=checkpoint,
        early_stopping=early_stopping,
        device="cpu",
    )


# Initialization

def test_trainer_creation():
    trainer = create_trainer()

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.criterion is not None
    assert isinstance(trainer.history, TrainingHistory)


def test_epoch_property_initial_value():
    trainer = create_trainer()

    assert trainer.epoch == 0


def test_repr_contains_class_name():
    trainer = create_trainer()

    text = repr(trainer)

    assert "Trainer" in text


# train_one_epoch

def test_train_one_epoch_returns_metrics():
    trainer = create_trainer()

    loader = create_train_loader()

    metrics = trainer.train_one_epoch(loader)

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_train_one_epoch_loss_is_float():
    trainer = create_trainer()

    loader = create_train_loader()

    metrics = trainer.train_one_epoch(loader)

    assert isinstance(metrics["loss"], float)


def test_train_one_epoch_accuracy_range():
    trainer = create_trainer()

    loader = create_train_loader()

    metrics = trainer.train_one_epoch(loader)

    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_train_one_epoch_updates_model_weights():
    trainer = create_trainer()

    loader = create_train_loader()

    before = [
        p.detach().clone()
        for p in trainer.model.parameters()
    ]

    trainer.train_one_epoch(loader)

    after = list(trainer.model.parameters())

    changed = any(
        not torch.equal(b, a)
        for b, a in zip(before, after)
    )

    assert changed


# validate

def test_validate_returns_metrics():
    trainer = create_trainer()

    loader = create_val_loader()

    metrics = trainer.validate(loader)

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_validate_loss_is_float():
    trainer = create_trainer()

    loader = create_val_loader()

    metrics = trainer.validate(loader)

    assert isinstance(metrics["loss"], float)


def test_validate_accuracy_range():
    trainer = create_trainer()

    loader = create_val_loader()

    metrics = trainer.validate(loader)

    assert 0.0 <= metrics["accuracy"] <= 1.0


# Predict

def test_predict_returns_tensor():
    trainer = create_trainer()

    loader = create_val_loader(
        num_samples=12,
        batch_size=4,
    )

    preds = trainer.predict(loader)

    assert isinstance(preds, torch.Tensor)


def test_predict_batch_size():
    trainer = create_trainer()

    loader = create_val_loader(
        num_samples=12,
        batch_size=4,
    )

    preds = trainer.predict(loader)

    assert len(preds) == 12


def test_predict_returns_class_indices():
    trainer = create_trainer()

    loader = create_val_loader(
        num_samples=12,
        batch_size=4,
    )

    preds = trainer.predict(loader)

    assert preds.ndim == 1
    assert preds.dtype == torch.long


# Evaluate

def test_evaluate_returns_metrics():
    trainer = create_trainer()

    loader = create_val_loader()

    metrics = trainer.evaluate(loader)

    assert "loss" in metrics
    assert "accuracy" in metrics


def test_evaluate_matches_validate_keys():
    trainer = create_trainer()

    loader = create_val_loader()

    eval_metrics = trainer.evaluate(loader)
    val_metrics = trainer.validate(loader)

    assert eval_metrics.keys() == val_metrics.keys()


# Callbacks

def test_callbacks_are_triggered():
    callback = RecordingCallback()
    
    callbacks = CallbackList([callback])

    trainer = create_trainer(
        callbacks=callbacks
    )

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    trainer.fit(
        train_loader,
        val_loader,
        epochs=1,
    )

    assert "train_begin" in callback.events
    assert "train_end" in callback.events

    assert any(
        e.startswith("epoch_begin")
        for e in callback.events
    )

    assert any(
        e.startswith("epoch_end")
        for e in callback.events
    )

    assert "batch_begin" in callback.events
    assert "batch_end" in callback.events


# Scheduler

def test_scheduler_step_called_each_epoch():
    scheduler = DummyScheduler()

    trainer = create_trainer(
        scheduler=scheduler
    )

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    trainer.fit(
        train_loader,
        val_loader,
        epochs=3,
    )

    assert scheduler.steps == 3


# History

def test_fit_returns_history():
    trainer = create_trainer()

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=2,
    )

    assert isinstance(history, TrainingHistory)


def test_history_contains_two_epochs():
    trainer = create_trainer()

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=2,
    )

    assert history.num_epochs() == 2


def test_epoch_property_after_training():
    trainer = create_trainer()

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    trainer.fit(
        train_loader,
        val_loader,
        epochs=3,
    )

    assert trainer.epoch == 3


# Early stopping

def test_fit_with_early_stopping():
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=0,
        mode="min",
    )

    trainer = create_trainer(
        early_stopping=early_stopping
    )

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=20,
    )

    assert history.num_epochs() <= 20


def test_early_stopping_object_attached():
    es = EarlyStopping()

    trainer = create_trainer(
        early_stopping=es
    )

    assert trainer.early_stopping is es


# Checkpoint

def test_fit_creates_last_checkpoint(tmp_path):
    checkpoint = ModelCheckpoint(tmp_path)

    trainer = create_trainer(
        checkpoint=checkpoint
    )

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    trainer.fit(
        train_loader,
        val_loader,
        epochs=1,
    )

    assert checkpoint.exists("last.pt")


def test_resume_restores_epoch(tmp_path):
    checkpoint = ModelCheckpoint(tmp_path)

    trainer = create_trainer(
        checkpoint=checkpoint
    )

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    trainer.fit(
        train_loader,
        val_loader,
        epochs=2,
    )

    new_trainer = create_trainer(
        checkpoint=checkpoint
    )
    
    checkpoint_file = tmp_path / "last.pt"

    new_trainer.resume(checkpoint_file)

    assert new_trainer.epoch == trainer.epoch


def test_resume_restores_history(tmp_path):
    checkpoint = ModelCheckpoint(tmp_path)

    trainer = create_trainer(
        checkpoint=checkpoint
    )

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    trainer.fit(
        train_loader,
        val_loader,
        epochs=2,
    )

    restored = create_trainer(
        checkpoint=checkpoint
    )

    checkpoint_file = tmp_path / "last.pt"

    restored.resume(checkpoint_file)

    assert (
        restored.history.num_epochs()
        == trainer.history.num_epochs()
    )


def test_resume_restores_scheduler(tmp_path):
    scheduler = DummyScheduler()

    checkpoint = ModelCheckpoint(tmp_path)

    trainer = create_trainer(
        scheduler=scheduler,
        checkpoint=checkpoint,
    )

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    trainer.fit(
        train_loader,
        val_loader,
        epochs=2,
    )

    restored_scheduler = DummyScheduler()

    restored = create_trainer(
        scheduler=restored_scheduler,
        checkpoint=checkpoint,
    )

    checkpoint_file = tmp_path / "last.pt"

    restored.resume(checkpoint_file)

    assert restored_scheduler.steps == scheduler.steps


# Edge cases

def test_fit_single_epoch():
    trainer = create_trainer()

    train_loader = create_train_loader()
    val_loader = create_val_loader()

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=1,
    )

    assert history.num_epochs() == 1


def test_train_loader_with_single_batch():
    trainer = create_trainer()

    loader = create_train_loader(
        num_samples=8,
        batch_size=8,
    )

    metrics = trainer.train_one_epoch(loader)

    assert "loss" in metrics


def test_validate_loader_with_single_batch():
    trainer = create_trainer()

    loader = create_val_loader(
        num_samples=8,
        batch_size=8,
    )

    metrics = trainer.validate(loader)

    assert "accuracy" in metrics

