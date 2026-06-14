import pytest
import torch
import torch.nn as nn

from src.training.optimizers import (
    OPTIMIZER_REGISTRY,
    available_optimizers,
    get_optimizer_class,
    build_optimizer,
    build_optimizer_from_config,
    optimizer_to_dict,
    count_trainable_parameters,
)

# pytest tests/training/test_optimizers.py -v

class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(
            self.fc1(x)
        )


def test_available_optimizers():
    names = available_optimizers()

    assert "adam" in names
    assert "sgd" in names


def test_registry_exists():
    assert isinstance(
        OPTIMIZER_REGISTRY,
        dict,
    )


def test_get_optimizer_class():
    cls = get_optimizer_class("adam")

    assert cls is torch.optim.Adam


def test_get_optimizer_class_case_insensitive():
    cls = get_optimizer_class("ADAM")

    assert cls is torch.optim.Adam


def test_unknown_optimizer():
    with pytest.raises(KeyError):
        get_optimizer_class(
            "unknown"
        )


def test_build_adam():
    model = DummyModel()

    optimizer = build_optimizer(
        model,
        name="adam",
        lr=1e-3,
    )

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )


def test_build_sgd():
    model = DummyModel()

    optimizer = build_optimizer(
        model,
        name="sgd",
        lr=0.01,
    )

    assert isinstance(
        optimizer,
        torch.optim.SGD,
    )


def test_build_optimizer_from_config():
    model = DummyModel()

    config = {
        "name": "adam",
        "lr": 0.001,
    }

    optimizer = (
        build_optimizer_from_config(
            model,
            config,
        )
    )

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )


def test_build_optimizer_missing_name():
    model = DummyModel()

    with pytest.raises(KeyError):
        build_optimizer_from_config(
            model,
            {"lr": 1e-3},
        )


def test_optimizer_to_dict():
    model = DummyModel()

    optimizer = build_optimizer(
        model,
        name="adam",
        lr=0.001,
        weight_decay=0.01,
    )

    info = optimizer_to_dict(
        optimizer
    )

    assert info["lr"] == 0.001
    assert (
        info["weight_decay"]
        == 0.01
    )


def test_count_trainable_parameters():
    model = DummyModel()

    count = (
        count_trainable_parameters(
            model
        )
    )

    expected = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    assert count == expected


def test_ignore_frozen_parameters():
    model = DummyModel()

    for param in model.fc1.parameters():
        param.requires_grad = False

    count = (
        count_trainable_parameters(
            model
        )
    )

    expected = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    assert count == expected


def test_optimizer_step():
    model = DummyModel()

    optimizer = build_optimizer(
        model,
        name="adam",
        lr=1e-3,
    )

    x = torch.randn(
        4,
        10,
    )

    y = model(x)

    loss = y.mean()

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    assert True