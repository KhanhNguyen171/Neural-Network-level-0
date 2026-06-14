import pytest
import torch
import torch.nn as nn

from src.training.schedulers import (
    SCHEDULER_REGISTRY,
    available_schedulers,
    get_scheduler_class,
    build_scheduler,
    build_scheduler_from_config,
    get_learning_rate,
    scheduler_to_dict,
)

# pytest tests/training/test_schedulers.py -v

class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(
            10,
            2,
        )

    def forward(
        self,
        x,
    ):
        return self.fc(x)


@pytest.fixture
def optimizer():
    model = DummyModel()

    return torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )


def test_scheduler_registry():
    assert isinstance(
        SCHEDULER_REGISTRY,
        dict
    )


def test_available_schedulers():
    schedulers = (
        available_schedulers()
    )

    assert "step" in schedulers
    assert "plateau" in schedulers


def test_get_scheduler_class():
    cls = get_scheduler_class(
        "step"
    )

    assert (
        cls.__name__
        == "StepLR"
    )


def test_get_scheduler_class_case():
    cls = get_scheduler_class(
        "STEP"
    )

    assert (
        cls.__name__
        == "StepLR"
    )


def test_unknown_scheduler():
    with pytest.raises(KeyError):
        get_scheduler_class(
            "unknown"
        )


def test_build_step_scheduler(
    optimizer,
):
    scheduler = build_scheduler(
        optimizer,
        name="step",
        step_size=5,
        gamma=0.1,
    )

    assert (
        scheduler.__class__.__name__
        == "StepLR"
    )


def test_build_multistep_scheduler(
    optimizer,
):
    scheduler = build_scheduler(
        optimizer,
        name="multistep",
        milestones=[5, 10],
        gamma=0.1,
    )

    assert (
        scheduler.__class__.__name__
        == "MultiStepLR"
    )


def test_build_cosine_scheduler(
    optimizer,
):
    scheduler = build_scheduler(
        optimizer,
        name="cosine",
        T_max=10,
    )

    assert (
        scheduler.__class__.__name__
        == "CosineAnnealingLR"
    )


def test_build_plateau_scheduler(
    optimizer,
):
    scheduler = build_scheduler(
        optimizer,
        name="plateau",
        factor=0.5,
        patience=2,
    )

    assert (
        scheduler.__class__.__name__
        == "ReduceLROnPlateau"
    )


def test_build_scheduler_from_config(
    optimizer,
):
    config = {
        "name": "step",
        "step_size": 5,
        "gamma": 0.1,
    }

    scheduler = (
        build_scheduler_from_config(
            optimizer,
            config,
        )
    )

    assert (
        scheduler.__class__.__name__
        == "StepLR"
    )


def test_scheduler_missing_name(
    optimizer,
):
    with pytest.raises(KeyError):
        build_scheduler_from_config(
            optimizer,
            {
                "gamma": 0.1
            },
        )


def test_get_learning_rate(
    optimizer,
):
    lr = get_learning_rate(
        optimizer
    )

    assert lr == 0.01


def test_step_scheduler_updates_lr(
    optimizer,
):
    scheduler = build_scheduler(
        optimizer,
        name="step",
        step_size=1,
        gamma=0.1,
    )

    before = get_learning_rate(
        optimizer
    )

    scheduler.step()

    after = get_learning_rate(
        optimizer
    )

    assert after < before


def test_plateau_scheduler_step(
    optimizer,
):
    scheduler = build_scheduler(
        optimizer,
        name="plateau",
        factor=0.5,
        patience=1,
    )

    scheduler.step(1.0)

    assert True


def test_scheduler_to_dict(
    optimizer,
):
    scheduler = build_scheduler(
        optimizer,
        name="step",
        step_size=5,
    )

    info = scheduler_to_dict(
        scheduler
    )

    assert (
        info["scheduler"]
        == "StepLR"
    )