import pytest
import torch.nn as nn

from src.training.losses import (
    available_losses,
    get_loss_class,
    build_loss,
    build_loss_from_config,
    get_loss_function,
)

# pytest tests/training/test_losses.py -v

def test_available_losses():

    losses = available_losses()

    assert "cross_entropy" in losses
    assert "mse" in losses


def test_get_loss_class():

    cls = get_loss_class(
        "cross_entropy"
    )

    assert cls == nn.CrossEntropyLoss


def test_unknown_loss():

    with pytest.raises(
        ValueError
    ):
        get_loss_class(
            "unknown"
        )


def test_build_loss():

    loss = build_loss(
        "cross_entropy"
    )

    assert isinstance(
        loss,
        nn.CrossEntropyLoss,
    )


def test_build_loss_from_config():

    config = {
        "name":
            "cross_entropy"
    }

    loss = (
        build_loss_from_config(
            config
        )
    )

    assert isinstance(
        loss,
        nn.CrossEntropyLoss,
    )


def test_get_loss_function():

    loss = get_loss_function(
        "mse"
    )

    assert isinstance(
        loss,
        nn.MSELoss,
    )