from typing import Any, Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)


SCHEDULER_REGISTRY = {
    "step": StepLR,
    "multistep": MultiStepLR,
    "exponential": ExponentialLR,
    "cosine": CosineAnnealingLR,
    "plateau": ReduceLROnPlateau,
}


def available_schedulers():
    """
    Return supported scheduler names.
    """
    return list(
        SCHEDULER_REGISTRY.keys()
    )


def get_scheduler_class(
    name: str,
):
    """
    Retrieve scheduler class.
    """
    name = name.lower()

    if name not in SCHEDULER_REGISTRY:
        raise KeyError(
            f"Scheduler '{name}' not found."
        )

    return SCHEDULER_REGISTRY[name]


def build_scheduler(
    optimizer: Optimizer,
    name: str,
    **kwargs,
):
    """
    Build scheduler from name.
    """
    scheduler_cls = get_scheduler_class(
        name
    )

    return scheduler_cls(
        optimizer,
        **kwargs,
    )


def build_scheduler_from_config(
    optimizer: Optimizer,
    config: Dict[str, Any],
):
    """
    Build scheduler from config.

    Example
    -------
    {
        "name": "step",
        "step_size": 10,
        "gamma": 0.1
    }
    """
    if "name" not in config:
        raise KeyError(
            "Config must contain 'name'."
        )

    config = config.copy()

    name = config.pop("name")

    return build_scheduler(
        optimizer,
        name=name,
        **config,
    )


def get_learning_rate(
    optimizer: Optimizer,
) -> float:
    """
    Get current learning rate.
    """
    return optimizer.param_groups[0]["lr"]


def scheduler_to_dict(
    scheduler,
):
    """
    Convert scheduler info to dict.
    """
    return {
        "scheduler": scheduler.__class__.__name__,
    }