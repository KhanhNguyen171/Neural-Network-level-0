from typing import Dict, Any

import torch
from torch import nn
from torch.optim import Optimizer


OPTIMIZER_REGISTRY = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}


def available_optimizers():
    """
    Return supported optimizer names.
    """
    return list(OPTIMIZER_REGISTRY.keys())


def get_optimizer_class(name: str):
    """
    Retrieve optimizer class by name.
    """
    name = name.lower()

    if name not in OPTIMIZER_REGISTRY:
        raise KeyError(
            f"Optimizer '{name}' not found."
        )

    return OPTIMIZER_REGISTRY[name]


def build_optimizer(
    model: nn.Module,
    name: str = "adam",
    **kwargs,
) -> Optimizer:
    """
    Build optimizer from model.
    """
    optimizer_cls = get_optimizer_class(name)

    params = (
        p
        for p in model.parameters()
        if p.requires_grad
    )

    return optimizer_cls(
        params,
        **kwargs,
    )


def build_optimizer_from_config(
    model: nn.Module,
    config: Dict[str, Any],
) -> Optimizer:
    """
    Build optimizer from config dict.

    Example
    -------
    {
        "name": "adam",
        "lr": 1e-3,
        "weight_decay": 1e-4
    }
    """
    if "name" not in config:
        raise KeyError(
            "Config must contain 'name'."
        )

    config = config.copy()

    name = config.pop("name")

    return build_optimizer(
        model=model,
        name=name,
        **config,
    )


def optimizer_to_dict(
    optimizer: Optimizer,
):
    """
    Convert optimizer settings to dict.
    """
    group = optimizer.param_groups[0]

    return {
        "lr": group.get("lr"),
        "weight_decay": group.get(
            "weight_decay",
            0.0,
        ),
    }


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable parameters.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )