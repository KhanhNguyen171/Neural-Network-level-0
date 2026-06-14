import torch
import torch.nn as nn


LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "nll": nn.NLLLoss,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "bce": nn.BCELoss,
    "bce_logits": nn.BCEWithLogitsLoss,
    "smooth_l1": nn.SmoothL1Loss,
}


def available_losses():
    """
    Return available loss names.
    """
    return sorted(
        LOSS_REGISTRY.keys()
    )


def get_loss_class(
    name: str,
):
    """
    Get loss class by name.
    """

    name = name.lower()

    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: {name}"
        )

    return LOSS_REGISTRY[name]


def build_loss(
    name: str,
    **kwargs,
):
    """
    Build loss instance.
    """

    loss_cls = get_loss_class(
        name
    )

    return loss_cls(
        **kwargs
    )


def build_loss_from_config(
    config: dict,
):
    """
    Build loss from config.

    Example
    -------
    {
        "name": "cross_entropy"
    }
    """

    if "name" not in config:
        raise KeyError(
            "Loss config must contain 'name'"
        )

    config = dict(config)

    name = config.pop(
        "name"
    )

    return build_loss(
        name,
        **config,
    )


def loss_to_dict(
    criterion,
):
    """
    Convert loss object to dict.
    """

    return {
        "type":
            criterion.__class__.__name__
    }


def get_loss_function(
    name: str,
    **kwargs,
):
    """
    Alias used by train.py
    """

    return build_loss(
        name,
        **kwargs,
    )