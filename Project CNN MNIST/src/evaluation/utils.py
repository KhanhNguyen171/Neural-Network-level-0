from __future__ import annotations

from typing import Tuple

import torch


def validate_predictions(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> None:
    """
    Validate prediction tensors.
    """
    if y_true.numel() == 0:
        raise ValueError(
            "y_true cannot be empty."
        )

    if y_pred.numel() == 0:
        raise ValueError(
            "y_pred cannot be empty."
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have same shape."
        )


def move_to_device(
    tensor: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Move tensor to device.
    """
    return tensor.to(device)


def logits_to_predictions(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Convert logits to class predictions.
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits must be 2-dimensional."
        )

    return logits.argmax(dim=1)


def logits_to_probabilities(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Convert logits to probabilities.
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits must be 2-dimensional."
        )

    return torch.softmax(
        logits,
        dim=1,
    )


def batch_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute batch accuracy.
    """
    predictions = logits_to_predictions(
        logits
    )

    correct = (
        predictions == targets
    ).sum().item()

    return correct / len(targets)


def count_correct(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> int:
    """
    Count correct predictions.
    """
    validate_predictions(
        targets,
        predictions,
    )

    return int(
        (predictions == targets)
        .sum()
        .item()
    )


def prediction_distribution(
    predictions: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Count predictions per class.
    """
    if num_classes <= 0:
        raise ValueError(
            "num_classes must be positive."
        )

    return torch.bincount(
        predictions,
        minlength=num_classes,
    )


def confidence_scores(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Maximum class probability.
    """
    probs = logits_to_probabilities(
        logits
    )

    return probs.max(
        dim=1
    ).values


def topk_predictions(
    logits: torch.Tensor,
    k: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k predictions.
    """
    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    probs = logits_to_probabilities(
        logits
    )

    scores, indices = torch.topk(
        probs,
        k=k,
        dim=1,
    )

    return indices, scores