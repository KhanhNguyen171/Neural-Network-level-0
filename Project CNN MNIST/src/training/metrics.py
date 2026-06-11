# src/training/metrics.py

from __future__ import annotations

from typing import Dict

import torch


def accuracy_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    logits : torch.Tensor
        Shape:
            (N, C)

    targets : torch.Tensor
        Shape:
            (N,)

    Returns
    -------
    float
        Accuracy in range [0, 1]
    """

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    correct = (predictions == targets).sum().item()

    total = targets.size(0)

    return correct / total


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Compute Top-K accuracy.

    Parameters
    ----------
    logits : torch.Tensor
        Shape:
            (N, C)

    targets : torch.Tensor
        Shape:
            (N,)

    k : int

    Returns
    -------
    float
    """

    k = min(
        k,
        logits.size(1),
    )

    topk_predictions = torch.topk(
        logits,
        k=k,
        dim=1,
    ).indices

    correct = (
        topk_predictions
        == targets.unsqueeze(1)
    ).any(dim=1)

    return correct.float().mean().item()


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    logits : torch.Tensor

    targets : torch.Tensor

    Returns
    -------
    dict
    """

    return {
        "accuracy": accuracy_score(
            logits,
            targets,
        )
    }


class AverageMeter:
    """
    Running average tracker.

    Example
    -------
    meter = AverageMeter()

    meter.update(loss, batch_size)

    avg_loss = meter.avg
    """

    def __init__(self) -> None:

        self.reset()

    def reset(self) -> None:

        self.value = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(
        self,
        value: float,
        n: int = 1,
    ) -> None:

        self.value = float(value)

        self.sum += value * n

        self.count += n

        self.avg = self.sum / self.count

    def __repr__(self) -> str:

        return (
            f"AverageMeter("
            f"value={self.value:.4f}, "
            f"avg={self.avg:.4f})"
        )