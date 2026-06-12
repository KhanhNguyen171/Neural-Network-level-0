# src/evaluation/confusion_matrix.py

from __future__ import annotations

import torch


def compute_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels.

    y_pred : torch.Tensor
        Predicted labels.

    num_classes : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        Shape (num_classes, num_classes)

        Rows    : true labels
        Columns : predicted labels
    """

    if y_true.numel() == 0:
        raise ValueError(
            "y_true cannot be empty."
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have same shape."
        )

    if num_classes <= 0:
        raise ValueError(
            "num_classes must be positive."
        )

    cm = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )

    for true_label, pred_label in zip(
        y_true,
        y_pred,
    ):
        cm[
            int(true_label),
            int(pred_label),
        ] += 1

    return cm


def normalize_confusion_matrix(
    confusion_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Row-wise normalization.

    Each row sums to 1.
    """

    if confusion_matrix.ndim != 2:
        raise ValueError(
            "confusion_matrix must be 2D."
        )

    row_sums = confusion_matrix.sum(
        dim=1,
        keepdim=True,
    )

    row_sums = torch.where(
        row_sums == 0,
        torch.ones_like(row_sums),
        row_sums,
    )

    return (
        confusion_matrix.float()
        / row_sums.float()
    )


def per_class_accuracy(
    confusion_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-class accuracy.

    Returns
    -------
    torch.Tensor
        Shape (num_classes,)
    """

    if confusion_matrix.ndim != 2:
        raise ValueError(
            "confusion_matrix must be 2D."
        )

    correct = torch.diag(
        confusion_matrix
    ).float()

    totals = confusion_matrix.sum(
        dim=1
    ).float()

    totals = torch.where(
        totals == 0,
        torch.ones_like(totals),
        totals,
    )

    return correct / totals
