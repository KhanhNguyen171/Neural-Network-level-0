# src/evaluation/metrics.py

from __future__ import annotations

from typing import Dict

import torch


def accuracy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """
    Classification accuracy.

    Parameters
    ----------
    y_true : torch.Tensor
    y_pred : torch.Tensor

    Returns
    -------
    float
    """

    if y_true.numel() == 0:
        raise ValueError(
            "y_true cannot be empty."
        )

    return float(
        (y_true == y_pred)
        .float()
        .mean()
        .item()
    )


def precision(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_id: int,
) -> float:
    """
    Precision for a specific class.
    """

    predicted_positive = (
        y_pred == class_id
    )

    tp = (
        predicted_positive
        & (y_true == class_id)
    ).sum().item()

    fp = (
        predicted_positive
        & (y_true != class_id)
    ).sum().item()

    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return tp / denominator


def recall(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_id: int,
) -> float:
    """
    Recall for a specific class.
    """

    actual_positive = (
        y_true == class_id
    )

    tp = (
        actual_positive
        & (y_pred == class_id)
    ).sum().item()

    fn = (
        actual_positive
        & (y_pred != class_id)
    ).sum().item()

    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return tp / denominator


def f1_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_id: int,
) -> float:
    """
    F1 score for a specific class.
    """

    p = precision(
        y_true,
        y_pred,
        class_id,
    )

    r = recall(
        y_true,
        y_pred,
        class_id,
    )

    denominator = p + r

    if denominator == 0:
        return 0.0

    return (
        2.0
        * p
        * r
        / denominator
    )


def macro_precision(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """
    Macro precision.
    """

    classes = torch.unique(y_true)

    scores = [
        precision(
            y_true,
            y_pred,
            int(cls),
        )
        for cls in classes
    ]

    return float(sum(scores) / len(scores))


def macro_recall(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """
    Macro recall.
    """

    classes = torch.unique(y_true)

    scores = [
        recall(
            y_true,
            y_pred,
            int(cls),
        )
        for cls in classes
    ]

    return float(sum(scores) / len(scores))


def macro_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """
    Macro F1 score.
    """

    classes = torch.unique(y_true)

    scores = [
        f1_score(
            y_true,
            y_pred,
            int(cls),
        )
        for cls in classes
    ]

    return float(sum(scores) / len(scores))


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Top-k accuracy.
    """

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    _, indices = torch.topk(
        logits,
        k,
        dim=1,
    )

    correct = (
        indices
        == targets.unsqueeze(1)
    )

    return float(
        correct.any(dim=1)
        .float()
        .mean()
        .item()
    )


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute common evaluation metrics.
    """

    return {
        "accuracy": accuracy(
            y_true,
            y_pred,
        ),
        "macro_precision": macro_precision(
            y_true,
            y_pred,
        ),
        "macro_recall": macro_recall(
            y_true,
            y_pred,
        ),
        "macro_f1": macro_f1(
            y_true,
            y_pred,
        ),
    }