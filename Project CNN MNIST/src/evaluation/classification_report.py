from __future__ import annotations

from typing import Dict

import torch

from .metrics import (
    precision,
    recall,
    f1_score,
    accuracy,
)


def classification_report(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> Dict:
    """
    Generate a classification report.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth labels.
    y_pred : torch.Tensor
        Predicted labels.
    num_classes : int
        Number of classes.

    Returns
    -------
    Dict
        Report dictionary.
    """
    if y_true.numel() == 0:
        raise ValueError("y_true cannot be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape.")

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    report = {}

    for cls in range(num_classes):
        true_mask = y_true == cls
        pred_mask = y_pred == cls

        tp = (true_mask & pred_mask).sum().item()
        fp = (~true_mask & pred_mask).sum().item()
        fn = (true_mask & ~pred_mask).sum().item()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        support = int(true_mask.sum().item())

        report[str(cls)] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1-score": float(f1),
            "support": support,
        }

    macro_precision = sum(
        report[str(i)]["precision"]
        for i in range(num_classes)
    ) / num_classes

    macro_recall = sum(
        report[str(i)]["recall"]
        for i in range(num_classes)
    ) / num_classes

    macro_f1 = sum(
        report[str(i)]["f1-score"]
        for i in range(num_classes)
    ) / num_classes

    report["accuracy"] = float(
        accuracy(y_true, y_pred)
    )

    report["macro avg"] = {
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1-score": float(macro_f1),
        "support": int(len(y_true)),
    }

    return report


def report_to_string(
    report: Dict,
) -> str:
    """
    Convert report dictionary to printable string.
    """
    lines = []

    header = (
        f"{'class':<12}"
        f"{'precision':>12}"
        f"{'recall':>12}"
        f"{'f1-score':>12}"
        f"{'support':>12}"
    )

    lines.append(header)
    lines.append("-" * len(header))

    for key, value in report.items():
        if key in {"accuracy", "macro avg"}:
            continue

        lines.append(
            f"{key:<12}"
            f"{value['precision']:>12.4f}"
            f"{value['recall']:>12.4f}"
            f"{value['f1-score']:>12.4f}"
            f"{value['support']:>12}"
        )

    macro = report["macro avg"]

    lines.append("")
    lines.append(
        f"{'macro avg':<12}"
        f"{macro['precision']:>12.4f}"
        f"{macro['recall']:>12.4f}"
        f"{macro['f1-score']:>12.4f}"
        f"{macro['support']:>12}"
    )

    lines.append("")
    lines.append(
        f"accuracy: {report['accuracy']:.4f}"
    )

    return "\n".join(lines)