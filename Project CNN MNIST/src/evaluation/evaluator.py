# src/evaluation/evaluator.py

from __future__ import annotations

from typing import Dict
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import (
    accuracy,
    macro_precision,
    macro_recall,
    macro_f1,
)


class Evaluator:
    """
    Model evaluation utility.

    Supports:
    - prediction
    - probability prediction
    - evaluation metrics

    Example
    -------
    >>> evaluator = Evaluator(model)
    >>> metrics = evaluator.evaluate(loader)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        criterion: Optional[nn.Module] = None,
    ) -> None:

        self.model = model
        self.device = torch.device(device)
        self.criterion = criterion

        self.model.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
    ) -> torch.Tensor:
        """
        Predict class labels.
        """

        self.model.eval()

        predictions = []

        for inputs, _ in dataloader:

            inputs = inputs.to(self.device)

            outputs = self.model(inputs)

            preds = outputs.argmax(dim=1)

            predictions.append(
                preds.cpu()
            )

        return torch.cat(
            predictions,
            dim=0,
        )

    @torch.no_grad()
    def predict_proba(
        self,
        dataloader: DataLoader,
    ) -> torch.Tensor:
        """
        Predict class probabilities.
        """

        self.model.eval()

        probabilities = []

        for inputs, _ in dataloader:

            inputs = inputs.to(self.device)

            outputs = self.model(inputs)

            probs = torch.softmax(
                outputs,
                dim=1,
            )

            probabilities.append(
                probs.cpu()
            )

        return torch.cat(
            probabilities,
            dim=0,
        )

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate model.
        """

        self.model.eval()

        y_true = []
        y_pred = []

        total_loss = 0.0
        total_samples = 0

        for inputs, targets in dataloader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)

            preds = outputs.argmax(dim=1)

            y_true.append(
                targets.cpu()
            )

            y_pred.append(
                preds.cpu()
            )

            if self.criterion is not None:

                loss = self.criterion(
                    outputs,
                    targets,
                )

                batch_size = inputs.size(0)

                total_loss += (
                    loss.item()
                    * batch_size
                )

                total_samples += (
                    batch_size
                )

        y_true = torch.cat(
            y_true,
            dim=0,
        )

        y_pred = torch.cat(
            y_pred,
            dim=0,
        )

        metrics = {
            "accuracy": accuracy(
                y_true,
                y_pred,
            ),
            "macro_precision": (
                macro_precision(
                    y_true,
                    y_pred,
                )
            ),
            "macro_recall": (
                macro_recall(
                    y_true,
                    y_pred,
                )
            ),
            "macro_f1": macro_f1(
                y_true,
                y_pred,
            ),
        }

        if (
            self.criterion is not None
            and total_samples > 0
        ):
            metrics["loss"] = (
                total_loss
                / total_samples
            )

        return metrics

    def __repr__(self) -> str:

        return (
            f"Evaluator("
            f"model={self.model.__class__.__name__}, "
            f"device='{self.device}'"
            f")"
        )
