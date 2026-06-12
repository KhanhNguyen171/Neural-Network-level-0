from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


class InferenceEngine:
    """
    Lightweight inference wrapper.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict class indices.
        """
        inputs = inputs.to(self.device)

        logits = self.model(inputs)

        return logits.argmax(dim=1)

    @torch.no_grad()
    def predict_proba(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict probabilities.
        """
        inputs = inputs.to(self.device)

        logits = self.model(inputs)

        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_topk(
        self,
        inputs: torch.Tensor,
        k: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        Top-k predictions.
        """
        if k <= 0:
            raise ValueError("k must be positive.")

        probs = self.predict_proba(inputs)

        scores, indices = torch.topk(
            probs,
            k=k,
            dim=1,
        )

        return {
            "indices": indices,
            "scores": scores,
        }

    @torch.no_grad()
    def predict_single(
        self,
        image: torch.Tensor,
    ) -> int:
        """
        Predict a single image.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        prediction = self.predict(image)

        return int(prediction.item())

    def __repr__(self) -> str:
        return (
            f"InferenceEngine("
            f"device='{self.device}'"
            f")"
        )