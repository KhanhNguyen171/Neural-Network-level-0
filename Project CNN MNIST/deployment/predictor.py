from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.cnn import MNISTCNN
from deployment.schemas import (
    PredictionResult,
    TopKPrediction,
)


class Predictor:
    """
    High-level prediction interface for deployment.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.model = MNISTCNN()
        self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.1307,),
                    std=(0.3081,),
                ),
            ]
        )

    def _load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(
                    checkpoint["model_state_dict"]
                )

            elif "state_dict" in checkpoint:
                self.model.load_state_dict(
                    checkpoint["state_dict"]
                )

            else:
                self.model.load_state_dict(checkpoint)

        else:
            self.model.load_state_dict(checkpoint)

    def preprocess_image(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        tensor = self.transform(image)

        return tensor.unsqueeze(0)

    def predict_tensor(
        self,
        tensor: torch.Tensor,
    ) -> PredictionResult:
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

            probs = F.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = torch.max(
                probs,
                dim=1,
            )

        return PredictionResult(
            prediction=int(prediction.item()),
            confidence=float(confidence.item()),
        )

    def predict_image(
        self,
        image: Image.Image,
    ) -> PredictionResult:
        tensor = self.preprocess_image(image)

        return self.predict_tensor(tensor)

    def predict_file(
        self,
        image_path: str | Path,
    ) -> PredictionResult:
        image = Image.open(image_path)

        return self.predict_image(image)

    def predict_proba(
        self,
        image: Image.Image,
    ) -> List[float]:
        tensor = self.preprocess_image(image)

        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

            probs = F.softmax(
                logits,
                dim=1,
            )

        return probs.squeeze(0).cpu().tolist()

    def predict_topk(
        self,
        image: Image.Image,
        k: int = 3,
    ) -> List[TopKPrediction]:
        tensor = self.preprocess_image(image)

        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

            probs = F.softmax(
                logits,
                dim=1,
            )

            values, indices = torch.topk(
                probs,
                k=k,
                dim=1,
            )

        values = values.squeeze(0)
        indices = indices.squeeze(0)

        results = []

        for cls, prob in zip(
            indices.tolist(),
            values.tolist(),
        ):
            results.append(
                TopKPrediction(
                    class_id=int(cls),
                    probability=float(prob),
                )
            )

        return results

    def batch_predict(
        self,
        images: List[Image.Image],
    ) -> List[PredictionResult]:
        tensors = [
            self.transform(img)
            for img in images
        ]

        batch = torch.stack(tensors)
        batch = batch.to(self.device)

        with torch.no_grad():
            logits = self.model(batch)

            probs = F.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = torch.max(
                probs,
                dim=1,
            )

        results = []

        for pred, conf in zip(
            prediction.tolist(),
            confidence.tolist(),
        ):
            results.append(
                PredictionResult(
                    prediction=int(pred),
                    confidence=float(conf),
                )
            )

        return results