from dataclasses import dataclass
from typing import List


@dataclass
class PredictionResult:
    prediction: int
    confidence: float


@dataclass
class TopKPrediction:
    class_id: int
    probability: float


@dataclass
class PredictionResponse:
    prediction: int
    confidence: float
    probabilities: List[float]


@dataclass
class TopKResponse:
    predictions: List[TopKPrediction]


@dataclass
class ErrorResponse:
    error: str