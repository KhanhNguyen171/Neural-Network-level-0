"""
losses.py

CNN From Scratch

Loss Functions

- MSELoss
- CrossEntropyLoss
- SoftmaxCrossEntropy
"""

from __future__ import annotations

import numpy as np


class MSELoss:
    """
    Mean Squared Error

    L = (1/n) * Σ(y_pred - y_true)^2
    """

    def __init__(self) -> None:

        self.prediction = None
        self.target = None

        self.loss_cache = None

    def forward(
        self,
        prediction: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        Compute MSE Loss.
        """

        prediction = np.asarray(
            prediction,
            dtype=np.float64
        )

        target = np.asarray(
            target,
            dtype=np.float64
        )

        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target "
                "must have the same shape."
            )

        self.prediction = prediction
        self.target = target

        loss = np.mean(
            (prediction - target) ** 2
        )

        self.loss_cache = loss

        return float(loss)

    def backward(self) -> np.ndarray:
        """
        dL/dY =
        2(y_pred - y_true)
        ------------------
                n
        """

        if self.prediction is None:
            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        n = self.prediction.size

        grad = (
            2.0
            * (self.prediction - self.target)
            / n
        )

        return grad

    def __repr__(self) -> str:

        return "MSELoss()"


class CrossEntropyLoss:
    """
    Cross Entropy Loss

    Input:
        probabilities

    Target:
        one-hot vector
    """

    def __init__(self) -> None:

        self.prediction = None
        self.target = None

        self.loss_cache = None

        self.eps = 1e-12

    def forward(
        self,
        prediction: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        Compute Cross Entropy Loss.
        """

        prediction = np.asarray(
            prediction,
            dtype=np.float64
        )

        target = np.asarray(
            target,
            dtype=np.float64
        )

        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target "
                "must have the same shape."
            )

        self.prediction = prediction
        self.target = target

        loss = -np.sum(
            target
            * np.log(
                prediction + self.eps
            )
        )

        self.loss_cache = loss

        return float(loss)

    def backward(self) -> np.ndarray:
        """
        dL/dY

               -target
        ---------------------
          prediction + eps
        """

        if self.prediction is None:
            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        grad = (
            -self.target
            / (
                self.prediction
                + self.eps
            )
        )

        return grad

    def __repr__(self) -> str:

        return "CrossEntropyLoss()"


class SoftmaxCrossEntropy:
    """
    Softmax + Cross Entropy

    Input:
        logits

    Target:
        one-hot vector

    Backward:
        softmax(logits) - target
    """

    def __init__(self) -> None:

        self.logits = None
        self.probabilities = None
        self.target = None

        self.loss_cache = None

        self.eps = 1e-12

    def forward(
        self,
        logits: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        Compute:

        logits
            ↓
        softmax
            ↓
        cross entropy
        """

        logits = np.asarray(
            logits,
            dtype=np.float64
        )

        target = np.asarray(
            target,
            dtype=np.float64
        )

        if logits.shape != target.shape:
            raise ValueError(
                "logits and target "
                "must have the same shape."
            )

        self.logits = logits
        self.target = target

        shifted = (
            logits
            - np.max(logits)
        )

        exp_values = np.exp(
            shifted
        )

        probabilities = (
            exp_values
            / np.sum(exp_values)
        )

        self.probabilities = probabilities

        loss = -np.sum(
            target
            * np.log(
                probabilities
                + self.eps
            )
        )

        self.loss_cache = loss

        return float(loss)

    def backward(self) -> np.ndarray:
        """
        dL/dZ

        =
        softmax(z)
        -
        target
        """

        if self.probabilities is None:
            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        return (
            self.probabilities
            - self.target
        )

    def __repr__(self) -> str:

        return (
            "SoftmaxCrossEntropy()"
        )