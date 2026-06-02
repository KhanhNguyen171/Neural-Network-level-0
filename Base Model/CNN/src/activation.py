"""
activation.py

CNN From Scratch

Activation Functions:
    - ReLU
    - Sigmoid
    - Tanh
    - Softmax

All activations support:
    - forward()
    - backward()

Author: CNN From Scratch
"""

from __future__ import annotations

import numpy as np


class ReLU:
    """
    ReLU Activation

    Forward:
        f(x) = max(0, x)

    Backward:
        1 if x > 0
        0 otherwise
    """

    def __init__(self) -> None:

        self.input_cache = None
        self.output_cache = None

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        self.input_cache = x

        output = np.maximum(0, x)

        self.output_cache = output

        return output

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:

        grad_input = grad_output.copy()

        grad_input[
            self.input_cache <= 0
        ] = 0

        return grad_input


class Sigmoid:
    """
    Sigmoid Activation

    Forward:
        σ(x) = 1 / (1 + e^-x)

    Backward:
        σ(x)(1-σ(x))
    """

    def __init__(self) -> None:

        self.input_cache = None
        self.output_cache = None

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        self.input_cache = x

        output = (
            1.0 /
            (1.0 + np.exp(-x))
        )

        self.output_cache = output

        return output

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:

        sigmoid = self.output_cache

        grad_input = (
            grad_output *
            sigmoid *
            (1.0 - sigmoid)
        )

        return grad_input


class Tanh:
    """
    Tanh Activation

    Forward:
        tanh(x)

    Backward:
        1 - tanh²(x)
    """

    def __init__(self) -> None:

        self.input_cache = None
        self.output_cache = None

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        self.input_cache = x

        output = np.tanh(x)

        self.output_cache = output

        return output

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:

        grad_input = (
            grad_output *
            (1 - self.output_cache ** 2)
        )

        return grad_input


class Softmax:
    """
    Softmax Activation

    Forward:
        e^zi / Σ e^zj

    Backward:
        Full Jacobian implementation

    Notes:
        In practice, Softmax is usually
        combined with Cross Entropy Loss.
    """

    def __init__(self) -> None:

        self.input_cache = None
        self.output_cache = None

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        self.input_cache = x

        shifted = x - np.max(
            x,
            axis=-1,
            keepdims=True
        )

        exp_values = np.exp(shifted)

        output = (
            exp_values /
            np.sum(
                exp_values,
                axis=-1,
                keepdims=True
            )
        )

        self.output_cache = output

        return output

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:
        """
        Generic Jacobian-based backward.

        Useful for learning purposes.
        """

        probabilities = self.output_cache

        grad_input = np.zeros_like(
            probabilities
        )

        if probabilities.ndim == 1:

            probabilities = probabilities.reshape(
                -1,
                1
            )

            jacobian = (
                np.diagflat(probabilities)
                -
                probabilities @ probabilities.T
            )

            grad_input = (
                jacobian @ grad_output
            )

            return grad_input

        batch_size = probabilities.shape[0]

        for i in range(batch_size):

            p = probabilities[i].reshape(
                -1,
                1
            )

            jacobian = (
                np.diagflat(p)
                -
                p @ p.T
            )

            grad_input[i] = (
                jacobian @ grad_output[i]
            )

        return grad_input


# Factory

def get_activation(
    name: str
):
    """
    Activation Factory

    Example:
        relu = get_activation("relu")
    """

    name = name.lower()

    if name == "relu":
        return ReLU()

    if name == "sigmoid":
        return Sigmoid()

    if name == "tanh":
        return Tanh()

    if name == "softmax":
        return Softmax()

    raise ValueError(
        f"Unknown activation: {name}"
    )