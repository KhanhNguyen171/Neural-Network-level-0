"""
activation.py

CNN From Scratch

Activation Functions

- ReLU
- Sigmoid
- Tanh
- Softmax

All activations support:

- forward()
- backward()
- parameters()
- zero_grad()
"""

from __future__ import annotations

import numpy as np


# ReLU

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

        if self.input_cache is None:
            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        grad_input = grad_output.copy()

        grad_input[
            self.input_cache <= 0
        ] = 0

        return grad_input

    def parameters(self):

        return []

    def zero_grad(self):

        pass

    def __repr__(self):

        return "ReLU()"


# Sigmoid

class Sigmoid:
    """
    Sigmoid Activation

    Forward:
        σ(x)

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

        x = np.clip(
            x,
            -500,
            500
        )

        self.input_cache = x

        output = (
            1.0
            /
            (
                1.0
                + np.exp(-x)
            )
        )

        self.output_cache = output

        return output

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:

        if self.output_cache is None:
            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        sigmoid = self.output_cache

        grad_input = (
            grad_output
            * sigmoid
            * (1.0 - sigmoid)
        )

        return grad_input

    def parameters(self):

        return []

    def zero_grad(self):

        pass

    def __repr__(self):

        return "Sigmoid()"


# Tanh

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

        if self.output_cache is None:
            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        grad_input = (
            grad_output
            * (
                1.0
                - self.output_cache ** 2
            )
        )

        return grad_input

    def parameters(self):

        return []

    def zero_grad(self):

        pass

    def __repr__(self):

        return "Tanh()"


# Softmax

class Softmax:
    """
    Softmax Activation

    Forward:
        e^zi / Σ e^zj

    Backward:
        Full Jacobian

    Notes
    -----
    Educational implementation.

    For training classification models,
    prefer SoftmaxCrossEntropy in
    losses.py
    """

    def __init__(self) -> None:

        self.input_cache = None
        self.output_cache = None

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        self.input_cache = x

        shifted = (
            x
            - np.max(
                x,
                axis=-1,
                keepdims=True
            )
        )

        exp_values = np.exp(
            shifted
        )

        output = (
            exp_values
            /
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
        Jacobian-based backward.

        Useful for learning.
        """

        if self.output_cache is None:
            raise RuntimeError(
                "forward() must be called "
                "before backward()."
            )

        probs = self.output_cache

        if probs.ndim == 1:

            p = probs.reshape(-1)

            jacobian = (
                np.diag(p)
                -
                np.outer(p, p)
            )

            return (
                jacobian
                @ grad_output
            )

        grad_input = np.zeros_like(
            probs
        )

        batch_size = probs.shape[0]

        for i in range(batch_size):

            p = probs[i]

            jacobian = (
                np.diag(p)
                -
                np.outer(p, p)
            )

            grad_input[i] = (
                jacobian
                @ grad_output[i]
            )

        return grad_input

    def parameters(self):

        return []

    def zero_grad(self):

        pass

    def __repr__(self):

        return "Softmax()"


# Factory

def get_activation(
    name: str
):
    """
    Activation Factory

    Example
    -------
    relu = get_activation(
        "relu"
    )
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