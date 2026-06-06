"""
optimizers.py

CNN From Scratch

Optimizers

- SGD
- Momentum
- Adam
"""

from __future__ import annotations

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent

    θ = θ - lr * gradient
    """

    def __init__(
        self,
        parameters,
        lr: float = 0.01
    ) -> None:

        self.parameters = list(
            parameters
        )

        self.lr = lr

    def step(self) -> None:
        """
        Update parameters.
        """

        for param, grad in self.parameters:

            param -= (
                self.lr
                * grad
            )

    def zero_grad(self) -> None:
        """
        Reset gradients.
        """

        for _, grad in self.parameters:

            grad.fill(0)

    def __repr__(self) -> str:

        return (
            f"SGD(lr={self.lr})"
        )


class Momentum:
    """
    SGD + Momentum

    v = beta * v - lr * grad

    param += v
    """

    def __init__(
        self,
        parameters,
        lr: float = 0.01,
        beta: float = 0.9
    ) -> None:

        self.parameters = list(
            parameters
        )

        self.lr = lr
        self.beta = beta

        self.velocity = []

        for param, _ in self.parameters:

            self.velocity.append(
                np.zeros_like(param)
            )

    def step(self) -> None:

        for i, (param, grad) in enumerate(
            self.parameters
        ):

            self.velocity[i] = (
                self.beta
                * self.velocity[i]
                -
                self.lr
                * grad
            )

            param += (
                self.velocity[i]
            )

    def zero_grad(self) -> None:

        for _, grad in self.parameters:

            grad.fill(0)

    def __repr__(self) -> str:

        return (
            f"Momentum("
            f"lr={self.lr}, "
            f"beta={self.beta}"
            f")"
        )


class Adam:
    """
    Adam Optimizer

    Adaptive Moment Estimation
    """

    def __init__(
        self,
        parameters,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ) -> None:

        self.parameters = list(
            parameters
        )

        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2

        self.eps = eps

        self.t = 0

        self.m = []
        self.v = []

        for param, _ in self.parameters:

            self.m.append(
                np.zeros_like(param)
            )

            self.v.append(
                np.zeros_like(param)
            )

    def step(self) -> None:

        self.t += 1

        for i, (param, grad) in enumerate(
            self.parameters
        ):

            self.m[i] = (
                self.beta1
                * self.m[i]
                +
                (1 - self.beta1)
                * grad
            )

            self.v[i] = (
                self.beta2
                * self.v[i]
                +
                (1 - self.beta2)
                * (grad ** 2)
            )

            m_hat = (
                self.m[i]
                /
                (
                    1
                    - self.beta1 ** self.t
                )
            )

            v_hat = (
                self.v[i]
                /
                (
                    1
                    - self.beta2 ** self.t
                )
            )

            param -= (
                self.lr
                * m_hat
                /
                (
                    np.sqrt(v_hat)
                    + self.eps
                )
            )

    def zero_grad(self) -> None:

        for _, grad in self.parameters:

            grad.fill(0)

    def __repr__(self) -> str:

        return (
            f"Adam("
            f"lr={self.lr}, "
            f"beta1={self.beta1}, "
            f"beta2={self.beta2}"
            f")"
        )