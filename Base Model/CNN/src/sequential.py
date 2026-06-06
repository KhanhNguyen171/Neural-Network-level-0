"""
sequential.py

CNN From Scratch

Sequential Model Container
"""

from __future__ import annotations

import numpy as np


class Sequential:
    """
    Sequential Container

    Example
    -------
    model = Sequential(
        [
            Dense(10, 20),
            ReLU(),
            Dense(20, 5)
        ]
    )
    """

    def __init__(
        self,
        layers: list
    ) -> None:

        self.layers = layers

    # Forward

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass through
        all layers.
        """

        for layer in self.layers:

            x = layer.forward(x)

        return x

    # Backward

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:
        """
        Backward pass through
        all layers.
        """

        grad = grad_output

        for layer in reversed(
            self.layers
        ):

            if hasattr(
                layer,
                "backward"
            ):

                grad = layer.backward(
                    grad
                )

        return grad

    # Parameters

    def parameters(self):
        """
        Collect all trainable
        parameters.
        """

        params = []

        for layer in self.layers:

            if hasattr(
                layer,
                "parameters"
            ):

                params.extend(
                    layer.parameters()
                )

        return params

    # Gradients

    def zero_grad(self) -> None:
        """
        Reset all gradients.
        """

        for layer in self.layers:

            if hasattr(
                layer,
                "zero_grad"
            ):

                layer.zero_grad()

    # Utilities

    def add(
        self,
        layer
    ) -> None:
        """
        Add layer dynamically.
        """

        self.layers.append(
            layer
        )

    def summary(self) -> None:
        """
        Print model summary.
        """

        print("\nSequential Model")
        print("=" * 50)

        total_params = 0

        for idx, layer in enumerate(
            self.layers,
            start=1
        ):

            print(
                f"{idx:02d}. "
                f"{layer}"
            )

            if hasattr(
                layer,
                "num_parameters"
            ):

                total_params += (
                    layer.num_parameters
                )

        print("-" * 50)

        print(
            f"Total Parameters: "
            f"{total_params}"
        )

        print("=" * 50)

    def __len__(self) -> int:

        return len(
            self.layers
        )

    def __getitem__(
        self,
        index: int
    ):

        return self.layers[index]

    def __repr__(self) -> str:

        layer_names = [
            layer.__class__.__name__
            for layer in self.layers
        ]

        return (
            "Sequential("
            + " -> ".join(
                layer_names
            )
            + ")"
        )