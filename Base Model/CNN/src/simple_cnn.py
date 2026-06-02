"""
simple_cnn.py

CNN From Scratch

Simple CNN Architecture

Image
 ↓
Conv
 ↓
ReLU
 ↓
Pool
 ↓
Flatten
 ↓
Dense
 ↓
Softmax

Author: CNN From Scratch
"""

from __future__ import annotations

import numpy as np

from .convolution import Convolution2D
from .activation import ReLU, Softmax
from .pooling import MaxPooling2D
from .flatten import Flatten
from .dense import Dense


class SimpleCNN:
    """
    A minimal CNN for educational purposes.

    Architecture:

    Input
        ↓
    Conv
        ↓
    ReLU
        ↓
    Pool
        ↓
    Flatten
        ↓
    Dense
        ↓
    Softmax
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        num_filters: int = 8,
        kernel_size: int = 3,
        pool_size: int = 2,
        random_seed: int | None = None
    ) -> None:

        channels, height, width = input_shape

        # Feature Extractor

        self.conv = Convolution2D(
            in_channels=channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            random_seed=random_seed
        )

        self.relu = ReLU()

        self.pool = MaxPooling2D(
            pool_size=pool_size,
            stride=pool_size
        )

        self.flatten = Flatten()

        # Compute Dense Input Size

        conv_h = (
            (height - kernel_size + 2)
            // 1
        ) + 1

        conv_w = (
            (width - kernel_size + 2)
            // 1
        ) + 1

        pool_h = (
            (conv_h - pool_size)
            // pool_size
        ) + 1

        pool_w = (
            (conv_w - pool_size)
            // pool_size
        ) + 1

        flattened_size = (
            num_filters
            * pool_h
            * pool_w
        )

        # Classifier

        self.dense = Dense(
            input_dim=flattened_size,
            output_dim=num_classes,
            random_seed=random_seed
        )

        self.softmax = Softmax()

        # Cache

        self.logits = None
        self.probabilities = None

    # Forward

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Forward Pass

        Input:
            (C,H,W)

        Output:
            probabilities
        """

        x = self.conv.forward(x)

        x = self.relu.forward(x)

        x = self.pool.forward(x)

        x = self.flatten.forward(x)

        logits = self.dense.forward(x)

        probabilities = self.softmax.forward(
            logits
        )

        self.logits = logits
        self.probabilities = probabilities

        return probabilities

    # Prediction

    def predict(
        self,
        x: np.ndarray
    ) -> int:
        """
        Predict class index.
        """

        probabilities = self.forward(x)

        return int(
            np.argmax(probabilities)
        )

    # Top-K Prediction

    def predict_proba(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        return self.forward(x)

    # Summary

    def summary(self) -> None:

        print("\n" + "=" * 50)
        print("SimpleCNN")
        print("=" * 50)

        self.conv.summary()

        self.pool.summary()

        self.dense.summary()

        total_params = (
            self.conv.num_parameters
            + self.dense.num_parameters
        )

        print("\nTotal Parameters")
        print("-" * 50)
        print(total_params)
        print("=" * 50)

    # Phase B

    def backward(
        self,
        grad_output: np.ndarray
    ) -> None:
        """
        Notebook 16

        Backpropagation order:

        Dense
            ↓
        Flatten
            ↓
        Pool
            ↓
        ReLU
            ↓
        Conv
        """

        raise NotImplementedError(
            "Implemented in notebook 16"
        )

    def update(
        self,
        learning_rate: float
    ) -> None:
        """
        Update trainable parameters.

        Notebook 16
        """

        self.dense.update(
            learning_rate
        )

        # Conv update sẽ được thêm
        # sau khi triển khai
        # convolution backward.

    # Utility

    @property
    def num_parameters(self) -> int:

        return (
            self.conv.num_parameters
            + self.dense.num_parameters
        )

    def __repr__(self) -> str:

        return (
            f"SimpleCNN("
            f"parameters={self.num_parameters}"
            f")"
        )