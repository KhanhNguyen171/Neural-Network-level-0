"""
simple_cnn.py

CNN From Scratch

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
"""

from __future__ import annotations

import numpy as np

from .convolution import Convolution2D
from .activation import ReLU, Softmax
from .pooling import MaxPooling2D
from .flatten import Flatten
from .dense import Dense


class SimpleCNN:

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        num_filters: int = 8,
        kernel_size: int = 3,
        pool_size: int = 2,
        random_seed: int | None = None
    ):

        channels, height, width = input_shape

        # Feature Extractor

        self.conv = Convolution2D(
            in_channels=channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=1
        )

        self.relu = ReLU()

        self.pool = MaxPooling2D(
            pool_size=pool_size,
            stride=pool_size
        )

        self.flatten = Flatten()

        # Calculate Flatten Size

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

    # Forward Logits

    def forward_logits(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        x = self.conv.forward(x)

        x = self.relu.forward(x)

        x = self.pool.forward(x)

        x = self.flatten.forward(x)

        logits = self.dense.forward(x)

        self.logits = logits

        return logits

    # Prediction

    def predict(
        self,
        x: np.ndarray
    ) -> int:

        probabilities = self.forward(x)

        return int(
            np.argmax(probabilities)
        )

    def predict_proba(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        return self.forward(x)

    # Optimizer Support

    def parameters(self):

        params = []

        params.extend(
            self.conv.parameters()
        )

        params.extend(
            self.dense.parameters()
        )

        return params

    def zero_grad(self):

        self.conv.zero_grad()

        self.dense.zero_grad()

    # Summary

    def summary(self):

        print("\n" + "=" * 50)
        print("SimpleCNN")
        print("=" * 50)

        self.conv.summary()

        print()

        self.pool.summary()

        print()

        self.dense.summary()

        try:

            total_params = (
                self.conv.num_parameters
                + self.dense.num_parameters
            )

            print("\nTotal Parameters")
            print("-" * 50)
            print(total_params)

        except AttributeError:

            pass

        print("=" * 50)

    # Phase B

    def backward(
        self,
        grad_output: np.ndarray
    ):

        raise NotImplementedError(
            "Implemented in notebook 16."
        )

    def update(
        self,
        learning_rate: float
    ):


        self.dense.update(
            learning_rate
        )


    # Utility

    @property
    def num_parameters(self):

        total = 0

        if hasattr(
            self.conv,
            "num_parameters"
        ):
            total += (
                self.conv.num_parameters
            )

        if hasattr(
            self.dense,
            "num_parameters"
        ):
            total += (
                self.dense.num_parameters
            )

        return total

    def __repr__(self):

        return (
            f"SimpleCNN("
            f"parameters={self.num_parameters}"
            f")"
        )