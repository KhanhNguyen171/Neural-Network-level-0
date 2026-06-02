"""
flatten.py

CNN From Scratch

Flatten Layer

Forward:
    (C, H, W) -> (N)

Backward:
    (N) -> (C, H, W)

Author: CNN From Scratch
"""

from __future__ import annotations

import numpy as np


class Flatten:
    """
    Flatten Layer

    Converts a multi-dimensional tensor into
    a 1D feature vector.

    Example
    -------
    Input:
        (16, 8, 8)

    Output:
        (1024,)
    """

    def __init__(self) -> None:

        # -------------------------
        # Cache
        # -------------------------
        self.input_cache = None
        self.output_cache = None

        # dùng cho backward
        self.original_shape = None

    # =====================================
    # Forward
    # =====================================

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Forward Pass

        Parameters
        ----------
        x : np.ndarray
            Shape:
                (C,H,W)
                hoặc bất kỳ shape nào

        Returns
        -------
        np.ndarray
            Flattened vector
        """

        self.input_cache = x
        self.original_shape = x.shape

        output = x.reshape(-1)

        self.output_cache = output

        return output

    # =====================================
    # Backward
    # =====================================

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:
        """
        Reshape gradient về shape ban đầu.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient từ layer phía sau

        Returns
        -------
        np.ndarray
            Gradient cùng shape với input
        """

        if self.original_shape is None:
            raise RuntimeError(
                "forward() must be called before backward()"
            )

        grad_input = grad_output.reshape(
            self.original_shape
        )

        return grad_input

    # =====================================
    # Utilities
    # =====================================

    @property
    def output_size(self) -> int:
        """
        Number of features after flatten.
        """

        if self.output_cache is None:
            return 0

        return self.output_cache.size

    def summary(self) -> None:
        """
        Print layer information.
        """

        print("\nFlatten")
        print("-" * 40)

        if self.original_shape is not None:

            print(
                f"Input Shape : {self.original_shape}"
            )

            print(
                f"Output Size : {self.output_size}"
            )

        else:

            print(
                "Layer has not been used yet."
            )

        print("-" * 40)

    def __repr__(self) -> str:

        return "Flatten()"