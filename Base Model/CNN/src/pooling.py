"""
pooling.py

CNN From Scratch

Pooling Layers:
    - MaxPooling2D
    - AveragePooling2D

Author: CNN From Scratch
"""

from __future__ import annotations

import numpy as np


class MaxPooling2D:
    """
    Max Pooling Layer

    Input:
        (C,H,W)

    Output:
        (C,Hout,Wout)
    """

    def __init__(
        self,
        pool_size: int = 2,
        stride: int | None = None
    ) -> None:

        self.pool_size = pool_size
        self.stride = (
            stride if stride is not None
            else pool_size
        )

        self.input_cache = None
        self.output_cache = None

        # dùng cho backward
        self.argmax_mask = None

    # Utilities

    def compute_output_shape(
        self,
        height: int,
        width: int
    ) -> tuple[int, int]:

        h_out = (
            (height - self.pool_size)
            // self.stride
        ) + 1

        w_out = (
            (width - self.pool_size)
            // self.stride
        ) + 1

        return h_out, w_out

    # Forward

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        if x.ndim != 3:
            raise ValueError(
                "Input must have shape (C,H,W)"
            )

        self.input_cache = x

        channels, height, width = x.shape

        h_out, w_out = self.compute_output_shape(
            height,
            width
        )

        output = np.zeros(
            (
                channels,
                h_out,
                w_out
            )
        )

        self.argmax_mask = np.zeros_like(x)

        for c in range(channels):

            for row in range(h_out):

                for col in range(w_out):

                    r = row * self.stride
                    c0 = col * self.stride

                    window = x[
                        c,
                        r:r+self.pool_size,
                        c0:c0+self.pool_size
                    ]

                    max_value = np.max(window)

                    output[c, row, col] = max_value

                    # lưu vị trí max cho backward
                    max_idx = np.unravel_index(
                        np.argmax(window),
                        window.shape
                    )

                    self.argmax_mask[
                        c,
                        r + max_idx[0],
                        c0 + max_idx[1]
                    ] = 1

        self.output_cache = output

        return output

    # Backward

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:
        """
        Gradient chỉ truyền
        tới phần tử lớn nhất.
        """

        channels, h_out, w_out = grad_output.shape

        grad_input = np.zeros_like(
            self.input_cache
        )

        for c in range(channels):

            for row in range(h_out):

                for col in range(w_out):

                    r = row * self.stride
                    c0 = col * self.stride

                    window_mask = self.argmax_mask[
                        c,
                        r:r+self.pool_size,
                        c0:c0+self.pool_size
                    ]

                    grad_input[
                        c,
                        r:r+self.pool_size,
                        c0:c0+self.pool_size
                    ] += (
                        window_mask
                        * grad_output[c, row, col]
                    )

        return grad_input

    # Summary

    def summary(self) -> None:

        print("\nMaxPooling2D")
        print("-" * 40)

        print(
            f"Pool Size : {self.pool_size}"
        )

        print(
            f"Stride    : {self.stride}"
        )

        print("-" * 40)


class AveragePooling2D:
    """
    Average Pooling Layer
    """

    def __init__(
        self,
        pool_size: int = 2,
        stride: int | None = None
    ) -> None:

        self.pool_size = pool_size

        self.stride = (
            stride if stride is not None
            else pool_size
        )

        self.input_cache = None
        self.output_cache = None

    # Utilities

    def compute_output_shape(
        self,
        height: int,
        width: int
    ) -> tuple[int, int]:

        h_out = (
            (height - self.pool_size)
            // self.stride
        ) + 1

        w_out = (
            (width - self.pool_size)
            // self.stride
        ) + 1

        return h_out, w_out

    # Forward

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        self.input_cache = x

        channels, height, width = x.shape

        h_out, w_out = self.compute_output_shape(
            height,
            width
        )

        output = np.zeros(
            (
                channels,
                h_out,
                w_out
            )
        )

        for c in range(channels):

            for row in range(h_out):

                for col in range(w_out):

                    r = row * self.stride
                    c0 = col * self.stride

                    window = x[
                        c,
                        r:r+self.pool_size,
                        c0:c0+self.pool_size
                    ]

                    output[c, row, col] = (
                        np.mean(window)
                    )

        self.output_cache = output

        return output

    # Backward

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:

        grad_input = np.zeros_like(
            self.input_cache
        )

        channels, h_out, w_out = grad_output.shape

        area = (
            self.pool_size
            * self.pool_size
        )

        for c in range(channels):

            for row in range(h_out):

                for col in range(w_out):

                    r = row * self.stride
                    c0 = col * self.stride

                    grad = (
                        grad_output[c, row, col]
                        / area
                    )

                    grad_input[
                        c,
                        r:r+self.pool_size,
                        c0:c0+self.pool_size
                    ] += grad

        return grad_input

    # Summary

    def summary(self) -> None:

        print("\nAveragePooling2D")
        print("-" * 40)

        print(
            f"Pool Size : {self.pool_size}"
        )

        print(
            f"Stride    : {self.stride}"
        )

        print("-" * 40)