"""
convolution.py

CNN From Scratch

Phase A:
    - Single Channel Convolution
    - Multi Channel Convolution
    - Multi Filter Convolution
    - Padding
    - Stride

Phase B:
    - Backpropagation
    - Gradient Calculation

Author: CNN From Scratch
"""

from __future__ import annotations

import numpy as np


class Convolution2D:
    """
    2D Convolution Layer

    Input Shape:
        (C, H, W)

    Filters Shape:
        (F, C, K, K)

    Output Shape:
        (F, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        random_seed: int | None = None
    ) -> None:

        if random_seed is not None:
            np.random.seed(random_seed)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        # He Initialization
        scale = np.sqrt(
            2.0 / (in_channels * kernel_size * kernel_size)
        )

        self.filters = (
            np.random.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            )
            * scale
        )

        self.bias = np.zeros(out_channels)

        # Cache (Phase B)
        self.input_cache = None
        self.output_cache = None

        # Gradients (Phase B)
        self.d_filters = None
        self.d_bias = None

    # Utility Functions

    def compute_output_shape(
        self,
        height: int,
        width: int
    ) -> tuple[int, int]:
        """
        Compute output height and width.

        Formula:
            Hout = floor((H-K+2P)/S)+1
            Wout = floor((W-K+2P)/S)+1
        """

        h_out = (
            (height - self.kernel_size + 2 * self.padding)
        ) // self.stride + 1

        w_out = (
            (width - self.kernel_size + 2 * self.padding) 
        ) // self.stride + 1

        return h_out, w_out

    def apply_padding(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Apply zero padding.

        Input:
            (C,H,W)

        Output:
            (C,H+2P,W+2P)
        """

        if self.padding == 0:
            return x

        return np.pad(
            x,
            pad_width=(
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding)
            ),
            mode="constant"
        )

    # Single Channel Convolution

    @staticmethod
    def convolve_single_channel(
        image: np.ndarray,
        kernel: np.ndarray,
        stride: int = 1
    ) -> np.ndarray:
        """
        Convolution on one channel.

        image:
            (H,W)

        kernel:
            (K,K)
        """

        h, w = image.shape
        k = kernel.shape[0]

        h_out = ((h - k) // stride) + 1
        w_out = ((w - k) // stride) + 1

        output = np.zeros((h_out, w_out))

        for row in range(h_out):
            for col in range(w_out):

                r = row * stride
                c = col * stride

                region = image[
                    r:r + k,
                    c:c + k
                ]

                output[row, col] = np.sum(
                    region * kernel
                )

        return output

    # Multi Channel Convolution

    def convolve_multi_channel(
        self,
        x: np.ndarray,
        kernel: np.ndarray
    ) -> np.ndarray:
        """
        Multi-channel convolution.

        x:
            (C,H,W)

        kernel:
            (C,K,K)

        Output:
            (Hout,Wout)
        """

        channels = x.shape[0]

        result = None

        for c in range(channels):

            conv = self.convolve_single_channel(
                x[c],
                kernel[c],
                self.stride
            )

            if result is None:
                result = conv
            else:
                result += conv

        return result

    # Forward

    def forward(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Forward Propagation

        Input:
            (C,H,W)

        Output:
            (F,Hout,Wout)
        """

        if x.ndim != 3:
            raise ValueError(
                "Input must have shape (C,H,W)"
            )

        if x.shape[0] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, "
                f"received {x.shape[0]}"
            )

        self.input_cache = x

        x_padded = self.apply_padding(x)

        _, height, width = x.shape

        h_out, w_out = self.compute_output_shape(
            height,
            width
        )

        output = np.zeros(
            (
                self.out_channels,
                h_out,
                w_out
            )
        )

        for f in range(self.out_channels):

            output[f] = (
                self.convolve_multi_channel(
                    x_padded,
                    self.filters[f]
                )
                + self.bias[f]
            )

        self.output_cache = output

        return output

    # Information

    @property
    def num_parameters(self) -> int:
        """
        Total trainable parameters.
        """

        return (
            self.filters.size
            + self.bias.size
        )

    def summary(self) -> None:
        """
        Print layer information.
        """

        print("\nConvolution2D")
        print("-" * 40)

        print(
            f"In Channels : {self.in_channels}"
        )
        print(
            f"Out Channels: {self.out_channels}"
        )
        print(
            f"Kernel Size : {self.kernel_size}"
        )
        print(
            f"Stride      : {self.stride}"
        )
        print(
            f"Padding     : {self.padding}"
        )

        print(
            f"Parameters  : {self.num_parameters}"
        )

        print("-" * 40)

    # Phase B

    def backward(
        self,
        grad_output: np.ndarray
    ) -> np.ndarray:
        """
        Backpropagation

        Notebook:
            15_conv_backward.ipynb

        To implement later:
            d_filters
            d_bias
            d_input
        """

        raise NotImplementedError(
            "Implemented in Phase B"
        )