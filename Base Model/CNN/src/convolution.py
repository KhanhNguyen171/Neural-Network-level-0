import numpy as np


class Convolution2D:
    """
    Educational CNN Convolution Layer

    Input Shape:
        (C, H, W)

    Output Shape:
        (F, H_out, W_out)

    Parameters:
        in_channels
        out_channels
        kernel_size
        stride
        padding
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(
            2.0 /
            (in_channels * kernel_size * kernel_size)
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

        # caches

        self.input_cache = None
        self.output_cache = None

        # gradients
        # sẽ dùng ở notebook 15

        self.d_filters = np.zeros_like(
            self.filters
        )

        self.db = np.zeros_like(
            self.bias
        )

    def forward(self, x):
        """
        x shape:
            (C, H, W)
        """

        self.input_cache = x

        channels, height, width = x.shape

        k = self.kernel_size
        s = self.stride
        p = self.padding

        if p > 0:

            x_padded = np.pad(
                x,
                (
                    (0, 0),
                    (p, p),
                    (p, p)
                ),
                mode="constant"
            )

        else:

            x_padded = x

        out_height = (
            (height - k + 2 * p)
            // s
        ) + 1

        out_width = (
            (width - k + 2 * p)
            // s
        ) + 1

        output = np.zeros(
            (
                self.out_channels,
                out_height,
                out_width
            )
        )

        for f in range(
            self.out_channels
        ):

            kernel = self.filters[f]

            for i in range(
                out_height
            ):

                for j in range(
                    out_width
                ):

                    h_start = i * s
                    h_end = h_start + k

                    w_start = j * s
                    w_end = w_start + k

                    region = x_padded[
                        :,
                        h_start:h_end,
                        w_start:w_end
                    ]

                    output[
                        f,
                        i,
                        j
                    ] = (
                        np.sum(
                            region * kernel
                        )
                        + self.bias[f]
                    )

        self.output_cache = output

        return output

    def backward(
        self,
        grad_output
    ):
        """
        Notebook 15:
            Conv Backward

        Will compute:

            d_filters
            db
            d_input
        """

        raise NotImplementedError(
            "Implement in notebook 15."
        )

    def zero_grad(self):

        self.d_filters.fill(0)

        self.db.fill(0)

    def parameters(self):

        return [
            (
                self.filters,
                self.d_filters
            ),
            (
                self.bias,
                self.db
            )
        ]

    def summary(self):

        print(
            "Convolution2D Layer"
        )

        print(
            f"Input Channels : {self.in_channels}"
        )

        print(
            f"Output Channels: {self.out_channels}"
        )

        print(
            f"Kernel Size    : {self.kernel_size}"
        )

        print(
            f"Stride         : {self.stride}"
        )

        print(
            f"Padding        : {self.padding}"
        )

        print(
            f"Filters Shape  : {self.filters.shape}"
        )

    def __repr__(self):

        return (
            f"Convolution2D("
            f"{self.in_channels}, "
            f"{self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}"
            f")"
        )