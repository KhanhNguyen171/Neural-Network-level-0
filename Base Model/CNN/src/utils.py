"""
utils.py

Utility Functions
for CNN From Scratch

Author: CNN From Scratch
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


# Image Loading

def load_image(
    image_path: str,
    grayscale: bool = False
) -> np.ndarray:
    """
    Load image from disk.

    Parameters
    ----------
    image_path : str

    grayscale : bool

    Returns
    -------
    np.ndarray

    RGB:
        (C,H,W)

    Gray:
        (1,H,W)
    """

    image = Image.open(image_path)

    if grayscale:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if grayscale:

        image = image[np.newaxis, :, :]

    else:

        image = np.transpose(
            image,
            (2, 0, 1)
        )

    return image


# Normalization

def normalize_image(
    image: np.ndarray
) -> np.ndarray:
    """
    Scale image to [0,1]
    """

    return image.astype(
        np.float32
    ) / 255.0


# Denormalization

def denormalize_image(
    image: np.ndarray
) -> np.ndarray:
    """
    Scale image back to [0,255]
    """

    image = image * 255.0

    image = np.clip(
        image,
        0,
        255
    )

    return image.astype(
        np.uint8
    )


# Image Info

def image_info(
    image: np.ndarray
) -> None:
    """
    Print image information.
    """

    print("\nImage Information")
    print("-" * 40)

    print(
        f"Shape : {image.shape}"
    )

    print(
        f"Dtype : {image.dtype}"
    )

    print(
        f"Min   : {image.min()}"
    )

    print(
        f"Max   : {image.max()}"
    )

    print("-" * 40)


# Visualization

def show_image(
    image: np.ndarray,
    title: str = "Image"
) -> None:
    """
    Display image.
    """

    plt.figure(figsize=(5, 5))

    if image.ndim == 3:

        if image.shape[0] == 1:

            plt.imshow(
                image[0],
                cmap="gray"
            )

        else:

            img = np.transpose(
                image,
                (1, 2, 0)
            )

            plt.imshow(img)

    elif image.ndim == 2:

        plt.imshow(
            image,
            cmap="gray"
        )

    else:

        raise ValueError(
            "Unsupported image shape."
        )

    plt.title(title)
    plt.axis("off")
    plt.show()


# Feature Map Visualization

def plot_feature_maps(
    feature_maps: np.ndarray,
    max_maps: int = 16
) -> None:
    """
    Plot CNN feature maps.

    Input:
        (F,H,W)
    """

    num_maps = min(
        feature_maps.shape[0],
        max_maps
    )

    cols = 4

    rows = int(
        np.ceil(num_maps / cols)
    )

    plt.figure(
        figsize=(12, 3 * rows)
    )

    for i in range(num_maps):

        plt.subplot(
            rows,
            cols,
            i + 1
        )

        plt.imshow(
            feature_maps[i],
            cmap="gray"
        )

        plt.title(
            f"Map {i}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Filter Visualization

def plot_filters(
    filters: np.ndarray
) -> None:
    """
    Plot convolution filters.

    Shape:
        (F,C,K,K)
    """

    num_filters = filters.shape[0]

    cols = 4

    rows = int(
        np.ceil(
            num_filters / cols
        )
    )

    plt.figure(
        figsize=(12, 3 * rows)
    )

    for i in range(num_filters):

        plt.subplot(
            rows,
            cols,
            i + 1
        )

        kernel = filters[i]

        kernel = np.mean(
            kernel,
            axis=0
        )

        plt.imshow(
            kernel,
            cmap="gray"
        )

        plt.title(
            f"Filter {i}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Histogram

def plot_histogram(
    data: np.ndarray,
    bins: int = 50,
    title: str = "Histogram"
) -> None:
    """
    Visualize value distribution.
    """

    plt.figure(figsize=(6, 4))

    plt.hist(
        data.flatten(),
        bins=bins
    )

    plt.title(title)

    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.show()


# Save Figure

def save_figure(
    filename: str,
    dpi: int = 300
) -> None:
    """
    Save current matplotlib figure.
    """

    Path(
        filename
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        filename,
        dpi=dpi,
        bbox_inches="tight"
    )


# Parameter Counter

def count_parameters(
    layer
) -> int:
    """
    Count trainable parameters.
    """

    total = 0

    if hasattr(layer, "weights"):
        total += layer.weights.size

    if hasattr(layer, "bias"):
        total += layer.bias.size

    if hasattr(layer, "filters"):
        total += layer.filters.size

    return total


# One Hot Encoding

def one_hot(
    label: int,
    num_classes: int
) -> np.ndarray:
    """
    One-hot encoding.
    """

    vector = np.zeros(
        num_classes
    )

    vector[label] = 1.0

    return vector


# Accuracy

def accuracy(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Classification accuracy.
    """

    predictions = np.asarray(
        predictions
    )

    targets = np.asarray(
        targets
    )

    return np.mean(
        predictions == targets
    )