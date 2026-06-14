from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_figure(
    filepath,
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    """
    Save current matplotlib figure.
    """
    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        filepath,
        dpi=dpi,
        bbox_inches=bbox_inches,
    )


def plot_loss_curve(
    train_loss,
    val_loss=None,
    title="Loss Curve",
):
    """
    Plot training/validation loss.
    """
    fig, ax = plt.subplots()

    ax.plot(
        train_loss,
        label="train_loss",
    )

    if val_loss is not None:
        ax.plot(
            val_loss,
            label="val_loss",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    return fig, ax


def plot_accuracy_curve(
    train_acc,
    val_acc=None,
    title="Accuracy Curve",
):
    """
    Plot training/validation accuracy.
    """
    fig, ax = plt.subplots()

    ax.plot(
        train_acc,
        label="train_accuracy",
    )

    if val_acc is not None:
        ax.plot(
            val_acc,
            label="val_accuracy",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()

    return fig, ax


def plot_confusion_matrix(
    cm,
    class_names=None,
    normalize=False,
    title="Confusion Matrix",
):
    """
    Plot confusion matrix.
    """
    cm = np.asarray(cm)

    if normalize:
        row_sum = cm.sum(
            axis=1,
            keepdims=True,
        )

        row_sum[row_sum == 0] = 1

        cm = cm / row_sum

    fig, ax = plt.subplots()

    image = ax.imshow(cm)

    plt.colorbar(image)

    if class_names is not None:
        ax.set_xticks(
            np.arange(len(class_names))
        )

        ax.set_yticks(
            np.arange(len(class_names))
        )

        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    return fig, ax


def show_images(
    images,
    labels=None,
    max_images=16,
):
    """
    Show image grid.
    """
    images = images[:max_images]

    n = len(images)

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        squeeze=False,
    )

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]

        if idx >= n:
            ax.axis("off")
            continue

        image = images[idx]

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        image = np.squeeze(image)

        ax.imshow(
            image,
            cmap="gray",
        )

        if labels is not None:
            ax.set_title(
                str(labels[idx])
            )

        ax.axis("off")

    return fig, axes


def close_figure(
    fig=None,
):
    """
    Close matplotlib figure.
    """
    if fig is None:
        plt.close()

    else:
        plt.close(fig)