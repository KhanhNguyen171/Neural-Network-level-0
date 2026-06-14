import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from src.utils.visualization import (
    save_figure,
    plot_loss_curve,
    plot_accuracy_curve,
    plot_confusion_matrix,
    show_images,
    close_figure,
)

# pytest tests/utils/test_visualization.py -v

def test_plot_loss_curve():
    fig, ax = plot_loss_curve(
        [1.0, 0.8, 0.5],
        [1.1, 0.9, 0.6],
    )

    assert fig is not None
    assert ax is not None

    close_figure(fig)


def test_plot_accuracy_curve():
    fig, ax = plot_accuracy_curve(
        [0.5, 0.7, 0.9],
        [0.4, 0.6, 0.8],
    )

    assert fig is not None
    assert ax is not None

    close_figure(fig)


def test_plot_confusion_matrix():
    cm = np.array(
        [
            [8, 2],
            [1, 9],
        ]
    )

    fig, ax = plot_confusion_matrix(
        cm
    )

    assert fig is not None
    assert ax is not None

    close_figure(fig)


def test_plot_confusion_matrix_normalized():
    cm = np.array(
        [
            [8, 2],
            [1, 9],
        ]
    )

    fig, ax = plot_confusion_matrix(
        cm,
        normalize=True,
    )

    assert fig is not None

    close_figure(fig)


def test_show_images_numpy():
    images = np.random.rand(
        8,
        28,
        28,
    )

    fig, axes = show_images(
        images
    )

    assert fig is not None
    assert axes is not None

    close_figure(fig)


def test_show_images_tensor():
    images = torch.rand(
        8,
        1,
        28,
        28,
    )

    fig, axes = show_images(
        images
    )

    assert fig is not None

    close_figure(fig)


def test_show_images_with_labels():
    images = torch.rand(
        4,
        1,
        28,
        28,
    )

    labels = [0, 1, 2, 3]

    fig, axes = show_images(
        images,
        labels,
    )

    assert fig is not None

    close_figure(fig)


def test_save_figure():
    fig, _ = plot_loss_curve(
        [1, 2, 3]
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = (
            Path(tmp)
            / "loss.png"
        )

        save_figure(path)

        assert path.exists()

    close_figure(fig)


def test_close_figure():
    fig, _ = plot_loss_curve(
        [1, 2]
    )

    close_figure(fig)