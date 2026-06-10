from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """
    PyTorch Dataset for processed MNIST data.

    Expected files:

    X_train.npy
    y_train.npy

    X_valid.npy
    y_valid.npy

    X_test.npy
    y_test.npy
    """

    VALID_SPLITS = {
        "train",
        "valid",
        "test"
    }

    def __init__(
        self,
        data_dir: str | Path,
        split: str
    ) -> None:

        self.data_dir = Path(data_dir)

        split = split.lower()

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self.VALID_SPLITS}"
            )

        self.split = split

        self.images = np.load(
            self.data_dir / f"X_{split}.npy"
        )

        self.labels = np.load(
            self.data_dir / f"y_{split}.npy"
        )

        if len(self.images) != len(self.labels):
            raise ValueError(
                "Images and labels have different lengths"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self,
        index: int
    ):

        image = self.images[index]
        label = self.labels[index]

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        if image.ndim == 2:
            image = image.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label

    @property
    def num_samples(self) -> int:
        return len(self)

    @property
    def image_shape(self) -> tuple:
        return self.images.shape[1:]

    @property
    def num_classes(self) -> int:
        return len(np.unique(self.labels))

    def summary(self) -> dict:

        return {
            "split": self.split,
            "num_samples": self.num_samples,
            "image_shape": self.image_shape,
            "num_classes": self.num_classes
        }