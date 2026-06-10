from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import MNISTDataset


def create_dataloader(
    data_dir: str | Path,
    split: str,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create DataLoader for a dataset split.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing processed .npy files.

    split : str
        train | valid | test

    batch_size : int
        Batch size.

    shuffle : bool
        Shuffle dataset.

    num_workers : int
        Number of worker processes.

    pin_memory : bool
        Pin memory for GPU training.

    Returns
    -------
    DataLoader
    """

    dataset = MNISTDataset(
        data_dir=data_dir,
        split=split
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def create_train_dataloader(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:

    return create_dataloader(
        data_dir=data_dir,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


def create_valid_dataloader(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:

    return create_dataloader(
        data_dir=data_dir,
        split="valid",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


def create_test_dataloader(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:

    return create_dataloader(
        data_dir=data_dir,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )