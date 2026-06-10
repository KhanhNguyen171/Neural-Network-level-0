import numpy as np
import pytest
import torch

from src.data.dataloader import (
    create_dataloader,
    create_train_dataloader,
    create_valid_dataloader,
    create_test_dataloader
)

# pytest tests/data/test_dataloader.py -v

@pytest.fixture
def sample_dataset(tmp_path):

    num_samples = 128

    X = np.random.rand(
        num_samples,
        28,
        28
    ).astype(np.float32)

    y = np.random.randint(
        0,
        10,
        size=num_samples
    )

    for split in [
        "train",
        "valid",
        "test"
    ]:

        np.save(
            tmp_path / f"X_{split}.npy",
            X
        )

        np.save(
            tmp_path / f"y_{split}.npy",
            y
        )

    return tmp_path

def test_create_dataloader(
    sample_dataset
):

    loader = create_dataloader(
        data_dir=sample_dataset,
        split="train",
        batch_size=32
    )

    assert loader is not None
    
def test_dataset_size(
    sample_dataset
):

    loader = create_dataloader(
        data_dir=sample_dataset,
        split="train",
        batch_size=32
    )

    assert len(
        loader.dataset
    ) == 128
    
def test_batch_size(
    sample_dataset
):

    loader = create_dataloader(
        data_dir=sample_dataset,
        split="train",
        batch_size=32
    )

    images, labels = next(
        iter(loader)
    )

    assert images.shape[0] == 32
    assert labels.shape[0] == 32
    
def test_image_shape(
    sample_dataset
):

    loader = create_dataloader(
        data_dir=sample_dataset,
        split="train",
        batch_size=16
    )

    images, _ = next(
        iter(loader)
    )

    assert images.shape == (
        16,
        1,
        28,
        28
    )
    
def test_label_shape(
    sample_dataset
):

    loader = create_dataloader(
        data_dir=sample_dataset,
        split="train",
        batch_size=16
    )

    _, labels = next(
        iter(loader)
    )

    assert labels.shape == (
        16,
    )
    
def test_tensor_type(
    sample_dataset
):

    loader = create_dataloader(
        data_dir=sample_dataset,
        split="train",
        batch_size=16
    )

    images, labels = next(
        iter(loader)
    )

    assert isinstance(
        images,
        torch.Tensor
    )

    assert isinstance(
        labels,
        torch.Tensor
    )
    
def test_train_loader(
    sample_dataset
):

    loader = create_train_dataloader(
        data_dir=sample_dataset,
        batch_size=64
    )

    assert loader.batch_size == 64
    
def test_valid_loader(
    sample_dataset
):

    loader = create_valid_dataloader(
        data_dir=sample_dataset,
        batch_size=64
    )

    assert loader.batch_size == 64
    
def test_test_loader(
    sample_dataset
):

    loader = create_test_dataloader(
        data_dir=sample_dataset,
        batch_size=64
    )

    assert loader.batch_size == 64
    
def test_multiple_batches(
    sample_dataset
):

    loader = create_dataloader(
        data_dir=sample_dataset,
        split="train",
        batch_size=32
    )

    total_samples = 0

    for images, labels in loader:

        total_samples += len(images)

    assert total_samples == 128