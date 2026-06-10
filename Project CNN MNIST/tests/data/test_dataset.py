from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import MNISTDataset

# pytest tests/data/test_dataset.py -v

@pytest.fixture
def sample_dataset(tmp_path):

    num_samples = 100

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

    np.save(
        tmp_path / "X_train.npy",
        X
    )

    np.save(
        tmp_path / "y_train.npy",
        y
    )

    return tmp_path

def test_dataset_load(
    sample_dataset
):

    dataset = MNISTDataset(
        data_dir=sample_dataset,
        split="train"
    )

    assert dataset is not None
    
def test_dataset_length(
    sample_dataset
):

    dataset = MNISTDataset(
        data_dir=sample_dataset,
        split="train"
    )

    assert len(dataset) == 100
    
def test_get_item(
    sample_dataset
):

    dataset = MNISTDataset(
        data_dir=sample_dataset,
        split="train"
    )

    image, label = dataset[0]

    assert isinstance(
        image,
        torch.Tensor
    )

    assert isinstance(
        label,
        torch.Tensor
    )
    
def test_image_shape(
    sample_dataset
):

    dataset = MNISTDataset(
        data_dir=sample_dataset,
        split="train"
    )

    image, _ = dataset[0]

    assert image.shape == (
        1,
        28,
        28
    )
    
def test_label_dtype(
    sample_dataset
):

    dataset = MNISTDataset(
        data_dir=sample_dataset,
        split="train"
    )

    _, label = dataset[0]

    assert label.dtype == torch.long
    
def test_invalid_split(
    sample_dataset
):

    with pytest.raises(
        ValueError
    ):

        MNISTDataset(
            data_dir=sample_dataset,
            split="unknown"
        )
        
def test_dataset_properties(
    sample_dataset
):

    dataset = MNISTDataset(
        data_dir=sample_dataset,
        split="train"
    )

    assert dataset.num_samples == 100

    assert dataset.image_shape == (
        28,
        28
    )

    assert dataset.num_classes > 0
    
def test_summary(
    sample_dataset
):

    dataset = MNISTDataset(
        data_dir=sample_dataset,
        split="train"
    )

    summary = dataset.summary()

    assert isinstance(
        summary,
        dict
    )

    assert summary["split"] == "train"

    assert summary["num_samples"] == 100