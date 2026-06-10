"""
Data schemas used across the MNIST project.

This module defines structured contracts for:

- Dataset metadata
- Dataset split information
- Validation reports
- Dataset release information

These schemas help standardize communication
between data processing, training, and evaluation
components.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Dataset Split Schema

@dataclass(frozen=True)
class DatasetSplitSchema:
    """
    Metadata for a dataset split.

    Example:
        train:
            X_train.npy
            y_train.npy
    """

    split_name: str
    features_path: Path
    labels_path: Path
    num_samples: int


# Dataset Release Schema

@dataclass(frozen=True)
class DatasetReleaseSchema:
    """
    Represents a complete released dataset.
    """

    train: DatasetSplitSchema
    valid: DatasetSplitSchema
    test: DatasetSplitSchema

    image_height: int
    image_width: int
    channels: int

    num_classes: int


# Dataset Statistics Schema

@dataclass(frozen=True)
class DatasetStatisticsSchema:
    """
    Statistical information about the dataset.
    """

    total_samples: int

    train_samples: int
    valid_samples: int
    test_samples: int

    num_classes: int

    pixel_min: float
    pixel_max: float

    pixel_mean: float
    pixel_std: float


# Validation Report Schema

@dataclass(frozen=True)
class ValidationReportSchema:
    """
    Output schema from validation.py.
    """

    passed: bool

    missing_values: int

    duplicate_rows: int

    invalid_labels: int

    total_records: int

    message: Optional[str] = None


# Dataset Metadata Schema

@dataclass(frozen=True)
class DatasetMetadataSchema:
    """
    High-level metadata stored with the dataset.
    """

    dataset_name: str

    version: str

    source: str

    created_at: str

    total_samples: int

    num_classes: int

    image_shape: tuple[int, int]

    description: str