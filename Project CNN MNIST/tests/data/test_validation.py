import numpy as np
import pytest

from src.data.validation import DatasetValidator

# pytest tests/data/test_validation.py -v

@pytest.fixture
def valid_dataset(tmp_path):
    """
    Create a valid MNIST-style dataset.
    """

    X = np.random.rand(
        100,
        28,
        28,
    ).astype(np.float32)

    y = np.random.randint(
        0,
        10,
        size=(100,),
        dtype=np.int64,
    )

    x_path = tmp_path / "X.npy"
    y_path = tmp_path / "y.npy"

    np.save(x_path, X)
    np.save(y_path, y)

    return x_path, y_path


@pytest.fixture
def validator(valid_dataset):
    """
    Create validator instance.
    """

    x_path, y_path = valid_dataset

    return DatasetValidator(
        x_path=x_path,
        y_path=y_path,
    )


# Load Dataset

def test_load_dataset(validator):

    X, y = validator._load()

    assert X.shape == (100, 28, 28)
    assert y.shape == (100,)


# Sample Count

def test_sample_count_match(validator):

    X, y = validator._load()

    mismatch = validator.check_sample_count(X, y)

    assert mismatch == 0


def test_sample_count_mismatch(validator):

    X, y = validator._load()

    mismatch = validator.check_sample_count(
        X[:-1],
        y,
    )

    assert mismatch == 1


# Missing Values

def test_missing_values_none(validator):

    X, _ = validator._load()

    missing = validator.check_missing_values(X)

    assert missing == 0


def test_missing_values_detected(validator):

    X, _ = validator._load()

    X[0, 0, 0] = np.nan

    missing = validator.check_missing_values(X)

    assert missing == 1


# Invalid Labels

def test_invalid_labels_none(validator):

    _, y = validator._load()

    invalid = validator.check_invalid_labels(y)

    assert invalid == 0


def test_invalid_labels_detected(validator):

    _, y = validator._load()

    y[0] = 99

    invalid = validator.check_invalid_labels(y)

    assert invalid == 1


# Duplicate Images

def test_duplicate_images_none(validator):

    X, _ = validator._load()

    duplicates = validator.check_duplicate_images(X)

    assert duplicates == 0


def test_duplicate_images_detected(validator):

    X, _ = validator._load()

    X[1] = X[0]

    duplicates = validator.check_duplicate_images(X)

    assert duplicates == 1


# Image Shape

def test_image_shape_valid(validator):

    X, _ = validator._load()

    assert validator.check_image_shape(X)


def test_image_shape_invalid():

    X = np.random.rand(
        100,
        32,
        32,
    ).astype(np.float32)

    validator = DatasetValidator(
        "dummy",
        "dummy",
    )

    assert validator.check_image_shape(X) is False


# Pixel Range

def test_pixel_range_valid(validator):

    X, _ = validator._load()

    assert validator.check_pixel_range(X)


def test_pixel_range_invalid(validator):

    X, _ = validator._load()

    X[0, 0, 0] = 2.0

    assert validator.check_pixel_range(X) is False


# Full Validation

def test_validation_pass(validator):

    report = validator.validate()

    assert report.passed is True

    assert report.total_records == 100

    assert report.missing_values == 0

    assert report.invalid_labels == 0

    assert report.duplicate_rows == 0


def test_validation_fail(tmp_path):

    X = np.random.rand(
        50,
        28,
        28,
    ).astype(np.float32)

    X[0, 0, 0] = np.nan

    y = np.random.randint(
        0,
        10,
        size=(50,),
    )

    y[0] = 99

    x_path = tmp_path / "X.npy"
    y_path = tmp_path / "y.npy"

    np.save(x_path, X)
    np.save(y_path, y)

    validator = DatasetValidator(
        x_path=x_path,
        y_path=y_path,
    )

    report = validator.validate()

    assert report.passed is False

    assert report.missing_values > 0

    assert report.invalid_labels > 0


# Summary

def test_summary(validator, capsys):

    validator.summary()

    captured = capsys.readouterr()

    assert "Dataset Validation Report" in captured.out
    assert "Passed" in captured.out
    assert "Total Records" in captured.out