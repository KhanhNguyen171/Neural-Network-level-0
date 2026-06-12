# tests/evaluation/test_confusion_matrix.py

import pytest
import torch

from src.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    normalize_confusion_matrix,
    per_class_accuracy,
)

# pytest tests/evaluation/test_confusion_matrix.py -v

# compute_confusion_matrix

def test_confusion_matrix_shape():
    y_true = torch.tensor(
        [0, 1, 2]
    )

    y_pred = torch.tensor(
        [0, 1, 2]
    )

    cm = compute_confusion_matrix(
        y_true,
        y_pred,
        num_classes=3,
    )

    assert cm.shape == (3, 3)


def test_confusion_matrix_perfect_prediction():
    y_true = torch.tensor(
        [0, 1, 2]
    )

    y_pred = torch.tensor(
        [0, 1, 2]
    )

    cm = compute_confusion_matrix(
        y_true,
        y_pred,
        num_classes=3,
    )

    expected = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    assert torch.equal(
        cm,
        expected,
    )


def test_confusion_matrix_values():
    y_true = torch.tensor(
        [0, 0, 1, 1]
    )

    y_pred = torch.tensor(
        [0, 1, 1, 1]
    )

    cm = compute_confusion_matrix(
        y_true,
        y_pred,
        num_classes=2,
    )

    expected = torch.tensor(
        [
            [1, 1],
            [0, 2],
        ]
    )

    assert torch.equal(
        cm,
        expected,
    )


def test_confusion_matrix_dtype():
    y_true = torch.tensor(
        [0, 1]
    )

    y_pred = torch.tensor(
        [0, 1]
    )

    cm = compute_confusion_matrix(
        y_true,
        y_pred,
        2,
    )

    assert cm.dtype == torch.int64


def test_confusion_matrix_empty_input():
    with pytest.raises(
        ValueError
    ):
        compute_confusion_matrix(
            torch.tensor([]),
            torch.tensor([]),
            2,
        )


def test_confusion_matrix_shape_mismatch():
    with pytest.raises(
        ValueError
    ):
        compute_confusion_matrix(
            torch.tensor([0, 1]),
            torch.tensor([0]),
            2,
        )


def test_confusion_matrix_invalid_num_classes():
    with pytest.raises(
        ValueError
    ):
        compute_confusion_matrix(
            torch.tensor([0]),
            torch.tensor([0]),
            0,
        )


# normalize_confusion_matrix

def test_normalized_shape():
    cm = torch.tensor(
        [
            [4, 1],
            [2, 3],
        ]
    )

    normalized = (
        normalize_confusion_matrix(cm)
    )

    assert normalized.shape == (
        2,
        2,
    )


def test_normalized_rows_sum_to_one():
    cm = torch.tensor(
        [
            [4, 1],
            [2, 3],
        ]
    )

    normalized = (
        normalize_confusion_matrix(cm)
    )

    row_sums = normalized.sum(
        dim=1
    )

    expected = torch.ones(2)

    assert torch.allclose(
        row_sums,
        expected,
    )


def test_normalized_dtype():
    cm = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ]
    )

    normalized = (
        normalize_confusion_matrix(cm)
    )

    assert (
        normalized.dtype
        == torch.float32
    )


def test_normalized_zero_row():
    cm = torch.tensor(
        [
            [0, 0],
            [1, 1],
        ]
    )

    normalized = (
        normalize_confusion_matrix(cm)
    )

    assert torch.all(
        torch.isfinite(normalized)
    )


def test_normalized_invalid_dimension():
    with pytest.raises(
        ValueError
    ):
        normalize_confusion_matrix(
            torch.tensor(
                [1, 2, 3]
            )
        )


# per_class_accuracy

def test_per_class_accuracy_shape():
    cm = torch.tensor(
        [
            [5, 0],
            [1, 4],
        ]
    )

    acc = per_class_accuracy(cm)

    assert acc.shape == (2,)


def test_per_class_accuracy_values():
    cm = torch.tensor(
        [
            [5, 0],
            [1, 4],
        ]
    )

    acc = per_class_accuracy(cm)

    expected = torch.tensor(
        [
            1.0,
            0.8,
        ]
    )

    assert torch.allclose(
        acc,
        expected,
    )


def test_per_class_accuracy_perfect():
    cm = torch.tensor(
        [
            [3, 0],
            [0, 5],
        ]
    )

    acc = per_class_accuracy(cm)

    expected = torch.tensor(
        [1.0, 1.0]
    )

    assert torch.allclose(
        acc,
        expected,
    )


def test_per_class_accuracy_zero_row():
    cm = torch.tensor(
        [
            [0, 0],
            [1, 2],
        ]
    )

    acc = per_class_accuracy(cm)

    assert torch.isfinite(
        acc
    ).all()


def test_per_class_accuracy_invalid_dimension():
    with pytest.raises(
        ValueError
    ):
        per_class_accuracy(
            torch.tensor(
                [1, 2, 3]
            )
        )


# MNIST-like example

def test_mnist_10_classes():
    y_true = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    y_pred = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    cm = compute_confusion_matrix(
        y_true,
        y_pred,
        num_classes=10,
    )

    assert cm.shape == (
        10,
        10,
    )

    assert torch.trace(cm) == 10
