# tests/evaluation/test_metrics.py

import pytest
import torch

from src.evaluation.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    macro_precision,
    macro_recall,
    macro_f1,
    topk_accuracy,
    compute_metrics,
)

# pytest tests/evaluation/test_metrics.py -v


def test_accuracy_perfect():
    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.tensor([0, 1, 2])

    assert accuracy(
        y_true,
        y_pred,
    ) == pytest.approx(1.0)


def test_accuracy_partial():
    y_true = torch.tensor([0, 1, 2, 3])
    y_pred = torch.tensor([0, 1, 0, 3])

    assert accuracy(
        y_true,
        y_pred,
    ) == pytest.approx(0.75)


def test_accuracy_empty():
    with pytest.raises(ValueError):
        accuracy(
            torch.tensor([]),
            torch.tensor([]),
        )


def test_precision():
    y_true = torch.tensor(
        [0, 1, 1, 0]
    )

    y_pred = torch.tensor(
        [0, 1, 0, 0]
    )

    assert precision(
        y_true,
        y_pred,
        1,
    ) == pytest.approx(1.0)


def test_recall():
    y_true = torch.tensor(
        [0, 1, 1, 0]
    )

    y_pred = torch.tensor(
        [0, 1, 0, 0]
    )

    assert recall(
        y_true,
        y_pred,
        1,
    ) == pytest.approx(0.5)


def test_f1_score():
    y_true = torch.tensor(
        [0, 1, 1, 0]
    )

    y_pred = torch.tensor(
        [0, 1, 0, 0]
    )

    score = f1_score(
        y_true,
        y_pred,
        1,
    )

    assert score == pytest.approx(
        2 / 3
    )


def test_precision_zero_division():
    y_true = torch.tensor(
        [0, 0, 0]
    )

    y_pred = torch.tensor(
        [0, 0, 0]
    )

    assert (
        precision(
            y_true,
            y_pred,
            1,
        )
        == 0.0
    )


def test_recall_zero_division():
    y_true = torch.tensor(
        [0, 0, 0]
    )

    y_pred = torch.tensor(
        [0, 0, 0]
    )

    assert (
        recall(
            y_true,
            y_pred,
            1,
        )
        == 0.0
    )


def test_f1_zero_division():
    y_true = torch.tensor(
        [0, 0, 0]
    )

    y_pred = torch.tensor(
        [0, 0, 0]
    )

    assert (
        f1_score(
            y_true,
            y_pred,
            1,
        )
        == 0.0
    )


def test_macro_precision():
    y_true = torch.tensor(
        [0, 1, 2]
    )

    y_pred = torch.tensor(
        [0, 1, 2]
    )

    assert macro_precision(
        y_true,
        y_pred,
    ) == pytest.approx(1.0)


def test_macro_recall():
    y_true = torch.tensor(
        [0, 1, 2]
    )

    y_pred = torch.tensor(
        [0, 1, 2]
    )

    assert macro_recall(
        y_true,
        y_pred,
    ) == pytest.approx(1.0)


def test_macro_f1():
    y_true = torch.tensor(
        [0, 1, 2]
    )

    y_pred = torch.tensor(
        [0, 1, 2]
    )

    assert macro_f1(
        y_true,
        y_pred,
    ) == pytest.approx(1.0)


def test_topk_accuracy_top1():
    logits = torch.tensor(
        [
            [10.0, 1.0],
            [1.0, 10.0],
        ]
    )

    targets = torch.tensor(
        [0, 1]
    )

    assert topk_accuracy(
        logits,
        targets,
        k=1,
    ) == pytest.approx(1.0)


def test_topk_accuracy_top2():
    logits = torch.tensor(
        [
            [10.0, 9.0, 1.0],
        ]
    )

    targets = torch.tensor([1])

    assert topk_accuracy(
        logits,
        targets,
        k=2,
    ) == pytest.approx(1.0)


def test_topk_accuracy_invalid_k():
    logits = torch.randn(4, 10)

    targets = torch.randint(
        0,
        10,
        (4,),
    )

    with pytest.raises(
        ValueError
    ):
        topk_accuracy(
            logits,
            targets,
            k=0,
        )


def test_compute_metrics_keys():
    y_true = torch.tensor(
        [0, 1, 2]
    )

    y_pred = torch.tensor(
        [0, 1, 2]
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    assert set(
        metrics.keys()
    ) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }


def test_compute_metrics_values_are_float():
    y_true = torch.tensor(
        [0, 1, 2]
    )

    y_pred = torch.tensor(
        [0, 1, 2]
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    for value in metrics.values():
        assert isinstance(
            value,
            float,
        )


def test_compute_metrics_perfect_prediction():
    y_true = torch.tensor(
        [0, 1, 2, 3]
    )

    y_pred = torch.tensor(
        [0, 1, 2, 3]
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    assert metrics[
        "accuracy"
    ] == pytest.approx(1.0)

    assert metrics[
        "macro_f1"
    ] == pytest.approx(1.0)
