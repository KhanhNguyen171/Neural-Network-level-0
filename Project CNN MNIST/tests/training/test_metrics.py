# tests/training/test_metrics.py

import pytest
import torch

from src.training.metrics import (
    AverageMeter,
    accuracy_score,
    topk_accuracy,
    compute_metrics,
)

# pytest tests/training/test_metrics.py -v


# accuracy_score

def test_accuracy_score_perfect_prediction():
    logits = torch.tensor(
        [
            [10.0, 1.0],
            [1.0, 10.0],
            [10.0, 1.0],
        ]
    )

    targets = torch.tensor([0, 1, 0])

    acc = accuracy_score(logits, targets)

    assert acc == pytest.approx(1.0)


def test_accuracy_score_partial_prediction():
    logits = torch.tensor(
        [
            [10.0, 1.0],
            [1.0, 10.0],
            [1.0, 10.0],
            [10.0, 1.0],
        ]
    )

    targets = torch.tensor([0, 1, 0, 1])

    acc = accuracy_score(logits, targets)

    assert acc == pytest.approx(0.5)


def test_accuracy_score_returns_float():
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    acc = accuracy_score(logits, targets)

    assert isinstance(acc, float)


def test_accuracy_score_batch_size_one():
    logits = torch.tensor([[0.1, 0.9]])
    targets = torch.tensor([1])

    acc = accuracy_score(logits, targets)

    assert acc == pytest.approx(1.0)


# topk_accuracy

def test_topk_accuracy_top1():
    logits = torch.tensor(
        [
            [10.0, 1.0, 0.0],
            [0.0, 10.0, 1.0],
        ]
    )

    targets = torch.tensor([0, 1])

    acc = topk_accuracy(logits, targets, k=1)

    assert acc == pytest.approx(1.0)


def test_topk_accuracy_top2():
    logits = torch.tensor(
        [
            [0.9, 0.8, 1.0],
        ]
    )

    targets = torch.tensor([0])

    acc = topk_accuracy(logits, targets, k=2)

    assert acc == pytest.approx(1.0)


def test_topk_accuracy_top1_fail():
    logits = torch.tensor(
        [
            [0.9, 0.8, 1.0],
        ]
    )

    targets = torch.tensor([0])

    acc = topk_accuracy(logits, targets, k=1)

    assert acc == pytest.approx(0.0)


def test_topk_accuracy_returns_float():
    logits = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))

    acc = topk_accuracy(logits, targets, k=3)

    assert isinstance(acc, float)


def test_topk_accuracy_k_equals_num_classes():
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    acc = topk_accuracy(logits, targets, k=10)

    assert acc == pytest.approx(1.0)


# compute_metrics

def test_compute_metrics_contains_accuracy():
    logits = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))

    metrics = compute_metrics(logits, targets)

    assert "accuracy" in metrics


def test_compute_metrics_returns_dict():
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    metrics = compute_metrics(logits, targets)

    assert isinstance(metrics, dict)


def test_compute_metrics_values_are_numeric():
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    metrics = compute_metrics(logits, targets)

    for value in metrics.values():
        assert isinstance(value, (float, int))


# AverageMeter

def test_average_meter_initial_state():
    meter = AverageMeter()

    assert meter.count == 0
    assert meter.sum == 0
    assert meter.avg == 0


def test_average_meter_single_update():
    meter = AverageMeter()

    meter.update(5)

    assert meter.count == 1
    assert meter.sum == 5
    assert meter.avg == pytest.approx(5.0)


def test_average_meter_multiple_updates():
    meter = AverageMeter()

    meter.update(2)
    meter.update(4)
    meter.update(6)

    assert meter.count == 3
    assert meter.sum == 12
    assert meter.avg == pytest.approx(4.0)


def test_average_meter_update_with_n():
    meter = AverageMeter()

    meter.update(2, n=5)

    assert meter.count == 5
    assert meter.sum == 10
    assert meter.avg == pytest.approx(2.0)


def test_average_meter_weighted_updates():
    meter = AverageMeter()

    meter.update(1, n=2)
    meter.update(3, n=4)

    expected = (1 * 2 + 3 * 4) / 6

    assert meter.avg == pytest.approx(expected)


def test_average_meter_reset():
    meter = AverageMeter()

    meter.update(5)
    meter.update(10)

    meter.reset()

    assert meter.count == 0
    assert meter.sum == 0
    assert meter.avg == 0


def test_average_meter_many_updates():
    meter = AverageMeter()

    for i in range(1, 101):
        meter.update(i)

    assert meter.count == 100
    assert meter.avg == pytest.approx(50.5)


def test_average_meter_float_values():
    meter = AverageMeter()

    meter.update(0.5)
    meter.update(1.5)

    assert meter.avg == pytest.approx(1.0)