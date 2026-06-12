import pytest
import torch

from src.evaluation.utils import (
    validate_predictions,
    move_to_device,
    logits_to_predictions,
    logits_to_probabilities,
    batch_accuracy,
    count_correct,
    prediction_distribution,
    confidence_scores,
    topk_predictions,
)

# pytest tests/evaluation/test_utils.py -v

def test_validate_predictions_pass():
    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.tensor([0, 1, 2])

    validate_predictions(
        y_true,
        y_pred,
    )


def test_validate_predictions_empty_true():
    with pytest.raises(ValueError):
        validate_predictions(
            torch.tensor([]),
            torch.tensor([]),
        )


def test_validate_predictions_empty_pred():
    with pytest.raises(ValueError):
        validate_predictions(
            torch.tensor([1]),
            torch.tensor([]),
        )


def test_validate_predictions_shape_mismatch():
    with pytest.raises(ValueError):
        validate_predictions(
            torch.tensor([1, 2]),
            torch.tensor([1]),
        )


def test_move_to_device_cpu():
    x = torch.randn(4)

    y = move_to_device(
        x,
        "cpu",
    )

    assert y.device.type == "cpu"


def test_logits_to_predictions():
    logits = torch.tensor(
        [
            [1.0, 5.0],
            [9.0, 2.0],
        ]
    )

    preds = logits_to_predictions(
        logits
    )

    expected = torch.tensor(
        [1, 0]
    )

    assert torch.equal(
        preds,
        expected,
    )


def test_logits_to_predictions_invalid_dim():
    with pytest.raises(ValueError):
        logits_to_predictions(
            torch.randn(10)
        )


def test_logits_to_probabilities_shape():
    logits = torch.randn(
        8,
        10,
    )

    probs = logits_to_probabilities(
        logits
    )

    assert probs.shape == (
        8,
        10,
    )


def test_logits_to_probabilities_sum_to_one():
    logits = torch.randn(
        8,
        10,
    )

    probs = logits_to_probabilities(
        logits
    )

    sums = probs.sum(
        dim=1
    )

    assert torch.allclose(
        sums,
        torch.ones_like(
            sums
        ),
        atol=1e-5,
    )


def test_logits_to_probabilities_invalid_dim():
    with pytest.raises(ValueError):
        logits_to_probabilities(
            torch.randn(10)
        )


def test_batch_accuracy_perfect():
    logits = torch.tensor(
        [
            [5.0, 1.0],
            [1.0, 5.0],
        ]
    )

    targets = torch.tensor(
        [0, 1]
    )

    assert (
        batch_accuracy(
            logits,
            targets,
        )
        == 1.0
    )


def test_batch_accuracy_partial():
    logits = torch.tensor(
        [
            [5.0, 1.0],
            [5.0, 1.0],
        ]
    )

    targets = torch.tensor(
        [0, 1]
    )

    assert (
        batch_accuracy(
            logits,
            targets,
        )
        == 0.5
    )


def test_count_correct():
    preds = torch.tensor(
        [0, 1, 1, 0]
    )

    targets = torch.tensor(
        [0, 1, 0, 0]
    )

    assert (
        count_correct(
            preds,
            targets,
        )
        == 3
    )


def test_prediction_distribution():
    preds = torch.tensor(
        [0, 1, 1, 2, 2, 2]
    )

    dist = prediction_distribution(
        preds,
        num_classes=3,
    )

    expected = torch.tensor(
        [1, 2, 3]
    )

    assert torch.equal(
        dist,
        expected,
    )


def test_prediction_distribution_invalid_classes():
    with pytest.raises(ValueError):
        prediction_distribution(
            torch.tensor([0]),
            0,
        )


def test_confidence_scores():
    logits = torch.tensor(
        [
            [10.0, 1.0],
            [1.0, 10.0],
        ]
    )

    scores = confidence_scores(
        logits
    )

    assert scores.shape == (
        2,
    )

    assert torch.all(
        scores > 0.0
    )

    assert torch.all(
        scores <= 1.0
    )


def test_topk_predictions_shapes():
    logits = torch.randn(
        4,
        10,
    )

    indices, scores = topk_predictions(
        logits,
        k=3,
    )

    assert indices.shape == (
        4,
        3,
    )

    assert scores.shape == (
        4,
        3,
    )


def test_topk_predictions_sorted():
    logits = torch.randn(
        4,
        10,
    )

    _, scores = topk_predictions(
        logits,
        k=3,
    )

    assert torch.all(
        scores[:, 0]
        >= scores[:, 1]
    )

    assert torch.all(
        scores[:, 1]
        >= scores[:, 2]
    )


def test_topk_invalid_k():
    with pytest.raises(ValueError):
        topk_predictions(
            torch.randn(
                4,
                10,
            ),
            k=0,
        )