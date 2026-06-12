import pytest
import torch

from src.evaluation.classification_report import (
    classification_report,
    report_to_string,
)

# pytest tests/evaluation/test_classification_report.py -v

def test_report_returns_dict():
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 1, 1])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=2,
    )

    assert isinstance(report, dict)


def test_report_contains_all_classes():
    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.tensor([0, 1, 2])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=3,
    )

    assert "0" in report
    assert "1" in report
    assert "2" in report


def test_report_contains_accuracy():
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 1, 1])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=2,
    )

    assert "accuracy" in report


def test_report_contains_macro_avg():
    y_true = torch.tensor([0, 1])
    y_pred = torch.tensor([0, 1])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=2,
    )

    assert "macro avg" in report


def test_perfect_predictions():
    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.tensor([0, 1, 2])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=3,
    )

    for cls in ["0", "1", "2"]:
        assert report[cls]["precision"] == 1.0
        assert report[cls]["recall"] == 1.0
        assert report[cls]["f1-score"] == 1.0

    assert report["accuracy"] == 1.0


def test_all_wrong_predictions():
    y_true = torch.tensor([0, 0, 1, 1])
    y_pred = torch.tensor([1, 1, 0, 0])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=2,
    )

    assert report["accuracy"] == 0.0


def test_support_values():
    y_true = torch.tensor([0, 0, 0, 1, 1])
    y_pred = torch.tensor([0, 0, 1, 1, 1])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=2,
    )

    assert report["0"]["support"] == 3
    assert report["1"]["support"] == 2


def test_metrics_range():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([0, 0, 2, 1, 1, 2])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=3,
    )

    for cls in ["0", "1", "2"]:
        assert 0.0 <= report[cls]["precision"] <= 1.0
        assert 0.0 <= report[cls]["recall"] <= 1.0
        assert 0.0 <= report[cls]["f1-score"] <= 1.0


def test_macro_avg_support():
    y_true = torch.tensor([0, 1, 2, 1])
    y_pred = torch.tensor([0, 1, 1, 1])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=3,
    )

    assert report["macro avg"]["support"] == 4


def test_empty_input_raises():
    with pytest.raises(ValueError):
        classification_report(
            torch.tensor([]),
            torch.tensor([]),
            num_classes=2,
        )


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        classification_report(
            torch.tensor([0, 1]),
            torch.tensor([0]),
            num_classes=2,
        )


def test_invalid_num_classes_raises():
    with pytest.raises(ValueError):
        classification_report(
            torch.tensor([0]),
            torch.tensor([0]),
            num_classes=0,
        )


def test_report_to_string_returns_string():
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 1, 1])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=2,
    )

    text = report_to_string(report)

    assert isinstance(text, str)


def test_report_to_string_contains_accuracy():
    y_true = torch.tensor([0, 1])
    y_pred = torch.tensor([0, 1])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=2,
    )

    text = report_to_string(report)

    assert "accuracy" in text


def test_report_to_string_contains_macro_avg():
    y_true = torch.tensor([0, 1])
    y_pred = torch.tensor([0, 1])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=2,
    )

    text = report_to_string(report)

    assert "macro avg" in text


def test_report_to_string_contains_class_names():
    y_true = torch.tensor([0, 1])
    y_pred = torch.tensor([0, 1])

    report = classification_report(
        y_true,
        y_pred,
        num_classes=2,
    )

    text = report_to_string(report)

    assert "0" in text
    assert "1" in text