# tests/evaluation/test_evaluator.py

import torch
import torch.nn as nn
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

from src.evaluation.evaluator import (
    Evaluator,
)

# pytest tests/evaluation/test_evaluator.py -v

# Helpers

class TinyNet(nn.Module):

    def __init__(
        self,
        in_features=20,
        num_classes=10,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                in_features,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                num_classes,
            ),
        )

    def forward(self, x):
        return self.net(x)


def create_model():

    return TinyNet()


def create_loader(
    samples=64,
    features=20,
    classes=10,
    batch_size=16,
):

    x = torch.randn(
        samples,
        features,
    )

    y = torch.randint(
        0,
        classes,
        (samples,),
    )

    dataset = TensorDataset(
        x,
        y,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


# Initialization

def test_evaluator_creation():

    model = create_model()

    evaluator = Evaluator(
        model=model,
    )

    assert evaluator.model is model


def test_repr_contains_class_name():

    evaluator = Evaluator(
        create_model()
    )

    text = repr(evaluator)

    assert "Evaluator" in text


# Predict

def test_predict_returns_tensor():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    preds = evaluator.predict(
        loader
    )

    assert isinstance(
        preds,
        torch.Tensor,
    )


def test_predict_shape():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader(
        samples=50
    )

    preds = evaluator.predict(
        loader
    )

    assert preds.shape == (
        50,
    )


def test_predict_dtype():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    preds = evaluator.predict(
        loader
    )

    assert preds.dtype == torch.long


def test_predict_class_range():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    preds = evaluator.predict(
        loader
    )

    assert preds.min() >= 0
    assert preds.max() < 10


# Predict Proba

def test_predict_proba_returns_tensor():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    probs = (
        evaluator.predict_proba(
            loader
        )
    )

    assert isinstance(
        probs,
        torch.Tensor,
    )


def test_predict_proba_shape():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader(
        samples=32
    )

    probs = (
        evaluator.predict_proba(
            loader
        )
    )

    assert probs.shape == (
        32,
        10,
    )


def test_predict_proba_row_sum_one():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    probs = (
        evaluator.predict_proba(
            loader
        )
    )

    row_sums = probs.sum(
        dim=1
    )

    expected = torch.ones(
        len(probs)
    )

    assert torch.allclose(
        row_sums,
        expected,
        atol=1e-5,
    )


# Evaluate

def test_evaluate_returns_dict():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    metrics = evaluator.evaluate(
        loader
    )

    assert isinstance(
        metrics,
        dict,
    )


def test_evaluate_keys():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    metrics = evaluator.evaluate(
        loader
    )

    assert set(
        metrics.keys()
    ) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }


def test_evaluate_with_loss():

    evaluator = Evaluator(
        model=create_model(),
        criterion=nn.CrossEntropyLoss(),
    )

    loader = create_loader()

    metrics = evaluator.evaluate(
        loader
    )

    assert "loss" in metrics


def test_evaluate_loss_is_float():

    evaluator = Evaluator(
        model=create_model(),
        criterion=nn.CrossEntropyLoss(),
    )

    loader = create_loader()

    metrics = evaluator.evaluate(
        loader
    )

    assert isinstance(
        metrics["loss"],
        float,
    )


def test_accuracy_range():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    metrics = evaluator.evaluate(
        loader
    )

    assert (
        0.0
        <= metrics["accuracy"]
        <= 1.0
    )


def test_macro_precision_range():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    metrics = evaluator.evaluate(
        loader
    )

    assert (
        0.0
        <= metrics[
            "macro_precision"
        ]
        <= 1.0
    )


def test_macro_recall_range():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    metrics = evaluator.evaluate(
        loader
    )

    assert (
        0.0
        <= metrics[
            "macro_recall"
        ]
        <= 1.0
    )


def test_macro_f1_range():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    metrics = evaluator.evaluate(
        loader
    )

    assert (
        0.0
        <= metrics[
            "macro_f1"
        ]
        <= 1.0
    )


# Device

def test_device_cpu():

    evaluator = Evaluator(
        model=create_model(),
        device="cpu",
    )

    assert (
        evaluator.device.type
        == "cpu"
    )


# Consistency

def test_predict_and_proba_same_length():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader(
        samples=40
    )

    preds = evaluator.predict(
        loader
    )

    probs = (
        evaluator.predict_proba(
            loader
        )
    )

    assert len(preds) == len(
        probs
    )


def test_evaluate_multiple_calls():

    evaluator = Evaluator(
        create_model()
    )

    loader = create_loader()

    m1 = evaluator.evaluate(
        loader
    )

    m2 = evaluator.evaluate(
        loader
    )

    assert (
        set(m1.keys())
        ==
        set(m2.keys())
    )
