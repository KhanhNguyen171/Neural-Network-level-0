import pytest
import torch
import torch.nn as nn

from src.evaluation.inference import (
    InferenceEngine,
)

# pytest tests/evaluation/test_inference.py -v

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10),
        )

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def model():
    return TinyNet()


@pytest.fixture
def engine(model):
    return InferenceEngine(model)


@pytest.fixture
def batch():
    return torch.randn(
        8,
        1,
        28,
        28,
    )


@pytest.fixture
def single_image():
    return torch.randn(
        1,
        28,
        28,
    )


def test_create_engine(engine):
    assert isinstance(
        engine,
        InferenceEngine,
    )


def test_repr(engine):
    text = repr(engine)

    assert "InferenceEngine" in text


def test_predict_returns_tensor(
    engine,
    batch,
):
    preds = engine.predict(batch)

    assert isinstance(
        preds,
        torch.Tensor,
    )


def test_predict_shape(
    engine,
    batch,
):
    preds = engine.predict(batch)

    assert preds.shape == (8,)


def test_predict_dtype(
    engine,
    batch,
):
    preds = engine.predict(batch)

    assert preds.dtype == torch.int64


def test_predict_class_range(
    engine,
    batch,
):
    preds = engine.predict(batch)

    assert torch.all(preds >= 0)
    assert torch.all(preds < 10)


def test_predict_proba_returns_tensor(
    engine,
    batch,
):
    probs = engine.predict_proba(batch)

    assert isinstance(
        probs,
        torch.Tensor,
    )


def test_predict_proba_shape(
    engine,
    batch,
):
    probs = engine.predict_proba(batch)

    assert probs.shape == (
        8,
        10,
    )


def test_predict_proba_sums_to_one(
    engine,
    batch,
):
    probs = engine.predict_proba(batch)

    sums = probs.sum(dim=1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
    )


def test_predict_topk_returns_dict(
    engine,
    batch,
):
    result = engine.predict_topk(
        batch,
        k=3,
    )

    assert isinstance(
        result,
        dict,
    )


def test_predict_topk_contains_keys(
    engine,
    batch,
):
    result = engine.predict_topk(
        batch,
        k=3,
    )

    assert "indices" in result
    assert "scores" in result


def test_predict_topk_shape(
    engine,
    batch,
):
    result = engine.predict_topk(
        batch,
        k=3,
    )

    assert result["indices"].shape == (
        8,
        3,
    )

    assert result["scores"].shape == (
        8,
        3,
    )


def test_predict_topk_sorted(
    engine,
    batch,
):
    result = engine.predict_topk(
        batch,
        k=3,
    )

    scores = result["scores"]

    assert torch.all(
        scores[:, 0] >= scores[:, 1]
    )

    assert torch.all(
        scores[:, 1] >= scores[:, 2]
    )


def test_predict_single_returns_int(
    engine,
    single_image,
):
    pred = engine.predict_single(
        single_image
    )

    assert isinstance(
        pred,
        int,
    )


def test_predict_single_range(
    engine,
    single_image,
):
    pred = engine.predict_single(
        single_image
    )

    assert 0 <= pred < 10


def test_predict_single_with_batch_dim(
    engine,
):
    image = torch.randn(
        1,
        1,
        28,
        28,
    )

    pred = engine.predict_single(
        image
    )

    assert isinstance(
        pred,
        int,
    )


def test_invalid_k_raises(
    engine,
    batch,
):
    with pytest.raises(ValueError):
        engine.predict_topk(
            batch,
            k=0,
        )


def test_topk_k_equals_one(
    engine,
    batch,
):
    result = engine.predict_topk(
        batch,
        k=1,
    )

    assert result["indices"].shape == (
        8,
        1,
    )


def test_prediction_matches_argmax(
    engine,
    batch,
):
    preds = engine.predict(batch)

    probs = engine.predict_proba(batch)

    expected = probs.argmax(dim=1)

    assert torch.equal(
        preds,
        expected,
    )