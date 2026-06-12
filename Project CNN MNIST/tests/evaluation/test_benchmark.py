import torch
import torch.nn as nn
import pytest

from src.evaluation.benchmark import (
    model_size_mb,
    benchmark_inference,
    benchmark_model,
    format_benchmark,
)

# pytest tests/evaluation/test_benchmark.py -v

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
def sample_input():
    return torch.randn(
        8,
        1,
        28,
        28,
    )


def test_model_size_mb_returns_float(model):
    size = model_size_mb(model)

    assert isinstance(size, float)


def test_model_size_mb_positive(model):
    size = model_size_mb(model)

    assert size > 0


def test_benchmark_inference_returns_dict(
    model,
    sample_input,
):
    results = benchmark_inference(
        model,
        sample_input,
        num_runs=5,
    )

    assert isinstance(results, dict)


def test_benchmark_inference_contains_keys(
    model,
    sample_input,
):
    results = benchmark_inference(
        model,
        sample_input,
        num_runs=5,
    )

    assert "total_time" in results
    assert "avg_latency_ms" in results
    assert "throughput" in results


def test_latency_positive(
    model,
    sample_input,
):
    results = benchmark_inference(
        model,
        sample_input,
        num_runs=5,
    )

    assert results["avg_latency_ms"] > 0


def test_total_time_positive(
    model,
    sample_input,
):
    results = benchmark_inference(
        model,
        sample_input,
        num_runs=5,
    )

    assert results["total_time"] > 0


def test_throughput_positive(
    model,
    sample_input,
):
    results = benchmark_inference(
        model,
        sample_input,
        num_runs=5,
    )

    assert results["throughput"] > 0


def test_benchmark_model_returns_dict(
    model,
):
    results = benchmark_model(
        model,
        num_runs=5,
    )

    assert isinstance(results, dict)


def test_benchmark_model_contains_parameters(
    model,
):
    results = benchmark_model(
        model,
        num_runs=5,
    )

    assert "parameters" in results


def test_benchmark_model_contains_size(
    model,
):
    results = benchmark_model(
        model,
        num_runs=5,
    )

    assert "model_size_mb" in results


def test_benchmark_model_parameter_count_positive(
    model,
):
    results = benchmark_model(
        model,
        num_runs=5,
    )

    assert results["parameters"] > 0


def test_benchmark_model_size_positive(
    model,
):
    results = benchmark_model(
        model,
        num_runs=5,
    )

    assert results["model_size_mb"] > 0


def test_format_benchmark_returns_string(
    model,
):
    benchmark = benchmark_model(
        model,
        num_runs=5,
    )

    text = format_benchmark(
        benchmark
    )

    assert isinstance(text, str)


def test_format_benchmark_contains_fields(
    model,
):
    benchmark = benchmark_model(
        model,
        num_runs=5,
    )

    text = format_benchmark(
        benchmark
    )

    assert "Parameters" in text
    assert "Latency" in text
    assert "Throughput" in text


def test_invalid_num_runs_raises(
    model,
    sample_input,
):
    with pytest.raises(ValueError):
        benchmark_inference(
            model,
            sample_input,
            num_runs=0,
        )


def test_negative_warmup_raises(
    model,
    sample_input,
):
    with pytest.raises(ValueError):
        benchmark_inference(
            model,
            sample_input,
            num_runs=5,
            warmup_runs=-1,
        )


def test_benchmark_runs_multiple_batch_sizes(
    model,
):
    for batch_size in [1, 8, 16]:
        x = torch.randn(
            batch_size,
            1,
            28,
            28,
        )

        results = benchmark_inference(
            model,
            x,
            num_runs=3,
        )

        assert results["throughput"] > 0


def test_benchmark_output_types(
    model,
):
    results = benchmark_model(
        model,
        num_runs=3,
    )

    assert isinstance(
        results["parameters"],
        int,
    )

    assert isinstance(
        results["model_size_mb"],
        float,
    )

    assert isinstance(
        results["avg_latency_ms"],
        float,
    )

    assert isinstance(
        results["throughput"],
        float,
    )