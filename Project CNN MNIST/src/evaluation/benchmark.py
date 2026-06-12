from __future__ import annotations

import time
from typing import Dict

import torch

from src.models.utils import count_parameters


def model_size_mb(
    model: torch.nn.Module,
) -> float:
    """
    Approximate model size in MB.
    """
    total_bytes = 0

    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes / (1024 ** 2)


def benchmark_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    Benchmark inference latency.
    """
    if num_runs <= 0:
        raise ValueError("num_runs must be positive.")

    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative.")

    model.eval()

    with torch.no_grad():
        for _ in range(warmup_runs):
            model(input_tensor)

        start = time.perf_counter()

        for _ in range(num_runs):
            model(input_tensor)

        end = time.perf_counter()

    total_time = end - start

    return {
        "total_time": total_time,
        "avg_latency_ms": (total_time / num_runs) * 1000.0,
        "throughput": (
            input_tensor.shape[0] * num_runs
        ) / total_time,
    }


def benchmark_model(
    model: torch.nn.Module,
    input_shape=(1, 1, 28, 28),
    num_runs: int = 100,
) -> Dict:
    """
    Complete benchmark report.
    """
    device = next(model.parameters()).device

    dummy = torch.randn(
        *input_shape,
        device=device,
    )

    results = benchmark_inference(
        model=model,
        input_tensor=dummy,
        num_runs=num_runs,
    )

    results["parameters"] = count_parameters(model)
    results["model_size_mb"] = model_size_mb(model)

    return results


def format_benchmark(
    benchmark: Dict,
) -> str:
    """
    Convert benchmark dictionary into string.
    """
    return (
        f"Parameters      : {benchmark['parameters']}\n"
        f"Model Size (MB) : {benchmark['model_size_mb']:.4f}\n"
        f"Latency (ms)    : {benchmark['avg_latency_ms']:.4f}\n"
        f"Throughput      : {benchmark['throughput']:.2f} samples/s"
    )