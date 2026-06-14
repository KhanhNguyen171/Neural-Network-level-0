# scripts/benchmark.py
"""
Benchmark script for model performance testing.

Usage:
    python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml
    python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml --batch-sizes 1 32 64 128
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import yaml
import time

from src.models.factory import ModelFactory
from src.data.dataloader import create_test_dataloader


# ============================================================
# Logging Setup
# ============================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================
# Config Management
# ============================================================

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ============================================================
# Device Management
# ============================================================

def get_device(device_str: Optional[str] = None) -> str:
    """Get device string."""
    if device_str is not None:
        if device_str.lower() == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Using CPU.")
                return "cpu"
            return "cuda"
        elif device_str.lower() == "cpu":
            return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# Checkpoint Loading
# ============================================================

def load_checkpoint(checkpoint_path: str | Path, device: str) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    return checkpoint


# ============================================================
# Benchmark Functions
# ============================================================

def benchmark_inference(
    model: nn.Module,
    device: str,
    input_shape: tuple = (1, 1, 28, 28),
    num_iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark model inference performance.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.

    device : str
        Device to benchmark on.

    input_shape : tuple
        Input tensor shape.

    num_iterations : int
        Number of iterations for benchmarking.

    Returns
    -------
    Dict[str, float]
        Benchmark results (latency, throughput, etc.).
    """
    model.eval()

    # Warmup
    logger.info("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            dummy_input = torch.randn(
                input_shape,
                device=device,
            )
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    logger.info(f"Running {num_iterations} iterations...")

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            dummy_input = torch.randn(
                input_shape,
                device=device,
            )
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start_time

    batch_size = input_shape[0]

    # Calculate metrics
    avg_latency = total_time / num_iterations * 1000  # ms
    throughput = (num_iterations * batch_size) / total_time  # samples/sec

    results = {
        "avg_latency_ms": avg_latency,
        "throughput_samples_per_sec": throughput,
        "total_time_sec": total_time,
        "num_iterations": num_iterations,
    }

    return results


def benchmark_memory(
    model: nn.Module,
    device: str,
    input_shape: tuple = (1, 1, 28, 28),
) -> Dict[str, float]:
    """
    Benchmark model memory usage.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.

    device : str
        Device to benchmark on.

    input_shape : tuple
        Input tensor shape.

    Returns
    -------
    Dict[str, float]
        Memory usage statistics (MB).
    """
    model.eval()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Measure peak memory
    with torch.no_grad():
        dummy_input = torch.randn(input_shape, device=device)
        _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        peak_memory = None

    # Model size
    model_size = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / 1024 / 1024  # MB

    results = {
        "model_size_mb": model_size,
        "peak_memory_mb": peak_memory,
    }

    return results


def benchmark_model_complexity(
    model: nn.Module,
) -> Dict[str, int]:
    """
    Get model complexity metrics.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.

    Returns
    -------
    Dict[str, int]
        Complexity metrics.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    results = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    return results


# ============================================================
# Benchmark Pipeline
# ============================================================

def benchmark(
    checkpoint_path: str | Path,
    config_path: str | Path,
    batch_sizes: Optional[List[int]] = None,
    num_iterations: int = 100,
    device: Optional[str] = None,
) -> None:
    """
    Run comprehensive model benchmarking.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to model checkpoint.

    config_path : str | Path
        Path to configuration file.

    batch_sizes : Optional[List[int]]
        Batch sizes to benchmark. If None, uses [1, 32, 64].

    num_iterations : int
        Number of iterations for latency benchmarking.

    device : Optional[str]
        Device to use. If None, auto-detects.
    """
    # Defaults
    if batch_sizes is None:
        batch_sizes = [1, 32, 64]

    # Get device
    device = get_device(device)
    logger.info(f"Using device: {device}")

    # Load configuration
    config = load_config(config_path)
    model_cfg = config.get("model", {})

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)

    # ========================================================
    # Model Creation and Loading
    # ========================================================

    logger.info("Creating and loading model...")

    model_name = model_cfg.get("name", "mnist_cnn")
    num_classes = model_cfg.get("num_classes", 10)
    hidden_features = model_cfg.get("hidden_features", 128)

    model = ModelFactory.create(
        model_name=model_name,
        num_classes=num_classes,
        hidden_dim=hidden_features,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # ========================================================
    # Complexity Benchmarking
    # ========================================================

    logger.info("Benchmarking model complexity...")

    complexity = benchmark_model_complexity(model)

    # ========================================================
    # Memory Benchmarking
    # ========================================================

    logger.info("Benchmarking memory usage...")

    memory = benchmark_memory(model, device)

    # ========================================================
    # Inference Benchmarking
    # ========================================================

    inference_results = {}

    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size {batch_size}...")

        input_shape = (batch_size, 1, 28, 28)

        results = benchmark_inference(
            model,
            device,
            input_shape=input_shape,
            num_iterations=num_iterations,
        )

        inference_results[batch_size] = results

    # ========================================================
    # Print Results
    # ========================================================

    print("\n" + "=" * 80)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # Complexity
    print("\n" + "-" * 80)
    print("COMPLEXITY METRICS")
    print("-" * 80)
    print(f"Total Parameters:     {complexity['total_parameters']:>15,}")
    print(f"Trainable Parameters: {complexity['trainable_parameters']:>15,}")

    # Memory
    print("\n" + "-" * 80)
    print("MEMORY METRICS")
    print("-" * 80)
    print(f"Model Size:     {memory['model_size_mb']:>10.2f} MB")
    if memory['peak_memory_mb'] is not None:
        print(f"Peak Memory:    {memory['peak_memory_mb']:>10.2f} MB")

    # Inference
    print("\n" + "-" * 80)
    print("INFERENCE LATENCY & THROUGHPUT")
    print("-" * 80)
    print(
        f"{'Batch Size':<15} "
        f"{'Avg Latency (ms)':<20} "
        f"{'Throughput (samples/sec)':<25}"
    )
    print("-" * 80)

    for batch_size in batch_sizes:
        results = inference_results[batch_size]
        print(
            f"{batch_size:<15} "
            f"{results['avg_latency_ms']:<20.4f} "
            f"{results['throughput_samples_per_sec']:<25.2f}"
        )

    print("=" * 80 + "\n")


# ============================================================
# CLI Argument Parsing
# ============================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark trained CNN model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml
  python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --batch-sizes 1 32 64 128 256
  python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --device cuda --num-iterations 200
        """
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/baseline.yaml",
        help="Path to configuration file (default: configs/experiments/baseline.yaml)",
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs='+',
        default=[1, 32, 64],
        help="Batch sizes to benchmark (default: 1 32 64)",
    )

    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of iterations for latency benchmarking (default: 100)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (cuda or cpu). If not specified, auto-detects.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)

    try:
        benchmark(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            batch_sizes=args.batch_sizes,
            num_iterations=args.num_iterations,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        raise
