# scripts/evaluate.py
"""
Evaluation script for trained CNN MNIST models.

Usage:
    python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml
    python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --split test
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import yaml

from src.data.dataloader import (
    create_test_dataloader,
    create_valid_dataloader,
)
from src.models.factory import ModelFactory
from src.evaluation.evaluator import Evaluator
from src.training.losses import get_loss_class


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
        else:
            raise ValueError(
                f"Unknown device: {device_str}. Use 'cuda' or 'cpu'."
            )

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# Checkpoint Loading
# ============================================================

def load_checkpoint(checkpoint_path: str | Path, device: str) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to checkpoint file.

    device : str
        Device to load checkpoint on.

    Returns
    -------
    Dict[str, Any]
        Checkpoint dictionary.
    """
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
# Evaluation Pipeline
# ============================================================

def evaluate(
    checkpoint_path: str | Path,
    config_path: str | Path,
    split: str = "test",
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to model checkpoint.

    config_path : str | Path
        Path to configuration file.

    split : str
        Dataset split: 'test' or 'valid' (default: 'test').

    device : Optional[str]
        Device to use. If None, auto-detects.

    Returns
    -------
    Dict[str, float]
        Evaluation metrics.
    """
    # Get device
    device = get_device(device)
    logger.info(f"Using device: {device}")

    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded config from: {config_path}")

    data_cfg = config.get("data", {})
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
    model.eval()

    logger.info(f"Model loaded: {model_name}")

    # ========================================================
    # Data Loading
    # ========================================================

    logger.info(f"Loading {split} dataset...")

    data_dir = data_cfg.get("data_dir", "data")
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 0)

    if split == "test":
        data_loader = create_test_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif split == "valid":
        data_loader = create_valid_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unknown split: {split}")

    logger.info(f"Data loader: {len(data_loader)} batches")

    # ========================================================
    # Loss Function
    # ========================================================

    loss_name = config.get("training", {}).get("loss", "cross_entropy")
    loss_cls = get_loss_class(loss_name)
    criterion = loss_cls()

    # ========================================================
    # Evaluation
    # ========================================================

    logger.info("Evaluating model...")

    evaluator = Evaluator(
        model=model,
        device=device,
        criterion=criterion,
    )

    metrics = evaluator.evaluate(
        dataloader=data_loader,
    )

    # ========================================================
    # Print Results
    # ========================================================

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({split.upper()} SET)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("-" * 60)

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:<30} {metric_value:>8.4f}")
        else:
            print(f"{metric_name:<30} {str(metric_value):>8}")

    print("=" * 60 + "\n")

    return metrics


# ============================================================
# CLI Argument Parsing
# ============================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained CNN model on MNIST dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml
  python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml --split test
  python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml --device cuda
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
        "--split",
        type=str,
        choices=["test", "valid"],
        default="test",
        help="Dataset split to evaluate on (default: test)",
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
        evaluate(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            split=args.split,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise
