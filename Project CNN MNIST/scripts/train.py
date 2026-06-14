# scripts/train.py
"""
Training script for CNN MNIST model.

Usage:
    python -m scripts.train --config configs/experiments/baseline.yaml
    python -m scripts.train --config configs/experiments/baseline.yaml --device cuda
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import yaml

from src.data.dataloader import (
    create_train_dataloader,
    create_valid_dataloader,
)
from src.models.factory import ModelFactory
from src.training.losses import get_loss_class
from src.training.optimizers import build_optimizer_from_config
from src.training.schedulers import build_scheduler
from src.training.trainer import Trainer
from src.training.checkpoint import ModelCheckpoint
from src.training.early_stopping import EarlyStopping


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
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to configuration file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    logger.info(f"Loading config from: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.

    Raises
    ------
    ValueError
        If configuration is invalid.
    """
    required_keys = ["data", "model", "training", "optimizer"]

    for key in required_keys:
        if key not in config:
            raise ValueError(
                f"Missing required config key: {key}"
            )

    # Validate data config
    if "data_dir" not in config["data"]:
        raise ValueError(
            "Missing 'data_dir' in data config"
        )

    if "batch_size" not in config["data"]:
        raise ValueError(
            "Missing 'batch_size' in data config"
        )

    # Validate model config
    if "name" not in config["model"]:
        raise ValueError(
            "Missing 'name' in model config"
        )

    # Validate training config
    if "epochs" not in config["training"]:
        raise ValueError(
            "Missing 'epochs' in training config"
        )


# ============================================================
# Device Management
# ============================================================

def get_device(device_str: Optional[str] = None) -> str:
    """
    Get device string, defaulting to CUDA if available.

    Parameters
    ----------
    device_str : Optional[str]
        Requested device: 'cuda', 'cpu', or None.

    Returns
    -------
    str
        Device string: 'cuda' or 'cpu'.
    """
    if device_str is not None:
        if device_str.lower() == "cuda":
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but not available. Using CPU."
                )
                return "cpu"
            return "cuda"
        elif device_str.lower() == "cpu":
            return "cpu"
        else:
            raise ValueError(
                f"Unknown device: {device_str}. Use 'cuda' or 'cpu'."
            )

    # Auto-detect
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device


# ============================================================
# Training Pipeline
# ============================================================

def train(
    config_path: str | Path,
    device: Optional[str] = None,
) -> None:
    """
    Main training function.

    Parameters
    ----------
    config_path : str | Path
        Path to configuration file.

    device : Optional[str]
        Device to use ('cuda' or 'cpu'). If None, auto-detects.
    """
    # Load and validate config
    config = load_config(config_path)
    validate_config(config)

    # Get device
    device = get_device(device)
    logger.info(f"Using device: {device}")

    # Extract configs
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})
    artifacts_cfg = config.get("artifacts", {})

    # ========================================================
    # Data Loading
    # ========================================================

    logger.info("Loading data...")

    data_dir = data_cfg.get("data_dir", "data")
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 0)

    train_loader = create_train_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    valid_loader = create_valid_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logger.info(
        f"Train loader: {len(train_loader)} batches, "
        f"Valid loader: {len(valid_loader)} batches"
    )

    # ========================================================
    # Model Creation
    # ========================================================

    logger.info("Creating model...")

    model_name = model_cfg.get("name", "mnist_cnn")
    num_classes = model_cfg.get("num_classes", 10)
    hidden_features = model_cfg.get("hidden_features", 128)

    model = ModelFactory.create(
        model_name=model_name,
        num_classes=num_classes,
        hidden_dim=hidden_features,
    )

    model = model.to(device)

    logger.info(
        f"Model: {model_name} "
        f"(Parameters: {model.num_parameters:,})"
    )

    # ========================================================
    # Loss Function
    # ========================================================

    loss_name = training_cfg.get("loss", "cross_entropy")
    loss_cls = get_loss_class(loss_name)
    criterion = loss_cls()

    logger.info(f"Loss function: {loss_name}")

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = build_optimizer_from_config(
        model=model,
        config=optimizer_cfg,
    )

    logger.info(
        f"Optimizer: {optimizer.__class__.__name__}"
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = None
    if scheduler_cfg and scheduler_cfg.get("name") != "none":
        scheduler = build_scheduler(
            optimizer=optimizer,
            config=scheduler_cfg,
        )
        logger.info(
            f"Scheduler: {scheduler.__class__.__name__}"
        )
    else:
        logger.info("Scheduler: None")

    # ========================================================
    # Checkpoint Manager
    # ========================================================

    checkpoint_dir = Path(
        artifacts_cfg.get(
            "checkpoint_dir",
            "artifacts/checkpoints"
        )
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_manager = ModelCheckpoint(checkpoint_dir)

    logger.info(f"Checkpoint dir: {checkpoint_dir}")

    # ========================================================
    # Early Stopping
    # ========================================================

    patience = training_cfg.get("patience", 10)
    early_stopping = EarlyStopping(patience=patience)

    logger.info(f"Early stopping patience: {patience}")

    # ========================================================
    # Trainer Setup
    # ========================================================

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint=checkpoint_manager,
        early_stopping=early_stopping,
    )

    # ========================================================
    # Training Loop
    # ========================================================

    epochs = training_cfg.get("epochs", 10)

    logger.info(f"Starting training for {epochs} epochs...")

    trainer.fit(
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs=epochs,
    )

    logger.info("Training completed!")

    # Initialize variables
    best_val_loss = None
    best_val_acc = None
    num_epochs_trained = 0

    if hasattr(trainer, "history"):
        history = trainer.history
        num_epochs_trained = history.num_epochs()
        logger.info(f"Epochs trained: {num_epochs_trained}")
        
        try:
            best_val_loss = history.best("val_loss", mode="min")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
        except (ValueError, KeyError):
            logger.info("No validation loss recorded")
        
        try:
            best_val_acc = history.best("val_accuracy", mode="max")
            logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        except (ValueError, KeyError):
            logger.info("No validation accuracy recorded")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    if best_val_loss is not None:
        print(f"Best val loss: {best_val_loss:.4f}")
    if best_val_acc is not None:
        print(f"Best val accuracy: {best_val_acc:.4f}")
    print("=" * 60 + "\n")


# ============================================================
# CLI Argument Parsing
# ============================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CNN model on MNIST dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.train --config configs/experiments/baseline.yaml
  python -m scripts.train --config configs/experiments/baseline.yaml --device cuda
  python -m scripts.train --config configs/experiments/cnn_small.yaml --device cpu
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/baseline.yaml",
        help="Path to configuration file (default: configs/experiments/baseline.yaml)",
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
        train(
            config_path=args.config,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
