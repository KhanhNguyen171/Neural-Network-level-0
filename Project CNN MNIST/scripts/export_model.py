# scripts/export_model.py
"""
Export trained model to different formats.

Usage:
    python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --format torchscript --output artifacts/models/model.pt
    python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --format onnx --output artifacts/models/model.onnx
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import yaml

from src.models.factory import ModelFactory


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
# Export Functions
# ============================================================

def export_torchscript(
    model: torch.nn.Module,
    output_path: str | Path,
    device: str,
) -> None:
    """
    Export model to TorchScript format.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to export.

    output_path : str | Path
        Output path for exported model.

    device : str
        Device model is on.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting model to TorchScript...")

    # Create dummy input
    dummy_input = torch.randn(1, 1, 28, 28, device=device)

    # Trace model
    traced_model = torch.jit.trace(model, dummy_input)

    # Save
    traced_model.save(str(output_path))

    logger.info(f"Model exported to: {output_path}")
    print(f"✓ TorchScript model exported to: {output_path}")


def export_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    device: str,
) -> None:
    """
    Export model to ONNX format.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to export.

    output_path : str | Path
        Output path for exported model.

    device : str
        Device model is on.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "ONNX export requires onnx package. "
            "Install with: pip install onnx"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting model to ONNX...")

    # Create dummy input
    dummy_input = torch.randn(1, 1, 28, 28, device=device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False,
    )

    logger.info(f"Model exported to: {output_path}")
    print(f"✓ ONNX model exported to: {output_path}")


def export_state_dict(
    model: torch.nn.Module,
    output_path: str | Path,
) -> None:
    """
    Export model state dictionary.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to export.

    output_path : str | Path
        Output path for exported model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting model state dictionary...")

    torch.save(model.state_dict(), output_path)

    logger.info(f"State dict exported to: {output_path}")
    print(f"✓ Model state dict exported to: {output_path}")


def export_full_model(
    model: torch.nn.Module,
    output_path: str | Path,
) -> None:
    """
    Export full model (including architecture).

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to export.

    output_path : str | Path
        Output path for exported model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting full model...")

    torch.save(model, output_path)

    logger.info(f"Full model exported to: {output_path}")
    print(f"✓ Full model exported to: {output_path}")


# ============================================================
# Export Pipeline
# ============================================================

def export(
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    export_format: str = "torchscript",
    device: Optional[str] = None,
) -> None:
    """
    Export trained model.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to model checkpoint.

    config_path : str | Path
        Path to configuration file.

    output_path : str | Path
        Output path for exported model.

    export_format : str
        Export format: 'torchscript', 'onnx', 'state_dict', or 'full'.

    device : Optional[str]
        Device to use. If None, auto-detects.
    """
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
    model.eval()

    logger.info(f"Model loaded: {model_name}")

    # ========================================================
    # Export
    # ========================================================

    export_format = export_format.lower()

    print("\n" + "=" * 60)
    print("MODEL EXPORT")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Format: {export_format}")
    print(f"Output: {output_path}")
    print("-" * 60)

    if export_format == "torchscript":
        export_torchscript(model, output_path, device)

    elif export_format == "onnx":
        export_onnx(model, output_path, device)

    elif export_format == "state_dict":
        export_state_dict(model, output_path)

    elif export_format == "full":
        export_full_model(model, output_path)

    else:
        raise ValueError(
            f"Unknown export format: {export_format}. "
            f"Supported formats: torchscript, onnx, state_dict, full"
        )

    print("=" * 60 + "\n")


# ============================================================
# CLI Argument Parsing
# ============================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export trained CNN model to different formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported export formats:
  - torchscript: TorchScript format for inference (default)
  - onnx: ONNX format for cross-framework compatibility
  - state_dict: Only model weights (PyTorch)
  - full: Full model with architecture

Examples:
  python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --format torchscript --output artifacts/models/model.pt
  python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --format onnx --output artifacts/models/model.onnx
  python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --format state_dict --output artifacts/models/weights.pt
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
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for exported model",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["torchscript", "onnx", "state_dict", "full"],
        default="torchscript",
        help="Export format (default: torchscript)",
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
        export(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            output_path=args.output,
            export_format=args.format,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise
