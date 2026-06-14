# scripts/infer.py
"""
Inference script for CNN MNIST model.

Usage:
    python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml --image <image_path>
    python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml --input-dir data/raw/mnist
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

import torch
import numpy as np
import yaml
from PIL import Image

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
        else:
            raise ValueError(
                f"Unknown device: {device_str}. Use 'cuda' or 'cpu'."
            )

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# Image Loading
# ============================================================

def load_image(image_path: str | Path) -> torch.Tensor:
    """
    Load and preprocess image.

    Parameters
    ----------
    image_path : str | Path
        Path to image file.

    Returns
    -------
    torch.Tensor
        Preprocessed image tensor (1, 1, 28, 28).
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    img = Image.open(image_path).convert('L')

    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to tensor and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)

    return img_tensor


def load_npy_image(npy_path: str | Path) -> torch.Tensor:
    """
    Load image from NPY file.

    Parameters
    ----------
    npy_path : str | Path
        Path to NPY file.

    Returns
    -------
    torch.Tensor
        Image tensor (1, 1, 28, 28) or (N, 1, 28, 28).
    """
    npy_path = Path(npy_path)

    if not npy_path.exists():
        raise FileNotFoundError(f"NPY file not found: {npy_path}")

    img_array = np.load(npy_path)

    # Ensure shape is (N, 28, 28) or (28, 28)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=0)

    # Convert to tensor and add channel dimension
    img_tensor = torch.from_numpy(img_array).float()

    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    elif img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(1)  # (N, 1, 28, 28)

    return img_tensor


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
# Inference Pipeline
# ============================================================

def infer_single(
    checkpoint_path: str | Path,
    config_path: str | Path,
    image_path: str | Path,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run inference on a single image.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to model checkpoint.

    config_path : str | Path
        Path to configuration file.

    image_path : str | Path
        Path to image file.

    device : Optional[str]
        Device to use. If None, auto-detects.

    Returns
    -------
    Dict[str, Any]
        Prediction results.
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

    # ========================================================
    # Load Image
    # ========================================================

    logger.info(f"Loading image from: {image_path}")

    # Determine file type
    image_path = Path(image_path)
    if image_path.suffix.lower() == '.npy':
        img_tensor = load_npy_image(image_path)
    else:
        img_tensor = load_image(image_path)

    img_tensor = img_tensor.to(device)

    # ========================================================
    # Inference
    # ========================================================

    logger.info("Running inference...")

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    # Get top-k predictions
    top_k = min(5, num_classes)
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

    # ========================================================
    # Format Results
    # ========================================================

    results = {
        "image": str(image_path),
        "prediction": predictions[0].item(),
        "confidence": probabilities[0, predictions[0]].item(),
        "top_k_predictions": [
            {
                "class": idx.item(),
                "probability": prob.item(),
            }
            for idx, prob in zip(top_indices[0], top_probs[0])
        ],
    }

    # Print results
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Predicted class: {results['prediction']}")
    print(f"Confidence: {results['confidence']:.4f}")
    print("\nTop-K Predictions:")
    for i, pred in enumerate(results["top_k_predictions"], 1):
        print(f"  {i}. Class {pred['class']}: {pred['probability']:.4f}")
    print("=" * 60 + "\n")

    return results


def infer_batch(
    checkpoint_path: str | Path,
    config_path: str | Path,
    input_dir: str | Path,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run inference on multiple images in a directory.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to model checkpoint.

    config_path : str | Path
        Path to configuration file.

    input_dir : str | Path
        Directory containing images.

    device : Optional[str]
        Device to use. If None, auto-detects.

    Returns
    -------
    List[Dict[str, Any]]
        List of prediction results.
    """
    # Get device
    device = get_device(device)

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

    # ========================================================
    # Find Image Files
    # ========================================================

    input_dir = Path(input_dir)
    image_files = []

    for ext in ['*.png', '*.jpg', '*.jpeg', '*.npy']:
        image_files.extend(input_dir.glob(f"**/{ext}"))
        image_files.extend(input_dir.glob(f"**/{ext.upper()}"))

    logger.info(f"Found {len(image_files)} images in {input_dir}")

    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return []

    # ========================================================
    # Batch Inference
    # ========================================================

    all_results = []

    for img_path in image_files:
        try:
            # Load image
            if img_path.suffix.lower() == '.npy':
                img_tensor = load_npy_image(img_path)
            else:
                img_tensor = load_image(img_path)

            img_tensor = img_tensor.to(device)

            # Inference
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

            result = {
                "image": str(img_path),
                "prediction": predictions[0].item(),
                "confidence": probabilities[0, predictions[0]].item(),
            }

            all_results.append(result)
            logger.info(
                f"Processed {img_path.name}: "
                f"Class {result['prediction']}, "
                f"Confidence {result['confidence']:.4f}"
            )

        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            continue

    # ========================================================
    # Print Summary
    # ========================================================

    print("\n" + "=" * 60)
    print(f"BATCH INFERENCE RESULTS ({len(all_results)} images)")
    print("=" * 60)

    for result in all_results:
        print(
            f"{Path(result['image']).name:<30} "
            f"Class: {result['prediction']:<3} "
            f"Confidence: {result['confidence']:.4f}"
        )

    print("=" * 60 + "\n")

    return all_results


# ============================================================
# CLI Argument Parsing
# ============================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained CNN model on MNIST dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml --image path/to/image.png
  python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml --image-dir data/raw/mnist
  python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml --image path/to/image.npy --device cuda
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
        "--image",
        type=str,
        default=None,
        help="Path to single image file (PNG, JPG, or NPY)",
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to directory containing multiple images",
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
        if args.image:
            infer_single(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                image_path=args.image,
                device=args.device,
            )
        elif args.image_dir:
            infer_batch(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                input_dir=args.image_dir,
                device=args.device,
            )
        else:
            print("Error: Please provide either --image or --image-dir")
            print("Use -h or --help for more information")

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise
