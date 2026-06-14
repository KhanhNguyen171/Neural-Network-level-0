"""
Scripts module for CNN MNIST project.

This module contains executable scripts for training, evaluation, inference, and benchmarking.

Available scripts:
- train.py: Train the model
- evaluate.py: Evaluate trained model
- infer.py: Run inference on images
- export_model.py: Export model to different formats
- benchmark.py: Benchmark model performance

Examples:
    python -m scripts.train --config configs/experiments/baseline.yaml
    python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt
    python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --image <path>
"""
