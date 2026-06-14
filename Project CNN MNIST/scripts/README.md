# CNN MNIST Scripts Documentation

Complete scripts for training, evaluating, and deploying CNN MNIST models.

## Overview

This directory contains production-ready scripts for the CNN MNIST project:

- **train.py** - Train model with configuration-based setup
- **evaluate.py** - Evaluate model on test/validation sets
- **infer.py** - Run inference on single images or batches
- **export_model.py** - Export models to different formats (TorchScript, ONNX, etc.)
- **benchmark.py** - Benchmark model performance

All scripts support:
- Configuration file loading (YAML)
- Device selection (CPU/CUDA)
- Comprehensive logging
- Error handling with informative messages

## Installation & Setup

### Prerequisites

```bash
# Navigate to project root
cd "D:\Nam 3\Neural Network\Project CNN MNIST"

# Install dependencies
pip install -r requirements.txt
```

### Directory Structure

```
scripts/
├── __init__.py
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── infer.py                 # Inference script
├── export_model.py          # Model export script
├── benchmark.py             # Performance benchmarking
└── README.md                # This file
```

---

## Scripts Guide

### 1. Training (train.py)

Train a new model or continue training from a checkpoint.

#### Usage

```bash
# Train with baseline configuration
python -m scripts.train --config configs/experiments/baseline.yaml

# Train with specific device
python -m scripts.train --config configs/experiments/baseline.yaml --device cuda

# Train with specific experiment config
python -m scripts.train --config configs/experiments/cnn_small.yaml

# Train with CPU
python -m scripts.train --config configs/experiments/baseline.yaml --device cpu

# Debug logging
python -m scripts.train --config configs/experiments/baseline.yaml --log-level DEBUG
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `configs/experiments/baseline.yaml` | Path to configuration file |
| `--device` | str | auto-detect | Device: 'cuda' or 'cpu' |
| `--log-level` | str | INFO | Logging level: DEBUG, INFO, WARNING, ERROR |
| `-h, --help` | | | Show help message |

#### Configuration File Structure

```yaml
experiment:
  name: baseline

data:
  data_dir: data                    # Data directory path
  batch_size: 64                    # Batch size
  num_workers: 0                    # Number of data loading workers

model:
  name: mnist_cnn                   # Model name (registered in factory)
  num_classes: 10                   # Number of output classes
  hidden_features: 128              # Hidden layer dimension

training:
  epochs: 20                        # Number of training epochs
  loss: cross_entropy               # Loss function name
  patience: 5                       # Early stopping patience

optimizer:
  name: adam                        # Optimizer name
  lr: 0.001                         # Learning rate
  weight_decay: 0.0                 # L2 regularization

scheduler:
  name: none                        # Learning rate scheduler (or 'none')

artifacts:
  checkpoint_dir: artifacts/checkpoints    # Checkpoint directory
```

#### Example Output

```
=============================================================
Device: cuda
=============================================================
Loading config from: configs/experiments/baseline.yaml
Using device: cuda
Loading data...
Train loader: 847 batches, Valid loader: 212 batches
Creating model...
Model: mnist_cnn (Parameters: 155,914)
Loss function: cross_entropy
Optimizer: Adam
Scheduler: None
Checkpoint dir: artifacts/checkpoints
Early stopping patience: 5
Starting training for 20 epochs...

[Training progresses...]

============================================================
TRAINING SUMMARY
============================================================
Model: mnist_cnn
Device: cuda
Epochs: 20
Batch size: 64
Best val loss: 0.0234
Best val accuracy: 0.9945
============================================================
```

---

### 2. Evaluation (evaluate.py)

Evaluate a trained model on test or validation dataset.

#### Usage

```bash
# Evaluate on test set (default)
python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt

# Evaluate with specific config
python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --config configs/experiments/baseline.yaml

# Evaluate on validation set
python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --split valid

# Evaluate on GPU
python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --device cuda

# Verbose logging
python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --log-level DEBUG
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | **required** | Path to checkpoint file |
| `--config` | str | `configs/experiments/baseline.yaml` | Path to configuration file |
| `--split` | str | test | Dataset split: 'test' or 'valid' |
| `--device` | str | auto-detect | Device: 'cuda' or 'cpu' |
| `--log-level` | str | INFO | Logging level |

#### Output Example

```
============================================================
EVALUATION RESULTS (TEST SET)
============================================================
Model: mnist_cnn
Device: cuda
Checkpoint: artifacts/checkpoints/best_model.pt
------------------------------------------------------------
loss                           0.0234
accuracy                       0.9945
precision                      0.9947
recall                         0.9945
f1                             0.9945
============================================================
```

---

### 3. Inference (infer.py)

Run inference on images to get predictions.

#### Usage

```bash
# Single image inference
python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --image path/to/image.png

# Inference from NPY file
python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --image path/to/image.npy

# Batch inference from directory
python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --image-dir data/raw/mnist

# Batch inference on GPU
python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --image-dir data/raw/mnist --device cuda

# Image from test set
python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --image data/processed/X_test.npy
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | **required** | Path to checkpoint file |
| `--config` | str | `configs/experiments/baseline.yaml` | Path to configuration file |
| `--image` | str | None | Path to single image (PNG, JPG, NPY) |
| `--image-dir` | str | None | Path to directory with multiple images |
| `--device` | str | auto-detect | Device: 'cuda' or 'cpu' |
| `--log-level` | str | INFO | Logging level |

**Note:** Provide either `--image` or `--image-dir`, not both.

#### Supported Image Formats

- PNG files (.png)
- JPEG files (.jpg, .jpeg)
- NumPy arrays (.npy)

#### Output Example (Single Image)

```
============================================================
INFERENCE RESULTS
============================================================
Image: path/to/image.png
Predicted class: 7
Confidence: 0.9987

Top-K Predictions:
  1. Class 7: 0.9987
  2. Class 1: 0.0008
  3. Class 3: 0.0003
  4. Class 8: 0.0002
  5. Class 2: 0.0000
============================================================
```

#### Output Example (Batch)

```
============================================================
BATCH INFERENCE RESULTS (1234 images)
============================================================
image_001.png                  Class: 7     Confidence: 0.9987
image_002.png                  Class: 3     Confidence: 0.9542
image_003.png                  Class: 2     Confidence: 0.8876
...
============================================================
```

---

### 4. Export Model (export_model.py)

Export trained model to different formats for deployment.

#### Usage

```bash
# Export to TorchScript (default)
python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --output artifacts/models/model.pt

# Export to ONNX format
python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --format onnx --output artifacts/models/model.onnx

# Export model state dictionary
python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --format state_dict --output artifacts/models/weights.pt

# Export full model
python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --format full --output artifacts/models/full_model.pt

# Export with GPU
python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --device cuda --output artifacts/models/model.pt
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | **required** | Path to checkpoint file |
| `--config` | str | `configs/experiments/baseline.yaml` | Path to configuration file |
| `--output` | str | **required** | Output path for exported model |
| `--format` | str | torchscript | Export format |
| `--device` | str | auto-detect | Device: 'cuda' or 'cpu' |

#### Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| `torchscript` | .pt | Production inference in Python |
| `onnx` | .onnx | Cross-framework deployment (TensorFlow, etc.) |
| `state_dict` | .pt | Transfer learning, fine-tuning |
| `full` | .pt | Complete model with architecture |

#### Output Example

```
============================================================
MODEL EXPORT
============================================================
Checkpoint: artifacts/checkpoints/best_model.pt
Model: mnist_cnn
Format: torchscript
Output: artifacts/models/model.pt
------------------------------------------------------------
✓ TorchScript model exported to: artifacts/models/model.pt
============================================================
```

---

### 5. Benchmark (benchmark.py)

Measure model performance (latency, throughput, memory).

#### Usage

```bash
# Benchmark with default batch sizes [1, 32, 64]
python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt

# Custom batch sizes
python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --batch-sizes 1 32 64 128 256

# Increase iterations for more accurate results
python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --num-iterations 500

# Benchmark on GPU
python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --device cuda

# Benchmark on CPU
python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --device cpu
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | **required** | Path to checkpoint file |
| `--config` | str | `configs/experiments/baseline.yaml` | Path to configuration file |
| `--batch-sizes` | int | 1 32 64 | Batch sizes to benchmark |
| `--num-iterations` | int | 100 | Number of iterations per batch size |
| `--device` | str | auto-detect | Device: 'cuda' or 'cpu' |

#### Output Example

```
================================================================================
MODEL BENCHMARK RESULTS
================================================================================

Model: mnist_cnn
Device: cuda
Checkpoint: artifacts/checkpoints/best_model.pt

--------------------------------------------------------------------------------
COMPLEXITY METRICS
--------------------------------------------------------------------------------
Total Parameters:                  155,914
Trainable Parameters:              155,914

--------------------------------------------------------------------------------
MEMORY METRICS
--------------------------------------------------------------------------------
Model Size:              1.50 MB
Peak Memory:             0.87 MB

--------------------------------------------------------------------------------
INFERENCE LATENCY & THROUGHPUT
--------------------------------------------------------------------------------
Batch Size      Avg Latency (ms)  Throughput (samples/sec)
--------------------------------------------------------------------------------
1               2.3450            427.43
32              15.6234            2047.21
64              28.9123            2213.87
================================================================================
```

---

## Workflow Examples

### Complete Training Pipeline

```bash
# 1. Train model
python -m scripts.train --config configs/experiments/baseline.yaml --device cuda

# 2. Evaluate on test set
python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --split test

# 3. Evaluate on validation set
python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --split valid

# 4. Run inference on sample images
python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --image-dir data/raw/mnist

# 5. Benchmark model
python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --batch-sizes 1 32 64

# 6. Export for deployment
python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --format torchscript --output artifacts/models/model.pt
```

### Quick Experiment Iteration

```bash
# Try different configuration
python -m scripts.train --config configs/experiments/cnn_small.yaml

# Quick evaluation
python -m scripts.evaluate --checkpoint artifacts/checkpoints/best_model.pt --split valid

# Verify inference works
python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --image data/processed/X_test.npy
```

### Deployment Preparation

```bash
# Export model for production
python -m scripts.export_model --checkpoint artifacts/checkpoints/best_model.pt --format onnx --output artifacts/models/model.onnx

# Verify export quality
python -m scripts.infer --checkpoint artifacts/checkpoints/best_model.pt --image-dir data/raw/mnist

# Performance baseline
python -m scripts.benchmark --checkpoint artifacts/checkpoints/best_model.pt --num-iterations 500
```

---

## Configuration Examples

### Small Model (Fast Training)

File: `configs/experiments/cnn_small.yaml`

```yaml
experiment:
  name: cnn_small

data:
  data_dir: data
  batch_size: 128
  num_workers: 4

model:
  name: mnist_cnn
  num_classes: 10
  hidden_features: 64

training:
  epochs: 10
  loss: cross_entropy
  patience: 3

optimizer:
  name: adam
  lr: 0.001
  weight_decay: 0.0

scheduler:
  name: none

artifacts:
  checkpoint_dir: artifacts/checkpoints
```

### Large Model (Better Accuracy)

File: `configs/experiments/cnn_large.yaml`

```yaml
experiment:
  name: cnn_large

data:
  data_dir: data
  batch_size: 32
  num_workers: 4

model:
  name: mnist_cnn
  num_classes: 10
  hidden_features: 256

training:
  epochs: 30
  loss: cross_entropy
  patience: 10

optimizer:
  name: adam
  lr: 0.0005
  weight_decay: 1e-5

scheduler:
  name: none

artifacts:
  checkpoint_dir: artifacts/checkpoints
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Use smaller batch size
python -m scripts.train --config configs/experiments/baseline.yaml
# Modify batch_size in config file to lower value

# Or switch to CPU
python -m scripts.train --config configs/experiments/baseline.yaml --device cpu
```

### Checkpoint Not Found

```
FileNotFoundError: Checkpoint not found: artifacts/checkpoints/best_model.pt
```

Solution: Ensure the checkpoint path is correct and the training script has completed successfully.

```bash
# Check available checkpoints
ls artifacts/checkpoints/
```

### Data Directory Not Found

```
FileNotFoundError: Data directory not found: data
```

Solution: Ensure data preprocessing has been completed. Run data preparation notebooks first.

### Import Errors

```
ModuleNotFoundError: No module named 'src'
```

Solution: Run scripts from project root directory:

```bash
cd "D:\Nam 3\Neural Network\Project CNN MNIST"
python -m scripts.train --config configs/experiments/baseline.yaml
```

---

## Tips & Best Practices

1. **Always use project root as working directory**
   ```bash
   cd "D:\Nam 3\Neural Network\Project CNN MNIST"
   python -m scripts.train ...
   ```

2. **Use appropriate device for your hardware**
   - CUDA (GPU) for large batches and faster training
   - CPU for development and debugging

3. **Monitor memory usage**
   - Reduce batch size if running out of memory
   - Use benchmarking to understand model requirements

4. **Version your experiments**
   - Create new config files for different experiments
   - Use meaningful config names (e.g., `cnn_small.yaml`, `cnn_large.yaml`)

5. **Save multiple checkpoints**
   - Keep best model, latest model, and milestone checkpoints
   - Use evaluation script to pick the best one

6. **Profile before deployment**
   - Always run benchmark on target hardware
   - Export to target format (TorchScript, ONNX)

---

## Performance Summary

Typical performance on single NVIDIA GPU (RTX 3090):

| Metric | Value |
|--------|-------|
| Training speed | ~2000 samples/sec |
| Inference latency (BS=1) | ~2-3ms |
| Inference throughput (BS=64) | ~2000-2500 samples/sec |
| Model size | ~1.5 MB |
| Peak memory | ~800 MB |

---

## Support & Documentation

For more information:
- See `README.md` in project root
- Check configuration files in `configs/`
- Review source code in `src/`
- Refer to design documents in `docs/`

---

Last Updated: 2026-06-14
