# Docker Setup for MNIST CNN Project

## Overview

This Docker setup provides a complete, reproducible environment for training and evaluating the MNIST CNN model.

## Quick Start

### 1. Build Docker Image

```bash
# Build from root directory
docker-compose build

# Or build specific service
docker build -f Docker/Dockerfile -t mnist-cnn:latest .
```

### 2. Run Container

```bash
# Using docker-compose (recommended)
docker-compose up -d mnist-training

# Or using Docker directly
docker run -it \
  --gpus all \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  -p 5000:5000 \
  -p 8000:8000 \
  mnist-cnn:latest
```

### 3. Access Services

- **Jupyter Lab**: http://localhost:8888
- **TensorBoard**: http://localhost:6006
- **API**: http://localhost:8000 (when running FastAPI)

## Available Services

### 1. Main Training Service

```bash
# Start training container
docker-compose up mnist-training

# Run training inside container
docker exec mnist-cnn-training python -m scripts.train \
  --config configs/experiments/baseline.yaml
```

### 2. TensorBoard

```bash
# Start TensorBoard service
docker-compose up tensorboard

# View at http://localhost:6006
```

## Common Commands Inside Container

### Training

```bash
# Quick test (debug config)
python -m scripts.train --config configs/experiments/debug.yaml

# Production (baseline config)
python -m scripts.train --config configs/experiments/baseline.yaml

# Best accuracy (large config)
python -m scripts.train --config configs/experiments/cnn_large.yaml
```

### Evaluation

```bash
python -m scripts.evaluate \
  --checkpoint artifacts/checkpoints/last.pt \
  --config configs/experiments/baseline.yaml
```

### Inference

```bash
python -m scripts.infer \
  --checkpoint artifacts/checkpoints/last.pt \
  --config configs/experiments/baseline.yaml \
  --image data/processed/X_test.npy
```

### Export Model

```bash
python -m scripts.export_model \
  --checkpoint artifacts/checkpoints/last.pt \
  --config configs/experiments/baseline.yaml \
  --format torchscript
```

### Benchmark

```bash
python -m scripts.benchmark \
  --checkpoint artifacts/checkpoints/last.pt \
  --config configs/experiments/baseline.yaml
```

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `.` | `/workspace` | Project root |
| `./data` | `/workspace/data` | Dataset |
| `./artifacts` | `/workspace/artifacts` | Outputs |
| `./notebooks` | `/workspace/notebooks` | Notebooks |

## GPU Support

### Enable GPU (NVIDIA)

```bash
# Docker Compose
GPU_SUPPORT=1 docker-compose up

# Docker CLI
docker run --gpus all -it mnist-cnn:latest
```

### Check GPU Inside Container

```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

## Environment Variables

```dockerfile
CUDA_VISIBLE_DEVICES=0       # GPU device selection
PYTHONUNBUFFERED=1           # Real-time output
DATA_DIR=/workspace/data     # Data directory
```

## Building Production Image

```bash
# Multi-stage build (if needed)
docker build \
  --target production \
  -f Docker/Dockerfile \
  -t mnist-cnn:prod \
  .

# Tag for registry
docker tag mnist-cnn:latest myregistry/mnist-cnn:1.0.0
docker push myregistry/mnist-cnn:1.0.0
```

## Docker Compose Operations

```bash
# Start all services
docker-compose up

# Start specific service
docker-compose up mnist-training

# Run command in container
docker-compose exec mnist-training bash

# View logs
docker-compose logs -f mnist-training

# Stop containers
docker-compose down

# Remove volumes
docker-compose down -v
```

## Dockerfile Details

### Base Image
- `pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04`
- Includes PyTorch 2.0 with CUDA 11.8 support

### Installed Packages
- **ML/Data**: torch, torchvision, numpy, pandas, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Notebooks**: jupyter, jupyterlab, ipython
- **APIs**: fastapi, uvicorn
- **Monitoring**: tensorboard
- **Export**: onnx

### Exposed Ports
- `8888`: Jupyter Lab
- `5000`: Flask/Custom API
- `8000`: FastAPI

## Troubleshooting

### GPU Not Detected

```bash
# Check if nvidia-docker is installed
nvidia-docker --version

# Run with GPU support
docker run --gpus all -it mnist-cnn:latest
```

### Permission Denied

```bash
# Use --allow-root flag in Jupyter
jupyter lab --allow-root

# Or run with proper user
docker run -u $(id -u):$(id -g) -it mnist-cnn:latest
```

### Port Already in Use

```bash
# Use different port mapping
docker run -p 8889:8888 mnist-cnn:latest

# Or stop existing containers
docker ps
docker stop <container_id>
```

### Data Not Accessible

```bash
# Verify volume mount
docker inspect <container_id> | grep Mounts

# Check data directory exists
ls -la ./data/processed/
```

## Development Workflow

### 1. Build Image
```bash
docker-compose build
```

### 2. Start Container
```bash
docker-compose up -d mnist-training
```

### 3. Work Inside Container
```bash
docker-compose exec mnist-training bash
cd /workspace
python -m scripts.train --config configs/experiments/debug.yaml
```

### 4. Monitor Output
```bash
docker-compose logs -f mnist-training
```

### 5. Clean Up
```bash
docker-compose down
```

## Production Deployment

### 1. Build Optimized Image
```bash
docker build -f Docker/Dockerfile \
  --build-arg OPTIMIZATION=1 \
  -t mnist-cnn:prod .
```

### 2. Use Gunicorn/uWSGI (if serving API)
```dockerfile
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "app:app"]
```

### 3. Push to Registry
```bash
docker tag mnist-cnn:prod registry.example.com/mnist-cnn:1.0
docker push registry.example.com/mnist-cnn:1.0
```

## References

- [PyTorch Docker Hub](https://hub.docker.com/r/pytorch/pytorch)
- [Docker Documentation](https://docs.docker.com)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

---

**Last Updated**: 2026-06-14  
**Version**: 1.0.0
