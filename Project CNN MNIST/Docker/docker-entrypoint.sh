#!/bin/bash

# Docker entrypoint script for MNIST CNN project
# Handles initialization and service startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🚀 MNIST CNN Docker Container Starting${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check Python installation
echo -e "${YELLOW}✓ Python version:${NC} $(python --version)"

# Check PyTorch installation
echo -e "${YELLOW}✓ PyTorch version:${NC} $(python -c 'import torch; print(torch.__version__)')"

# Check GPU availability
GPU_AVAILABLE=$(python -c 'import torch; print("Available" if torch.cuda.is_available() else "Not available")')
echo -e "${YELLOW}✓ GPU status:${NC} $GPU_AVAILABLE"

# Display available commands
echo ""
echo -e "${GREEN}📋 Available Commands:${NC}"
echo "  Training:"
echo "    python -m scripts.train --config configs/experiments/baseline.yaml"
echo "    python -m scripts.train --config configs/experiments/debug.yaml"
echo "    python -m scripts.train --config configs/experiments/cnn_large.yaml"
echo ""
echo "  Evaluation:"
echo "    python -m scripts.evaluate --checkpoint artifacts/checkpoints/last.pt --config configs/experiments/baseline.yaml"
echo ""
echo "  Inference:"
echo "    python -m scripts.infer --checkpoint artifacts/checkpoints/last.pt --config configs/experiments/baseline.yaml --image data/processed/X_test.npy"
echo ""
echo "  Export:"
echo "    python -m scripts.export_model --checkpoint artifacts/checkpoints/last.pt --config configs/experiments/baseline.yaml --format torchscript"
echo ""
echo "  Benchmark:"
echo "    python -m scripts.benchmark --checkpoint artifacts/checkpoints/last.pt --config configs/experiments/baseline.yaml"
echo ""
echo "  Jupyter Lab:"
echo "    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""

# Check data directory
if [ -d "/workspace/data/processed" ]; then
    echo -e "${GREEN}✓ Data directory found${NC}"
    file_count=$(find /workspace/data/processed -name "*.npy" | wc -l)
    echo "  Files: $file_count .npy files"
else
    echo -e "${YELLOW}⚠ Data directory not found${NC}"
    echo "  Please ensure data/processed/ contains training data"
fi

# Create necessary directories
mkdir -p /workspace/artifacts/{checkpoints,models,logs}
mkdir -p /workspace/notebooks/{experiments,reports,assets/{figures,tables}}

echo ""
echo -e "${GREEN}✓ Directory structure verified${NC}"

# Execute passed command or start Jupyter
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Starting Jupyter Lab...${NC}"
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
else
    echo -e "${YELLOW}Executing: $@${NC}"
    exec "$@"
fi
