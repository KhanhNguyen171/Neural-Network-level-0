# Base image with Python 3.9 and PyTorch support
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    torch==2.0.0 \
    torchvision==0.15.0 \
    pandas==1.5.3 \
    matplotlib==3.7.0 \
    seaborn==0.12.2 \
    scikit-learn==1.2.2 \
    pyyaml==6.0 \
    jupyter==1.0.0 \
    jupyterlab==3.6.0 \
    ipython==8.12.0 \
    scipy==1.10.0 \
    pillow==9.5.0 \
    tqdm==4.65.0 \
    tensorboard==2.12.0 \
    onnx==1.13.1 \
    fastapi==0.95.1 \
    uvicorn==0.21.0 \
    pydantic==1.10.7

# Copy project files
COPY . /workspace

# Create necessary directories
RUN mkdir -p /workspace/artifacts/checkpoints \
    && mkdir -p /workspace/artifacts/models \
    && mkdir -p /workspace/artifacts/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV DATA_DIR=/workspace/data

# Expose ports
EXPOSE 8888 5000 8000

# Default command - Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
