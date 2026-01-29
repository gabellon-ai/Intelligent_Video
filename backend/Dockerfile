# TensorRT-optimized backend Dockerfile
# For DGX Spark / NVIDIA GPUs with TensorRT support

FROM nvcr.io/nvidia/pytorch:24.01-py3

# This image includes:
# - PyTorch with CUDA support
# - TensorRT runtime and torch-tensorrt
# - CUDA toolkit

WORKDIR /app

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install torch-tensorrt for TensorRT optimization
# Note: The NGC PyTorch container already has this, but we ensure latest
RUN pip install --no-cache-dir torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121 || \
    echo "torch-tensorrt installation optional - will fall back to torch.compile"

# Copy application
COPY app/ ./app/

# Create directories
RUN mkdir -p uploads output models/optimized

# Pre-optimize model on first run (optional - can be done at runtime)
# RUN python -c "from app.services.detector import DetectorService; import asyncio; d = DetectorService(); asyncio.run(d.load_model())"

# Environment variables
ENV IV_ENABLE_OPTIMIZATION=true
ENV IV_OPTIMIZATION_BACKEND=auto
ENV IV_TENSORRT_PRECISION=fp16

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
