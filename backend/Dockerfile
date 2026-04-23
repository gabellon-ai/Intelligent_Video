# Backend Dockerfile for DGX Spark (aarch64 / Grace Blackwell GB10)
# Uses NVIDIA NGC PyTorch container — includes CUDA, cuDNN, TensorRT, torch-tensorrt
# pre-built for aarch64 with Blackwell support.

FROM nvcr.io/nvidia/pytorch:25.08-py3

# System deps for OpenCV / ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps. torch/torchvision/triton are already in the NGC image —
# pip will keep them if the pinned versions are satisfied.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p uploads output models/optimized

ENV IV_ENABLE_OPTIMIZATION=true
ENV IV_OPTIMIZATION_BACKEND=auto
ENV IV_TENSORRT_PRECISION=fp16

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
