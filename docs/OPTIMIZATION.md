# Model Optimization Guide

This guide covers the TensorRT and torch.compile optimizations available for the Intelligent Video project.

## Overview

The system supports multiple optimization backends for accelerating OWLv2 inference:

| Backend | Speedup | Requirements | Best For |
|---------|---------|--------------|----------|
| TensorRT | ~10x | NVIDIA GPU + TensorRT SDK | Production, DGX systems |
| torch.compile (inductor) | ~2-3x | NVIDIA/AMD GPU | Development, general use |
| Eager mode | 1x (baseline) | Any hardware | Debugging, CPU-only |

## Quick Start

### 1. Pre-optimize Model (Recommended)

Run the optimization script before starting the server:

```bash
# Optimize model with automatic backend detection
python scripts/optimize_model.py

# Force recompilation (ignore cache)
python scripts/optimize_model.py --force

# Optimize and run benchmark
python scripts/optimize_model.py --benchmark
```

### 2. Benchmark Performance

Compare eager vs optimized performance:

```bash
# Full comparison benchmark
python scripts/benchmark.py --compare

# Quick benchmark (optimized only)
python scripts/benchmark.py

# Custom parameters
python scripts/benchmark.py --compare --num-images 200 --batch-size 16
```

### 3. Docker Deployment

For GPU-optimized containers:

```bash
# Build with TensorRT support (DGX/GPU)
docker build -f backend/Dockerfile -t intelligent-video:gpu backend/

# Build CPU-only version
docker build -f backend/Dockerfile.cpu -t intelligent-video:cpu backend/
```

## Configuration

Environment variables control optimization behavior:

```bash
# Enable/disable optimization (default: true)
IV_ENABLE_OPTIMIZATION=true

# Backend preference: auto, tensorrt, torch_compile, eager
IV_OPTIMIZATION_BACKEND=auto

# TensorRT precision: fp32, fp16, int8
IV_TENSORRT_PRECISION=fp16

# torch.compile mode: default, reduce-overhead, max-autotune
IV_TORCH_COMPILE_MODE=max-autotune

# Warmup iterations on startup
IV_WARMUP_ITERATIONS=5
```

## Backend Details

### TensorRT (Best Performance)

TensorRT provides the highest performance through:
- Graph optimization and kernel fusion
- Mixed precision (FP16/INT8)
- Memory optimization

**Requirements:**
- NVIDIA GPU with Compute Capability 7.0+
- TensorRT SDK 8.6+
- torch-tensorrt 2.1+

**Installation:**
```bash
# Via NGC container (recommended)
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# Manual installation
pip install tensorrt torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
```

### torch.compile (Good Performance)

torch.compile with the inductor backend provides:
- Automatic graph compilation
- Kernel fusion via Triton
- No additional dependencies on Linux

**Requirements:**
- PyTorch 2.0+
- Triton (auto-installed on Linux)

**Installation:**
```bash
pip install triton  # Linux only
```

### Eager Mode (Baseline)

Standard PyTorch execution without optimization. Used as:
- Fallback when other backends unavailable
- Debugging reference
- CPU-only deployments

## API Endpoints

### Health Check with Optimization Status

```bash
curl http://localhost:8000/api/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "optimization": {
    "backend": "torch_inductor",
    "speedup_estimate": 2.5,
    "cached": false
  }
}
```

### Detailed Optimization Status

```bash
curl http://localhost:8000/api/optimization
```

Response:
```json
{
  "model_name": "google/owlv2-base-patch16-ensemble",
  "device": "cuda",
  "gpu_available": true,
  "optimization": {
    "backend": "torch_inductor",
    "optimization_time": 45.2,
    "speedup_estimate": 2.5,
    "cached": false
  },
  "settings": {
    "enable_optimization": true,
    "backend_preference": "auto",
    "tensorrt_precision": "fp16",
    "torch_compile_mode": "max-autotune"
  }
}
```

## Troubleshooting

### TensorRT not detected

```bash
# Check TensorRT installation
python -c "import tensorrt; print(tensorrt.__version__)"
python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
```

### torch.compile fails

```bash
# Check triton installation
python -c "import triton; print(triton.__version__)"

# Try reduce-overhead mode
IV_TORCH_COMPILE_MODE=reduce-overhead
```

### Out of memory during optimization

- Reduce batch size in config
- Use FP16 precision
- Clear cache: `rm -rf models/optimized/`

## Performance Tips

1. **Pre-warm the model**: Run optimization script before serving
2. **Use appropriate batch sizes**: Larger batches = better GPU utilization
3. **Enable FP16**: Use `IV_TENSORRT_PRECISION=fp16` for 2x memory savings
4. **Cache optimizations**: First run is slow, subsequent runs load from cache

## Expected Performance

On NVIDIA DGX Spark (GB10):

| Backend | Throughput | Latency (batch=8) |
|---------|------------|-------------------|
| Eager | ~5 fps | ~1600ms |
| torch.compile | ~15 fps | ~533ms |
| TensorRT FP16 | ~50 fps | ~160ms |

*Results vary based on image size, batch size, and GPU.*
