#!/usr/bin/env python3
"""
Model Optimization Script - First-run TensorRT/torch.compile optimization

Run this script once after deployment to pre-compile the model.
Subsequent runs will use cached optimizations.

Usage:
    python scripts/optimize_model.py
    python scripts/optimize_model.py --force  # Force recompilation
    python scripts/optimize_model.py --benchmark  # Run benchmark after optimization
"""

import argparse
import sys
import time
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("optimize_model")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize OWLv2 model for inference"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force recompilation even if cached"
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run benchmark after optimization"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Load model without optimization (baseline)"
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=100,
        help="Number of images for benchmark (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for benchmark (default: 8)"
    )
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Optimization will be limited.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
    
    # Import after path setup
    from app.services.detector import DetectorService
    
    logger.info("=" * 60)
    logger.info("Starting model optimization")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Create detector service
    detector = DetectorService()
    
    # Load and optimize model
    import asyncio
    asyncio.run(detector.load_model(
        optimize=not args.no_optimize,
        force_recompile=args.force
    ))
    
    load_time = time.time() - start_time
    logger.info(f"Model load/optimization time: {load_time:.2f}s")
    
    # Show optimization results
    opt_info = detector.optimization_info
    logger.info("-" * 40)
    logger.info("Optimization Results:")
    logger.info(f"  Backend: {opt_info.get('backend', 'unknown')}")
    logger.info(f"  Optimization time: {opt_info.get('optimization_time', 0):.2f}s")
    logger.info(f"  Estimated speedup: {opt_info.get('speedup_estimate', 1.0)}x")
    logger.info(f"  Cached: {opt_info.get('cached', False)}")
    logger.info("-" * 40)
    
    # Warmup
    logger.info("Running warmup...")
    detector.warmup(num_iterations=5)
    
    # Run benchmark if requested
    if args.benchmark:
        logger.info("=" * 60)
        logger.info(f"Running benchmark: {args.num_images} images, batch size {args.batch_size}")
        logger.info("=" * 60)
        
        results = detector.benchmark(
            num_images=args.num_images,
            batch_size=args.batch_size
        )
        
        logger.info("-" * 40)
        logger.info("Benchmark Results:")
        logger.info(f"  Total images: {results['total_images']}")
        logger.info(f"  Total time: {results['total_time']:.2f}s")
        logger.info(f"  Throughput: {results['images_per_second']:.1f} images/sec")
        logger.info(f"  Avg batch latency: {results['avg_batch_latency']*1000:.1f}ms")
        logger.info(f"  Backend: {results['optimization_backend']}")
        logger.info(f"  Device: {results['device']}")
        logger.info("-" * 40)
    
    logger.info("Optimization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
