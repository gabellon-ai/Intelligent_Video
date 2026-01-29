#!/usr/bin/env python3
"""
Inference Benchmark Script

Measures and compares inference performance across different optimization levels:
- Eager mode (baseline PyTorch)
- torch.compile optimized
- TensorRT optimized (if available)

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --num-images 200 --batch-size 16
    python scripts/benchmark.py --compare  # Compare all backends
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import torch
import numpy as np
from PIL import Image


def create_test_images(num_images: int, image_size: tuple = (640, 640)) -> list:
    """Generate random test images"""
    return [
        Image.fromarray(np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8))
        for _ in range(num_images)
    ]


def benchmark_detector(
    detector,
    images: list,
    batch_size: int,
    queries: list,
    threshold: float,
    num_warmup: int = 5
) -> dict:
    """Run benchmark on a detector instance"""
    
    # Warmup
    print(f"  Warming up ({num_warmup} iterations)...")
    detector.warmup(num_iterations=num_warmup)
    
    # Run timed inference
    times = []
    total_images = 0
    
    print(f"  Running inference on {len(images)} images (batch size {batch_size})...")
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        if detector.device == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        _ = detector._detect_batch_sync(batch, queries, threshold)
        
        if detector.device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed / len(batch))  # Time per image
        total_images += len(batch)
    
    # Calculate statistics
    times_ms = np.array(times) * 1000
    
    return {
        "total_images": total_images,
        "total_time_s": sum(times),
        "throughput_fps": total_images / sum(times),
        "latency_ms": {
            "mean": float(np.mean(times_ms)),
            "std": float(np.std(times_ms)),
            "min": float(np.min(times_ms)),
            "max": float(np.max(times_ms)),
            "p50": float(np.percentile(times_ms, 50)),
            "p95": float(np.percentile(times_ms, 95)),
            "p99": float(np.percentile(times_ms, 99)),
        },
        "batch_size": batch_size,
    }


def run_comparison_benchmark(args):
    """Run benchmark comparing eager vs optimized"""
    from app.services.detector import DetectorService
    from app.config import settings
    
    print("=" * 70)
    print("INFERENCE BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"Test parameters:")
    print(f"  Images: {args.num_images}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print()
    
    # System info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    
    # Generate test images
    print("Generating test images...")
    images = create_test_images(args.num_images, (args.image_size, args.image_size))
    
    results = {}
    
    # Benchmark eager mode (baseline)
    print("\n" + "-" * 70)
    print("1. EAGER MODE (Baseline)")
    print("-" * 70)
    
    detector_eager = DetectorService()
    import asyncio
    asyncio.run(detector_eager.load_model(optimize=False))
    
    results["eager"] = benchmark_detector(
        detector_eager,
        images,
        args.batch_size,
        settings.DEFAULT_QUERIES,
        settings.CONFIDENCE_THRESHOLD
    )
    results["eager"]["backend"] = "eager"
    
    print(f"  Throughput: {results['eager']['throughput_fps']:.1f} images/sec")
    print(f"  Latency (mean): {results['eager']['latency_ms']['mean']:.2f}ms")
    
    # Clean up
    del detector_eager
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Benchmark optimized mode
    print("\n" + "-" * 70)
    print("2. OPTIMIZED MODE (torch.compile / TensorRT)")
    print("-" * 70)
    
    detector_optimized = DetectorService()
    asyncio.run(detector_optimized.load_model(optimize=True))
    
    results["optimized"] = benchmark_detector(
        detector_optimized,
        images,
        args.batch_size,
        settings.DEFAULT_QUERIES,
        settings.CONFIDENCE_THRESHOLD
    )
    results["optimized"]["backend"] = detector_optimized.optimization_info.get("backend", "unknown")
    
    print(f"  Backend: {results['optimized']['backend']}")
    print(f"  Throughput: {results['optimized']['throughput_fps']:.1f} images/sec")
    print(f"  Latency (mean): {results['optimized']['latency_ms']['mean']:.2f}ms")
    
    # Calculate speedup
    speedup = results["optimized"]["throughput_fps"] / results["eager"]["throughput_fps"]
    latency_reduction = (1 - results["optimized"]["latency_ms"]["mean"] / results["eager"]["latency_ms"]["mean"]) * 100
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Eager':>15} {'Optimized':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Throughput (fps)':<25} {results['eager']['throughput_fps']:>15.1f} {results['optimized']['throughput_fps']:>15.1f} {speedup:>14.1f}x")
    print(f"{'Latency mean (ms)':<25} {results['eager']['latency_ms']['mean']:>15.2f} {results['optimized']['latency_ms']['mean']:>15.2f} {latency_reduction:>14.1f}%")
    print(f"{'Latency p95 (ms)':<25} {results['eager']['latency_ms']['p95']:>15.2f} {results['optimized']['latency_ms']['p95']:>15.2f}")
    print(f"{'Latency p99 (ms)':<25} {results['eager']['latency_ms']['p99']:>15.2f} {results['optimized']['latency_ms']['p99']:>15.2f}")
    print("=" * 70)
    print(f"\n🚀 Overall Speedup: {speedup:.1f}x")
    print(f"⚡ Latency Reduction: {latency_reduction:.1f}%")
    
    # Save results
    results["speedup"] = speedup
    results["latency_reduction_pct"] = latency_reduction
    results["timestamp"] = datetime.now().isoformat()
    results["config"] = {
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
    }
    
    output_file = Path("output") / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def run_single_benchmark(args):
    """Run benchmark on optimized model only"""
    from app.services.detector import DetectorService
    from app.config import settings
    
    print("=" * 70)
    print("INFERENCE BENCHMARK")
    print("=" * 70)
    
    # System info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    # Load model
    print("Loading optimized model...")
    detector = DetectorService()
    import asyncio
    asyncio.run(detector.load_model(optimize=True))
    
    opt_info = detector.optimization_info
    print(f"Optimization backend: {opt_info.get('backend', 'unknown')}")
    print()
    
    # Generate test images
    print(f"Generating {args.num_images} test images...")
    images = create_test_images(args.num_images, (args.image_size, args.image_size))
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = benchmark_detector(
        detector,
        images,
        args.batch_size,
        settings.DEFAULT_QUERIES,
        settings.CONFIDENCE_THRESHOLD
    )
    
    # Print results
    print("\n" + "-" * 50)
    print("Results:")
    print(f"  Throughput: {results['throughput_fps']:.1f} images/sec")
    print(f"  Latency (mean): {results['latency_ms']['mean']:.2f}ms")
    print(f"  Latency (p95): {results['latency_ms']['p95']:.2f}ms")
    print(f"  Latency (p99): {results['latency_ms']['p99']:.2f}ms")
    print("-" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark OWLv2 inference")
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare eager vs optimized performance"
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=100,
        help="Number of images to benchmark (default: 100)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--image-size", "-s",
        type=int,
        default=640,
        help="Image size (default: 640)"
    )
    args = parser.parse_args()
    
    if args.compare:
        run_comparison_benchmark(args)
    else:
        run_single_benchmark(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
