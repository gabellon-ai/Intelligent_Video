"""
Model Optimizer - TensorRT compilation with torch.compile fallback
Optimizes OWLv2 for high-performance inference on NVIDIA GPUs
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple, Any
import logging

import torch
from transformers import Owlv2ForObjectDetection

logger = logging.getLogger(__name__)


class OptimizationResult:
    """Results from model optimization attempt"""
    def __init__(
        self,
        model: Any,
        backend: str,
        optimization_time: float,
        cached: bool = False,
        speedup_estimate: Optional[float] = None
    ):
        self.model = model
        self.backend = backend  # "tensorrt", "torch_compile", "torch_inductor", "eager"
        self.optimization_time = optimization_time
        self.cached = cached
        self.speedup_estimate = speedup_estimate


class ModelOptimizer:
    """
    Handles model optimization for inference acceleration.
    
    Priority order:
    1. TensorRT (via torch_tensorrt) - best performance, ~10x speedup
    2. torch.compile with inductor backend - good performance, ~2-3x speedup
    3. Eager mode - baseline performance
    """
    
    CACHE_DIR = Path("models/optimized")
    METADATA_FILE = "optimization_meta.json"
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.cache_dir = self.CACHE_DIR / self._model_hash()
        self.tensorrt_available = self._check_tensorrt()
        self.torch_compile_available = self._check_torch_compile()
        
    def _model_hash(self) -> str:
        """Generate unique hash for model configuration"""
        config_str = f"{self.model_name}_{torch.__version__}_{self.device}"
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import torch_tensorrt
            # Verify TensorRT can actually be used
            torch_tensorrt.runtime.get_cudnn_enabled()
            logger.info("TensorRT available via torch_tensorrt")
            return True
        except ImportError:
            logger.info("torch_tensorrt not installed")
            return False
        except Exception as e:
            logger.warning(f"TensorRT check failed: {e}")
            return False
    
    def _check_torch_compile(self) -> bool:
        """Check if torch.compile is available"""
        try:
            if hasattr(torch, 'compile'):
                # Test on small tensor
                def dummy(x): return x * 2
                compiled = torch.compile(dummy)
                test = torch.tensor([1.0])
                _ = compiled(test)
                logger.info("torch.compile available")
                return True
        except Exception as e:
            logger.warning(f"torch.compile check failed: {e}")
        return False
    
    def _get_cache_path(self, backend: str) -> Path:
        """Get cache path for specific backend"""
        return self.cache_dir / f"model_{backend}.pt"
    
    def _load_metadata(self) -> dict:
        """Load optimization metadata"""
        meta_path = self.cache_dir / self.METADATA_FILE
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_metadata(self, metadata: dict):
        """Save optimization metadata"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.cache_dir / self.METADATA_FILE
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def optimize(
        self,
        model: Owlv2ForObjectDetection,
        sample_inputs: Optional[dict] = None,
        force_recompile: bool = False
    ) -> OptimizationResult:
        """
        Optimize model for inference.
        
        Args:
            model: The OWLv2 model to optimize
            sample_inputs: Example inputs for tracing (required for TensorRT)
            force_recompile: Force recompilation even if cached
            
        Returns:
            OptimizationResult with optimized model and metadata
        """
        model = model.to(self.device).eval()
        
        # Check for cached optimization
        if not force_recompile:
            cached_result = self._load_cached_model(model)
            if cached_result:
                return cached_result
        
        # Try optimization backends in priority order
        if self.tensorrt_available and sample_inputs is not None:
            result = self._optimize_tensorrt(model, sample_inputs)
            if result:
                self._cache_optimization(result)
                return result
        
        if self.torch_compile_available:
            result = self._optimize_torch_compile(model)
            if result:
                self._cache_optimization(result)
                return result
        
        # Fallback to eager mode
        logger.info("Using eager mode (no optimization)")
        return OptimizationResult(
            model=model,
            backend="eager",
            optimization_time=0.0,
            speedup_estimate=1.0
        )
    
    def _optimize_tensorrt(
        self,
        model: Owlv2ForObjectDetection,
        sample_inputs: dict
    ) -> Optional[OptimizationResult]:
        """Optimize with TensorRT"""
        try:
            import torch_tensorrt
            
            logger.info("Starting TensorRT optimization...")
            start_time = time.time()
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in sample_inputs.items()}
            
            # Configure TensorRT compilation
            # Use dynamic shapes for flexible batch sizes
            dynamic_batch = torch.export.Dim("batch", min=1, max=16)
            
            # Create input specs based on sample inputs
            dynamic_shapes = {}
            for key, tensor in inputs.items():
                if tensor.dim() >= 1:
                    dims = [dynamic_batch] + [None] * (tensor.dim() - 1)
                    dynamic_shapes[key] = tuple(dims)
            
            # Export model for TensorRT
            with torch.no_grad():
                # Trace the model
                exported = torch.export.export(
                    model,
                    (),  # No positional args
                    kwargs=inputs,
                    dynamic_shapes={"kwargs": dynamic_shapes} if dynamic_shapes else None
                )
            
            # Compile with TensorRT
            trt_model = torch_tensorrt.dynamo.compile(
                exported,
                inputs=list(inputs.values()),
                enabled_precisions={torch.float16, torch.float32},
                truncate_double=True,
                device=torch_tensorrt.Device(self.device),
                min_block_size=5,
                optimization_level=3,
            )
            
            optimization_time = time.time() - start_time
            logger.info(f"TensorRT optimization completed in {optimization_time:.2f}s")
            
            return OptimizationResult(
                model=trt_model,
                backend="tensorrt",
                optimization_time=optimization_time,
                speedup_estimate=10.0  # Conservative estimate
            )
            
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
            return None
    
    def _optimize_torch_compile(
        self,
        model: Owlv2ForObjectDetection
    ) -> Optional[OptimizationResult]:
        """Optimize with torch.compile"""
        try:
            logger.info("Starting torch.compile optimization...")
            start_time = time.time()
            
            # Try inductor backend first (best performance)
            try:
                compiled_model = torch.compile(
                    model,
                    backend="inductor",
                    mode="max-autotune",  # Best performance, longer compile
                    fullgraph=False,  # Allow graph breaks for complex models
                )
                backend = "torch_inductor"
                logger.info("Using inductor backend with max-autotune")
            except Exception as e:
                logger.info(f"Inductor failed ({e}), trying reduce-overhead mode")
                compiled_model = torch.compile(
                    model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                backend = "torch_compile"
            
            optimization_time = time.time() - start_time
            logger.info(f"torch.compile optimization completed in {optimization_time:.2f}s")
            
            return OptimizationResult(
                model=compiled_model,
                backend=backend,
                optimization_time=optimization_time,
                speedup_estimate=2.5  # Conservative estimate for torch.compile
            )
            
        except Exception as e:
            logger.warning(f"torch.compile optimization failed: {e}")
            return None
    
    def _load_cached_model(
        self,
        base_model: Owlv2ForObjectDetection
    ) -> Optional[OptimizationResult]:
        """Try to load a cached optimized model"""
        metadata = self._load_metadata()
        
        if not metadata:
            return None
        
        backend = metadata.get("backend")
        cache_path = self._get_cache_path(backend)
        
        if not cache_path.exists():
            return None
        
        try:
            # For torch.compile, we can't truly cache, but we can restore the compiled model
            if backend in ["torch_compile", "torch_inductor"]:
                logger.info(f"Reapplying {backend} optimization...")
                return self._optimize_torch_compile(base_model)
            
            # For TensorRT, try loading the cached engine
            elif backend == "tensorrt":
                import torch_tensorrt
                logger.info("Loading cached TensorRT model...")
                cached_model = torch.jit.load(cache_path)
                return OptimizationResult(
                    model=cached_model,
                    backend=backend,
                    optimization_time=metadata.get("optimization_time", 0),
                    cached=True,
                    speedup_estimate=metadata.get("speedup_estimate", 10.0)
                )
                
        except Exception as e:
            logger.warning(f"Failed to load cached model: {e}")
        
        return None
    
    def _cache_optimization(self, result: OptimizationResult):
        """Cache optimization metadata"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                "backend": result.backend,
                "optimization_time": result.optimization_time,
                "speedup_estimate": result.speedup_estimate,
                "torch_version": torch.__version__,
                "model_name": self.model_name,
                "cached_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # For TensorRT, save the compiled model
            if result.backend == "tensorrt":
                try:
                    cache_path = self._get_cache_path(result.backend)
                    torch.jit.save(result.model, cache_path)
                    metadata["cached_model"] = str(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to cache TensorRT model: {e}")
            
            self._save_metadata(metadata)
            logger.info(f"Optimization metadata cached to {self.cache_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to cache optimization: {e}")


def get_sample_inputs(processor, queries: list, batch_size: int = 1, device: str = "cuda") -> dict:
    """Generate sample inputs for optimization tracing"""
    from PIL import Image
    import numpy as np
    
    # Create dummy images
    dummy_images = [
        Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        for _ in range(batch_size)
    ]
    
    # Process inputs
    inputs = processor(
        text=[queries] * batch_size,
        images=dummy_images,
        return_tensors="pt",
        padding=True
    )
    
    return {k: v.to(device) for k, v in inputs.items()}
