"""
Detection service - OWLv2 zero-shot object detection
Optimized with TensorRT/torch.compile for high-performance inference
"""

import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from ..config import settings
from .model_optimizer import ModelOptimizer, OptimizationResult, get_sample_inputs

logger = logging.getLogger(__name__)


class DetectorService:
    """
    OWLv2-based object detection service with automatic GPU optimization.
    
    Optimization priority:
    1. TensorRT (via torch_tensorrt) - ~10x speedup
    2. torch.compile with inductor - ~2-3x speedup  
    3. Eager mode - baseline
    """
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.optimized_model = None
        self.device = None
        self.is_loaded = False
        self.gpu_available = torch.cuda.is_available()
        self.optimization_result: Optional[OptimizationResult] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._warmup_done = False
    
    async def load_model(self, optimize: bool = True, force_recompile: bool = False):
        """Load model asynchronously with optional optimization"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor, 
            self._load_model_sync,
            optimize,
            force_recompile
        )
    
    def _load_model_sync(self, optimize: bool = True, force_recompile: bool = False):
        """Synchronous model loading with optimization"""
        logger.info(f"Loading OWLv2 model: {settings.MODEL_NAME}")
        
        self.processor = Owlv2Processor.from_pretrained(settings.MODEL_NAME)
        self.model = Owlv2ForObjectDetection.from_pretrained(settings.MODEL_NAME)
        
        if self.gpu_available:
            self.device = "cuda"
            self.model = self.model.to(self.device)
            self.model = self.model.eval()
            
            if optimize:
                self._optimize_model(force_recompile)
            else:
                logger.info("Optimization disabled, using eager mode")
                self.optimized_model = self.model
        else:
            self.device = "cpu"
            self.model = self.model.eval()
            self.optimized_model = self.model
            logger.info("GPU not available, using CPU")
        
        self.is_loaded = True
        logger.info("Model loaded successfully")
    
    def _optimize_model(self, force_recompile: bool = False):
        """Apply model optimization"""
        optimizer = ModelOptimizer(settings.MODEL_NAME, self.device)
        
        logger.info(f"TensorRT available: {optimizer.tensorrt_available}")
        logger.info(f"torch.compile available: {optimizer.torch_compile_available}")
        
        # Generate sample inputs for TensorRT tracing
        sample_inputs = None
        if optimizer.tensorrt_available:
            try:
                sample_inputs = get_sample_inputs(
                    self.processor,
                    settings.DEFAULT_QUERIES,
                    batch_size=1,
                    device=self.device
                )
            except Exception as e:
                logger.warning(f"Failed to generate sample inputs: {e}")
        
        # Run optimization
        self.optimization_result = optimizer.optimize(
            self.model,
            sample_inputs=sample_inputs,
            force_recompile=force_recompile
        )
        
        self.optimized_model = self.optimization_result.model
        
        logger.info(f"Optimization backend: {self.optimization_result.backend}")
        logger.info(f"Optimization time: {self.optimization_result.optimization_time:.2f}s")
        logger.info(f"Estimated speedup: {self.optimization_result.speedup_estimate}x")
    
    def warmup(self, num_iterations: int = 3):
        """Warmup the optimized model to ensure JIT compilation is complete"""
        if self._warmup_done:
            return
        
        logger.info(f"Warming up model with {num_iterations} iterations...")
        
        # Create dummy input
        from PIL import Image
        import numpy as np
        dummy_image = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        
        for i in range(num_iterations):
            try:
                inputs = self.processor(
                    text=[settings.DEFAULT_QUERIES],
                    images=[dummy_image],
                    return_tensors="pt",
                    padding=True
                )
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    _ = self.optimized_model(**inputs)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                logger.debug(f"Warmup iteration {i+1}/{num_iterations} complete")
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")
        
        self._warmup_done = True
        logger.info("Warmup complete")
    
    @property
    def optimization_info(self) -> Dict[str, Any]:
        """Get optimization status information"""
        if self.optimization_result:
            return {
                "backend": self.optimization_result.backend,
                "optimization_time": self.optimization_result.optimization_time,
                "speedup_estimate": self.optimization_result.speedup_estimate,
                "cached": self.optimization_result.cached
            }
        return {"backend": "not_optimized"}
    
    async def detect_batch(
        self, 
        images: List[Image.Image], 
        queries: List[str] = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Run detection on a batch of images
        Returns list of detection results per image
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        queries = queries or settings.DEFAULT_QUERIES
        threshold = threshold or settings.CONFIDENCE_THRESHOLD
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._detect_batch_sync,
            images, queries, threshold
        )
    
    def _detect_batch_sync(
        self,
        images: List[Image.Image],
        queries: List[str],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Synchronous batch detection using optimized model"""
        
        # Ensure warmup is done
        if not self._warmup_done and self.device == "cuda":
            self.warmup()
        
        # Process all images in batch
        inputs = self.processor(
            text=[queries] * len(images),
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Use optimized model for inference
        with torch.no_grad():
            if self.device == "cuda":
                # Use CUDA events for accurate timing (optional debug)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
            outputs = self.optimized_model(**inputs)
            
            if self.device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                inference_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
                logger.debug(f"Batch inference time: {inference_time:.4f}s for {len(images)} images")
        
        # Post-process each image
        results = []
        for i, image in enumerate(images):
            target_sizes = torch.tensor([[image.size[1], image.size[0]]])
            if self.device == "cuda":
                target_sizes = target_sizes.to(self.device)
            
            # Get outputs for this image
            image_outputs = {
                k: v[i:i+1] for k, v in outputs.items()
            }
            
            processed = self.processor.post_process_grounded_object_detection(
                image_outputs,
                threshold=threshold,
                target_sizes=target_sizes,
                text=[queries]
            )[0]
            
            detections = []
            boxes = processed["boxes"].cpu().numpy()
            scores = processed["scores"].cpu().numpy()
            labels = processed["labels"].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                detections.append({
                    "class": queries[label],
                    "confidence": float(score),
                    "bbox": [float(x) for x in box]
                })
            
            results.append({
                "detections": detections,
                "counts": self._count_objects(detections)
            })
        
        return results
    
    def _count_objects(self, detections: List[Dict]) -> Dict[str, int]:
        """Count objects by class"""
        counts = {}
        for d in detections:
            cls = d["class"]
            counts[cls] = counts.get(cls, 0) + 1
        return counts
    
    async def detect_single(
        self,
        image: Image.Image,
        queries: List[str] = None,
        threshold: float = None
    ) -> Dict[str, Any]:
        """Convenience method for single image"""
        results = await self.detect_batch([image], queries, threshold)
        return results[0]
    
    def benchmark(
        self,
        num_images: int = 100,
        batch_size: int = 8,
        image_size: tuple = (640, 640)
    ) -> Dict[str, Any]:
        """
        Run inference benchmark.
        
        Returns timing statistics for the optimized model.
        """
        import numpy as np
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Warmup
        self.warmup(num_iterations=5)
        
        # Create dummy images
        images = [
            Image.fromarray(np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8))
            for _ in range(num_images)
        ]
        
        # Run benchmark
        times = []
        total_images = 0
        
        for i in range(0, num_images, batch_size):
            batch = images[i:i+batch_size]
            
            # Time the batch
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = self._detect_batch_sync(batch, settings.DEFAULT_QUERIES, settings.CONFIDENCE_THRESHOLD)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            total_images += len(batch)
        
        total_time = sum(times)
        images_per_second = total_images / total_time
        avg_latency = total_time / len(times)
        
        return {
            "total_images": total_images,
            "total_time": total_time,
            "images_per_second": images_per_second,
            "avg_batch_latency": avg_latency,
            "batch_size": batch_size,
            "optimization_backend": self.optimization_result.backend if self.optimization_result else "eager",
            "device": self.device
        }
