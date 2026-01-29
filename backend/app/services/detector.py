"""
Detection service - OWLv2 zero-shot object detection
Optimized for batch processing
"""

import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..config import settings


class DetectorService:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = None
        self.is_loaded = False
        self.gpu_available = torch.cuda.is_available()
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    async def load_model(self):
        """Load model asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_model_sync)
    
    def _load_model_sync(self):
        """Synchronous model loading"""
        self.processor = Owlv2Processor.from_pretrained(settings.MODEL_NAME)
        self.model = Owlv2ForObjectDetection.from_pretrained(settings.MODEL_NAME)
        
        if self.gpu_available:
            self.device = "cuda"
            self.model = self.model.to(self.device)
            # Enable inference optimizations
            self.model = self.model.eval()
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception:
                    pass  # torch.compile not available on all setups
        else:
            self.device = "cpu"
        
        self.is_loaded = True
    
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
        """Synchronous batch detection"""
        
        # Process all images in batch
        inputs = self.processor(
            text=[queries] * len(images),
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
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
