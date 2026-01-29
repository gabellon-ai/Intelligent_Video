"""
Video processing service
Handles frame extraction with smart sampling
"""

import cv2
from PIL import Image
from pathlib import Path
from typing import AsyncGenerator, Tuple, List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from ..config import settings


class VideoProcessor:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._get_info_sync, video_path
        )
    
    def _get_info_sync(self, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return info
    
    async def extract_frames(
        self,
        video_path: str,
        sample_fps: float = None,
        max_frames: int = None
    ) -> AsyncGenerator[Tuple[int, float, Image.Image], None]:
        """
        Extract frames from video with smart sampling
        Yields: (frame_number, timestamp_seconds, PIL.Image)
        """
        sample_fps = sample_fps or settings.SAMPLE_FPS
        
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue(maxsize=settings.BATCH_SIZE * 2)
        
        # Run extraction in thread
        extraction_task = loop.run_in_executor(
            self._executor,
            self._extract_frames_sync,
            video_path, sample_fps, max_frames, queue, loop
        )
        
        # Yield frames as they arrive
        while True:
            item = await queue.get()
            if item is None:  # Sentinel for completion
                break
            yield item
        
        await extraction_task
    
    def _extract_frames_sync(
        self,
        video_path: str,
        sample_fps: float,
        max_frames: int,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop
    ):
        """Synchronous frame extraction"""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval for desired sample rate
        frame_interval = max(1, int(video_fps / sample_fps))
        
        frame_num = 0
        extracted = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % frame_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                timestamp = frame_num / video_fps
                
                # Put in queue (blocking if full, which provides backpressure)
                future = asyncio.run_coroutine_threadsafe(
                    queue.put((frame_num, timestamp, pil_image)),
                    loop
                )
                future.result()  # Wait for queue space
                
                extracted += 1
                if max_frames and extracted >= max_frames:
                    break
            
            frame_num += 1
        
        cap.release()
        
        # Signal completion
        asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
    
    async def extract_frames_batch(
        self,
        video_path: str,
        batch_size: int = None,
        sample_fps: float = None
    ) -> AsyncGenerator[List[Tuple[int, float, Image.Image]], None]:
        """
        Extract frames in batches for efficient batch inference
        Yields batches of (frame_number, timestamp, image) tuples
        """
        batch_size = batch_size or settings.BATCH_SIZE
        batch = []
        
        async for frame_data in self.extract_frames(video_path, sample_fps):
            batch.append(frame_data)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining frames
        if batch:
            yield batch
