"""
Real-time streaming endpoints
WebSocket for live video analysis
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional
import json
import asyncio
import logging

from ..services.detector import DetectorService
from ..services.video_processor import VideoProcessor
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id] = websocket
    
    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]
    
    async def send_progress(self, job_id: str, data: dict):
        if job_id in self.active_connections:
            await self.active_connections[job_id].send_json(data)


manager = ConnectionManager()


@router.websocket("/ws/{job_id}")
async def analysis_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket for real-time analysis progress
    
    Messages sent to client:
    - {"type": "progress", "frame": N, "total": M, "percent": P}
    - {"type": "detection", "frame": N, "timestamp": T, "detections": [...]}
    - {"type": "summary", "counts": {...}, "timeline": [...]}
    - {"type": "complete"}
    - {"type": "error", "message": "..."}
    """
    await manager.connect(websocket, job_id)
    logger.info(f"[ws:{job_id}] connected")

    try:
        from .videos import jobs

        if job_id not in jobs:
            logger.warning(f"[ws:{job_id}] job not found")
            await websocket.send_json({"type": "error", "message": "Job not found"})
            return

        job = jobs[job_id]
        from ..main import get_detector
        detector = get_detector()
        processor = VideoProcessor()

        logger.info(f"[ws:{job_id}] probing video {job.filepath}")
        video_info = await processor.get_video_info(job.filepath)
        estimated_frames = int(video_info["duration"] * settings.SAMPLE_FPS)
        job.total_frames = estimated_frames
        logger.info(f"[ws:{job_id}] video_info={video_info} estimated_frames={estimated_frames}")

        await websocket.send_json({
            "type": "start",
            "video_info": video_info,
            "estimated_frames": estimated_frames,
        })
        logger.info(f"[ws:{job_id}] sent 'start'")

        job.status = "processing"
        all_detections = []
        frame_count = 0
        batch_idx = 0

        async for batch in processor.extract_frames_batch(job.filepath):
            batch_idx += 1
            frame_nums = [b[0] for b in batch]
            timestamps = [b[1] for b in batch]
            images = [b[2] for b in batch]
            logger.info(f"[ws:{job_id}] batch#{batch_idx} size={len(images)} running detect_batch")

            results = await detector.detect_batch(images)
            logger.info(f"[ws:{job_id}] batch#{batch_idx} detection done, sending frames")

            for i, (frame_num, timestamp, result) in enumerate(zip(frame_nums, timestamps, results)):
                frame_count += 1
                detection_data = {
                    "frame": frame_num,
                    "timestamp": timestamp,
                    "detections": result["detections"],
                    "counts": result["counts"],
                }
                all_detections.append(detection_data)

                await websocket.send_json({"type": "detection", **detection_data})

                if frame_count % 5 == 0:
                    await websocket.send_json({
                        "type": "progress",
                        "frame": frame_count,
                        "total": estimated_frames,
                        "percent": min(99, int(frame_count / estimated_frames * 100)),
                    })

        logger.info(f"[ws:{job_id}] loop complete, {frame_count} frames processed")
        summary = generate_summary(all_detections)
        job.summary = summary
        job.timeline = all_detections
        job.frames_processed = frame_count
        job.status = "completed"
        job.progress = 100

        await websocket.send_json({"type": "summary", **summary})
        await websocket.send_json({"type": "complete"})
        logger.info(f"[ws:{job_id}] done")

    except WebSocketDisconnect:
        logger.warning(f"[ws:{job_id}] WebSocketDisconnect raised")
    except Exception as e:
        logger.exception(f"[ws:{job_id}] handler crashed: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        manager.disconnect(job_id)
        logger.info(f"[ws:{job_id}] cleaned up")


def generate_summary(detections: list) -> dict:
    """Generate analysis summary from all detections"""
    total_counts = {}
    max_counts = {}
    timeline_summary = []
    
    for d in detections:
        for cls, count in d["counts"].items():
            total_counts[cls] = total_counts.get(cls, 0) + count
            max_counts[cls] = max(max_counts.get(cls, 0), count)
        
        # Simplified timeline entry
        if d["detections"]:
            timeline_summary.append({
                "timestamp": d["timestamp"],
                "counts": d["counts"]
            })
    
    return {
        "total_detections": sum(total_counts.values()),
        "unique_classes": list(total_counts.keys()),
        "total_counts": total_counts,
        "max_simultaneous": max_counts,
        "frames_with_detections": len([d for d in detections if d["detections"]]),
        "total_frames_analyzed": len(detections)
    }
