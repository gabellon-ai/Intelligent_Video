# Intelligent Video - Warehouse Vision Platform

[![CI](https://github.com/gabellon-ai/Intelligent_Video/actions/workflows/ci.yml/badge.svg)](https://github.com/gabellon-ai/Intelligent_Video/actions/workflows/ci.yml)

**Commercial-grade video analytics for warehouse and logistics operations.**

Upload video, get instant object detection with real-time visualization—forklifts, people, pallets, AGVs, and more.

![Dashboard Preview](docs/preview.png)

## Features

- **🎬 Video Upload** — Drag & drop any video format
- **⚡ Fast Analysis** — Smart frame sampling (5 FPS) + batch GPU inference
- **📊 Real-time Dashboard** — Watch detections appear as video processes
- **🎯 Zero-shot Detection** — No training needed, just describe what to find
- **📈 Summary Reports** — Object counts, timelines, activity heatmaps
- **🐳 Docker Ready** — One-command deployment

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and start
git clone https://github.com/gabellon-ai/Intelligent_Video.git
cd Intelligent_Video
docker-compose up --build

# Open http://localhost:3000
```

### Option 2: Local Development

```bash
# Backend (requires Python 3.10+, CUDA optional)
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev

# Open http://localhost:5173
```

## Detection Presets

| Preset | Objects Detected |
|--------|-----------------|
| **Warehouse - General** | Forklifts, pallets, people, boxes, conveyors |
| **Safety Focus** | People, safety vests, zone violations |
| **AGV Tracking** | AGVs, AMRs, autonomous robots |
| **Loading Dock** | Trucks, trailers, dock doors |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  React Frontend │────▶│  FastAPI Backend│────▶│   OWLv2 Model   │
│  (Vite + TW)    │◀────│  (WebSocket)    │◀────│   (GPU/CPU)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │
         │              ┌───────▼───────┐
         │              │ Video Process │
         │              │ (OpenCV)      │
         │              └───────────────┘
         │
    Real-time updates via WebSocket
```

## Performance

| Hardware | Mode | Processing Speed | Latency |
|----------|------|-----------------|---------|
| DGX Spark | TensorRT FP16 | ~50 FPS analyzed | <160ms/batch |
| RTX 4090 | TensorRT FP16 | ~45 FPS analyzed | <80ms/batch |
| RTX 4090 | torch.compile | ~15 FPS analyzed | <100ms/batch |
| RTX 3080 | torch.compile | ~10 FPS analyzed | <150ms/batch |
| CPU only | Eager | ~1 FPS analyzed | ~1s/batch |

**Optimization strategies:**
- **TensorRT optimization**: Up to 10x speedup on NVIDIA GPUs
- **torch.compile fallback**: 2-3x speedup when TensorRT unavailable
- Smart sampling: 5 FPS analyzed vs 30 FPS raw (6x faster)
- Batch inference: 8 frames at once (8x throughput)
- Progressive results: See detections immediately, don't wait for completion

### Model Optimization

Pre-optimize the model for best performance:

```bash
# Run optimization script (first time only)
python scripts/optimize_model.py --benchmark

# Compare performance
python scripts/benchmark.py --compare
```

See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) for detailed optimization guide.

## API Reference

### Upload Video
```bash
POST /api/videos/upload
Content-Type: multipart/form-data

# Returns: { job_id, status }
```

### WebSocket (Real-time)
```javascript
ws://localhost:8000/api/streams/ws/{job_id}

// Messages received:
{ type: "progress", percent: 50 }
{ type: "detection", frame: 100, detections: [...] }
{ type: "summary", total_counts: {...} }
{ type: "complete" }
```

### Get Results
```bash
GET /api/videos/{job_id}/results
```

## Project Structure

```
Intelligent_Video/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app
│   │   ├── config.py         # Settings
│   │   ├── routers/          # API endpoints
│   │   ├── services/         # Detection, video processing
│   │   └── models/           # Pydantic schemas
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/            # Upload, Analysis pages
│   │   └── components/       # VideoPlayer, Timeline, Summary
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
└── README.md
```

## Roadmap

- [x] OWLv2 zero-shot detection
- [x] Video upload + batch processing
- [x] Real-time WebSocket updates
- [x] Detection overlay visualization
- [x] TensorRT/torch.compile optimization (up to 10x speedup)
- [ ] RTSP live stream support
- [ ] Multi-camera dashboard
- [ ] Alert notifications
- [ ] Export to CSV/PDF
- [ ] Custom model fine-tuning

## License

Commercial license. Contact Blueshift Ops for pricing.

## Author

**Blueshift Ops**  
Enterprise warehouse intelligence solutions
