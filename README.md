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

| Hardware | Processing Speed | Latency |
|----------|-----------------|---------|
| RTX 4090 | ~15 FPS analyzed | <100ms/batch |
| RTX 3080 | ~10 FPS analyzed | <150ms/batch |
| CPU only | ~1 FPS analyzed | ~1s/batch |

**Optimization strategies:**
- Smart sampling: 5 FPS analyzed vs 30 FPS raw (6x faster)
- Batch inference: 8 frames at once (8x throughput)
- Progressive results: See detections immediately, don't wait for completion

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
- [ ] TensorRT optimization (10x speedup)
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
