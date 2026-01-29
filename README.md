# Intelligent Video - Warehouse Vision System

[![CI](https://github.com/gabellon-ai/Intelligent_Video/actions/workflows/ci.yml/badge.svg)](https://github.com/gabellon-ai/Intelligent_Video/actions/workflows/ci.yml)

Real-time industrial object detection using zero-shot foundation vision models. Designed for warehouse and logistics environments.

## Features

- **Zero-shot detection** — No training required, just describe what to find
- **Warehouse-optimized** — Pre-configured for forklifts, AGVs, pallets, people, boxes
- **GPU accelerated** — CUDA support for real-time inference
- **Extensible** — Easy to add new detection targets

## Detection Targets

| Object | Use Case |
|--------|----------|
| Forklifts | Traffic monitoring, safety zones |
| AGVs | Fleet tracking, ID recognition |
| Pallets | Inventory counting, location tracking |
| People | Safety compliance, zone violations |
| Boxes | Package detection, conveyor monitoring |
| Conveyors | Operational status (future) |

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU

### Installation

```bash
git clone https://github.com/gabellon-ai/Intelligent_Video.git
cd Intelligent_Video

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run detection on an image
python src/detect_demo.py path/to/warehouse_image.jpg
```

Output is saved to `output/detected_<filename>.jpg` with bounding boxes and confidence scores.

### Custom Detection Targets

Edit the `queries` list in `detect_demo.py`:

```python
queries = [
    "forklift",
    "pallet", 
    "person wearing safety vest",
    "cardboard box",
    "your custom object here"
]
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection | OWLv2 (Google) |
| Framework | PyTorch + Transformers |
| Inference | CUDA / CPU |

### Roadmap

- [ ] Grounding DINO 1.5 integration
- [ ] SAM 2 segmentation
- [ ] Florence-2 / PaddleOCR for text recognition
- [ ] Qwen2-VL for scene reasoning
- [ ] TensorRT optimization
- [ ] Real-time video streaming

## Project Structure

```
Intelligent_Video/
├── .github/workflows/  # CI pipeline
├── configs/            # Detection configurations
├── models/             # Downloaded model weights (gitignored)
├── output/             # Detection results (gitignored)
├── src/
│   └── detect_demo.py  # Main detection script
├── requirements.txt
└── README.md
```

## Hardware

Developed on NVIDIA DGX Spark (GB10). Works on any CUDA GPU or CPU (slower).

## License

MIT

## Author

Blueshift Ops
