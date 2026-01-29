# Warehouse Vision System

Real-time industrial engineering study using foundation vision models.

## Detection Targets
- AGVs with ID recognition (Dematic fleet)
- Pallets (count, location)
- People (safety zone violations)
- Dock door state
- Conveyor status (future)

## Tech Stack
- **Detection**: Grounding DINO 1.5
- **Segmentation**: SAM 2
- **OCR**: Florence-2 / PaddleOCR
- **VLM**: Qwen2-VL for scene reasoning
- **Inference**: TensorRT optimization
- **Hardware**: DGX Spark (GB10)

## Structure
```
warehouse-vision/
├── models/          # Downloaded model weights
├── src/             # Source code
├── output/          # Detection results
├── configs/         # Detection configs
└── README.md
```
