#!/usr/bin/env python3
"""
Warehouse Vision Demo - Zero-shot detection using OWLv2
Quick proof of concept before full Grounding DINO + SAM2 pipeline
"""

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import sys
import os

def load_model():
    """Load OWLv2 model for zero-shot detection"""
    print("Loading OWLv2 model...")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")
    
    return processor, model

def detect_objects(image_path, processor, model, queries):
    """Run detection on an image with text queries"""
    image = Image.open(image_path).convert("RGB")
    
    # Process image with text queries
    inputs = processor(text=queries, images=image, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process
    target_sizes = torch.tensor([image.size[::-1]])
    if torch.cuda.is_available():
        target_sizes = target_sizes.to("cuda")
    
    results = processor.post_process_grounded_object_detection(
        outputs, 
        threshold=0.1,
        target_sizes=target_sizes,
        text=queries
    )[0]
    
    return image, results

def draw_detections(image, results, queries):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    
    # Colors for different classes
    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
    
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.15:  # Confidence threshold
            x1, y1, x2, y2 = box
            color = colors[label % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            label_text = f"{queries[label]}: {score:.2f}"
            draw.text((x1, y1 - 15), label_text, fill=color)
            
            detections.append({
                "class": queries[label],
                "confidence": float(score),
                "bbox": [float(x) for x in [x1, y1, x2, y2]]
            })
    
    return image, detections

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_demo.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Warehouse-specific detection queries
    queries = [
        "forklift",
        "AGV automated guided vehicle", 
        "pallet",
        "person",
        "cardboard box",
        "conveyor belt"
    ]
    
    processor, model = load_model()
    image, results = detect_objects(image_path, processor, model, queries)
    annotated_image, detections = draw_detections(image, results, queries)
    
    # Save output
    output_dir = os.path.dirname(image_path).replace("frames", "output") or "../output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "detected_" + os.path.basename(image_path))
    annotated_image.save(output_path)
    
    print(f"\nDetections saved to: {output_path}")
    print(f"\nFound {len(detections)} objects:")
    for d in detections:
        print(f"  - {d['class']}: {d['confidence']:.2%}")

if __name__ == "__main__":
    main()
