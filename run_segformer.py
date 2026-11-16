#!/usr/bin/env python3
"""Run pretrained SegFormer semantic segmentation on one CARLA frame."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_ROOT / "outputs" / "frames" / "frame_000001.png"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SEGMENTATION_DIR = OUTPUT_ROOT / "segmentation"
LOGITS_DIR = OUTPUT_ROOT / "logits"
UNCERTAINTY_DIR = OUTPUT_ROOT / "uncertainty"
METADATA_DIR = OUTPUT_ROOT / "metadata"

CHECKPOINT = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


def make_palette(num_classes: int) -> np.ndarray:
    """Create a deterministic RGB palette for class IDs."""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        value = class_id
        red = green = blue = 0
        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        palette[class_id] = (red, green, blue)
    return palette


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

    for directory in (SEGMENTATION_DIR, LOGITS_DIR, UNCERTAINTY_DIR, METADATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(IMAGE_PATH).convert("RGB")
    width, height = image.size

    print(f"Loading checkpoint: {CHECKPOINT}")
    print(f"Device: {device}")
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    model = SegformerForSemanticSegmentation.from_pretrained(CHECKPOINT)
    model.to(device)
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start_time

    native_logits = outputs.logits
    logits = torch.nn.functional.interpolate(
        native_logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    probabilities = torch.softmax(logits, dim=1)
    msp, prediction = probabilities.max(dim=1)
    max_logit, _ = logits.max(dim=1)

    prediction_array = prediction[0].cpu().numpy().astype(np.uint8)
    num_classes = int(logits.shape[1])
    palette = make_palette(num_classes)
    colorized = Image.fromarray(palette[prediction_array], mode="RGB")
    overlay = Image.blend(image, colorized, alpha=0.5)

    colorized.save(SEGMENTATION_DIR / "frame_000001_segmentation.png")
    overlay.save(SEGMENTATION_DIR / "frame_000001_overlay.png")

    torch.save(
        {
            "checkpoint": CHECKPOINT,
            "native_logits": native_logits.detach().cpu(),
            "upsampled_logits": logits.detach().cpu(),
            "class_names": model.config.id2label,
        },
        LOGITS_DIR / "frame_000001_logits.pt",
    )
    torch.save(msp[0].detach().cpu(), UNCERTAINTY_DIR / "frame_000001_msp.pt")
    torch.save(max_logit[0].detach().cpu(), UNCERTAINTY_DIR / "frame_000001_maxlogit.pt")

    metadata = {
        "checkpoint": CHECKPOINT,
        "device": str(device),
        "visible_gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "input_image": str(IMAGE_PATH),
        "image_width": width,
        "image_height": height,
        "native_logit_shape": list(native_logits.shape),
        "upsampled_logit_shape": list(logits.shape),
        "num_classes": num_classes,
        "inference_seconds": inference_seconds,
        "inference_milliseconds": inference_seconds * 1000.0,
        "msp_definition": "maximum softmax probability per pixel after upsampling logits",
        "maxlogit_definition": "maximum raw class logit per pixel after upsampling logits",
    }
    (METADATA_DIR / "frame_000001_inference.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("SegFormer inference complete")
    print(f"Segmentation: {SEGMENTATION_DIR / 'frame_000001_segmentation.png'}")
    print(f"Overlay: {SEGMENTATION_DIR / 'frame_000001_overlay.png'}")
    print(f"Logits: {LOGITS_DIR / 'frame_000001_logits.pt'}")
    print(f"Inference time: {inference_seconds * 1000.0:.2f} ms")


if __name__ == "__main__":
    main()
