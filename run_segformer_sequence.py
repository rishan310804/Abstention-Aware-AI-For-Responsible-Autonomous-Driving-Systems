#!/usr/bin/env python3
"""Run SegFormer over a saved CARLA image sequence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


def make_palette(num_classes: int) -> np.ndarray:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="normal_001")
    parser.add_argument("--input-sequence", default=None)
    parser.add_argument("--output-sequence", default=None)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    input_sequence = args.input_sequence or args.sequence
    output_sequence = args.output_sequence or args.sequence
    input_dir = PROJECT_ROOT / "outputs" / "sequences" / input_sequence
    sequence_dir = PROJECT_ROOT / "outputs" / "sequences" / output_sequence
    frame_paths = sorted(input_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {input_dir}")

    prediction_dir = sequence_dir / "predictions"
    logits_dir = sequence_dir / "logits"
    uncertainty_dir = sequence_dir / "uncertainty"
    for directory in (prediction_dir, logits_dir, uncertainty_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    palette = make_palette(model.config.num_labels)
    records = []

    for index, frame_path in enumerate(frame_paths, start=1):
        image = Image.open(frame_path).convert("RGB")
        width, height = image.size
        inputs = processor(images=image, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

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

        stem = frame_path.stem
        prediction_image = Image.fromarray(
            palette[prediction[0].cpu().numpy().astype(np.uint8)], mode="RGB"
        )
        overlay = Image.blend(image, prediction_image, alpha=0.5)
        prediction_image.save(prediction_dir / f"{stem}_segmentation.png")
        overlay.save(prediction_dir / f"{stem}_overlay.png")

        torch.save(
            {
                "checkpoint": args.checkpoint,
                "native_logits": native_logits.detach().cpu(),
                "class_names": model.config.id2label,
            },
            logits_dir / f"{stem}_logits.pt",
        )
        torch.save(msp[0].detach().cpu(), uncertainty_dir / f"{stem}_msp.pt")
        torch.save(max_logit[0].detach().cpu(), uncertainty_dir / f"{stem}_maxlogit.pt")

        records.append(
            {
                "frame": stem,
                "input": str(frame_path),
                "inference_seconds": elapsed,
                "inference_milliseconds": elapsed * 1000.0,
                "native_logit_shape": list(native_logits.shape),
                "upsampled_logit_shape": list(logits.shape),
            }
        )
        print(f"[{index}/{len(frame_paths)}] {stem}: {elapsed * 1000.0:.2f} ms")

    summary = {
        "input_sequence": input_sequence,
        "output_sequence": output_sequence,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "visible_gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "frames_processed": len(records),
        "msp_definition": "maximum softmax probability per pixel after upsampling logits",
        "maxlogit_definition": "maximum raw class logit per pixel after upsampling logits",
        "records": records,
    }
    (sequence_dir / "segformer_metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Processed {len(records)} frames")
    print(f"Metadata: {sequence_dir / 'segformer_metadata.json'}")


if __name__ == "__main__":
    main()
