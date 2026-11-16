#!/usr/bin/env python3
"""Run Cityscapes Mask2Former semantic segmentation on a saved sequence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = "facebook/mask2former-swin-small-cityscapes-semantic"


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
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.checkpoint)

    # The published Cityscapes checkpoint does not contain the final Swin
    # LayerNorm tensors, while newer Transformers creates that module.  Leaving
    # it untouched would use randomly initialized parameters.  Replacing it
    # with Identity preserves the checkpoint's available computation and makes
    # the run deterministic instead of silently using random weights.
    model.model.pixel_level_module.encoder.swin.layernorm = torch.nn.Identity()

    model.to(device)
    model.eval()

    num_classes = model.config.num_labels
    palette = make_palette(num_classes)
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

            # Convert query-level Mask2Former outputs into semantic class scores.
            # The final class-query logit is the no-object class and is excluded.
            class_probabilities = outputs.class_queries_logits.softmax(dim=-1)[
                ..., :num_classes
            ]
            mask_probabilities = outputs.masks_queries_logits.sigmoid()
            semantic_scores = torch.einsum(
                "bqc,bqhw->bchw", class_probabilities, mask_probabilities
            )
            semantic_scores = torch.nn.functional.interpolate(
                semantic_scores,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            # Mask2Former semantic scores are class-query/mask scores, not
            # logits. Normalize them across classes before deriving MSP.
            probabilities = semantic_scores / semantic_scores.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            logits = torch.log(probabilities.clamp_min(1e-8))
            msp, prediction = probabilities.max(dim=1)
            max_logit, _ = semantic_scores.max(dim=1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

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
                "native_logits": logits.detach().cpu(),
                "class_names": model.config.id2label,
                "logit_definition": "log of normalized Mask2Former semantic scores; used for temporal argmax comparison",
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
                "semantic_score_shape": list(logits.shape),
            }
        )
        print(f"[{index}/{len(frame_paths)}] {stem}: {elapsed * 1000.0:.2f} ms")

    summary = {
        "input_sequence": input_sequence,
        "output_sequence": output_sequence,
        "checkpoint": args.checkpoint,
        "architecture": "Mask2Former",
        "checkpoint_loading_adjustment": (
            "replaced missing Swin final LayerNorm with Identity; "
            "the published checkpoint has no corresponding weights"
        ),
        "device": str(device),
        "visible_gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "frames_processed": len(records),
        "msp_definition": "maximum normalized Mask2Former semantic score per pixel",
        "maxlogit_definition": "maximum composed Mask2Former semantic score per pixel",
        "records": records,
    }
    (sequence_dir / "mask2former_metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Processed {len(records)} frames")
    print(f"Metadata: {sequence_dir / 'mask2former_metadata.json'}")


if __name__ == "__main__":
    main()
