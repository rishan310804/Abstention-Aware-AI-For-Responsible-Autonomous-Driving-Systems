#!/usr/bin/env python3
"""Apply validation-frozen thresholds to a held-out test sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_ground_truth import CARLA_TO_CITYSCAPES


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="Held-out test model sequence")
    parser.add_argument("--ground-truth-sequence", default="eval_test_001")
    parser.add_argument("--threshold-sequence", default=None)
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / "outputs" / "sequences" / args.sequence
    gt_dir = PROJECT_ROOT / "outputs" / "sequences" / args.ground_truth_sequence / "ground_truth"
    threshold_sequence = args.threshold_sequence or args.sequence.replace(
        "eval_test_001", "eval_val_001"
    )
    threshold_path = (
        PROJECT_ROOT / "outputs" / "sequences" / threshold_sequence / "calibrated_thresholds.json"
    )
    thresholds = json.loads(threshold_path.read_text())
    full_threshold = thresholds["full_image"]["threshold"]
    corridor_threshold = thresholds["driving_corridor"]["threshold"]
    if full_threshold is None or corridor_threshold is None:
        raise RuntimeError("Frozen threshold selection produced no usable threshold")

    logits_paths = sorted((model_dir / "logits").glob("*_logits.pt"))
    msp_paths = sorted((model_dir / "uncertainty").glob("*_msp.pt"))
    gt_paths = sorted(gt_dir.glob("frame_*.png"))
    if not (logits_paths and len(logits_paths) == len(msp_paths) == len(gt_paths)):
        raise FileNotFoundError("Expected matching test logits, MSP, and ground truth")

    full_confidence, full_correct = [], []
    corridor_confidence, corridor_correct = [], []
    for logits_path, msp_path, gt_path in zip(logits_paths, msp_paths, gt_paths):
        gt_tags = np.asarray(Image.open(gt_path))[..., 0]
        gt = np.full(gt_tags.shape, 255, dtype=np.uint8)
        for carla_tag, cityscapes_id in CARLA_TO_CITYSCAPES.items():
            gt[gt_tags == carla_tag] = cityscapes_id

        payload = torch.load(logits_path, map_location="cpu", weights_only=False)
        logits = torch.nn.functional.interpolate(
            payload["native_logits"], size=gt.shape, mode="bilinear", align_corners=False
        )
        prediction = logits.argmax(dim=1)[0].numpy()
        confidence = torch.load(msp_path, map_location="cpu", weights_only=False).numpy()
        valid = gt != 255
        correct = prediction == gt
        height, width = gt.shape
        corridor = np.zeros_like(valid)
        corridor[int(height * 0.40):, int(width * 0.25):int(width * 0.75)] = True

        full_confidence.append(confidence[valid])
        full_correct.append(correct[valid])
        corridor_valid = valid & corridor
        corridor_confidence.append(confidence[corridor_valid])
        corridor_correct.append(correct[corridor_valid])

    def summarize(confidence_parts, correct_parts, threshold):
        confidence = np.concatenate(confidence_parts)
        correct = np.concatenate(correct_parts)
        accepted = confidence >= threshold
        return {
            "threshold": threshold,
            "coverage": float(np.mean(accepted)),
            "abstention_rate": float(1.0 - np.mean(accepted)),
            "selective_risk": float(1.0 - np.mean(correct[accepted])) if np.any(accepted) else None,
            "accepted_pixels": int(np.sum(accepted)),
            "total_valid_pixels": int(len(correct)),
        }

    result = {
        "test_sequence": args.ground_truth_sequence,
        "model_sequence": args.sequence,
        "threshold_sequence": threshold_sequence,
        "full_image": summarize(full_confidence, full_correct, full_threshold),
        "driving_corridor": summarize(
            corridor_confidence, corridor_correct, corridor_threshold
        ),
        "threshold_status": "frozen from validation; no test retuning",
    }
    output_path = model_dir / "frozen_threshold_test_evaluation.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Evaluation: {output_path}")
    print("Full image:", result["full_image"])
    print("Driving corridor:", result["driving_corridor"])


if __name__ == "__main__":
    main()
