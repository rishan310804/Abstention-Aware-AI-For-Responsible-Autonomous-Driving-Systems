#!/usr/bin/env python3
"""Select confidence thresholds from validation correctness and ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_ground_truth import CARLA_TO_CITYSCAPES


PROJECT_ROOT = Path(__file__).resolve().parent


def load_prediction(logit_path: Path, shape: tuple[int, int]) -> np.ndarray:
    payload = torch.load(logit_path, map_location="cpu", weights_only=False)
    logits = payload["native_logits"]
    logits = torch.nn.functional.interpolate(
        logits, size=shape, mode="bilinear", align_corners=False
    )
    return logits.argmax(dim=1)[0].numpy()


def choose_threshold(confidence: np.ndarray, correct: np.ndarray, target_error: float) -> dict:
    best = None
    for threshold in np.arange(0.50, 1.001, 0.01):
        accepted = confidence >= threshold
        if not np.any(accepted):
            continue
        coverage = float(np.mean(accepted))
        risk = float(1.0 - np.mean(correct[accepted]))
        if risk <= target_error and (best is None or coverage > best["coverage"]):
            best = {
                "threshold": round(float(threshold), 2),
                "coverage": coverage,
                "selective_risk": risk,
                "accepted_pixels": int(np.sum(accepted)),
            }
    if best is None:
        best = {
            "threshold": None,
            "coverage": 0.0,
            "selective_risk": None,
            "accepted_pixels": 0,
        }
    best["target_error"] = target_error
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="Validation model output sequence")
    parser.add_argument("--ground-truth-sequence", default="eval_val_001")
    parser.add_argument("--target-error", type=float, default=0.05)
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / "outputs" / "sequences" / args.sequence
    gt_dir = PROJECT_ROOT / "outputs" / "sequences" / args.ground_truth_sequence / "ground_truth"
    logits_paths = sorted((model_dir / "logits").glob("*_logits.pt"))
    msp_paths = sorted((model_dir / "uncertainty").glob("*_msp.pt"))
    gt_paths = sorted(gt_dir.glob("frame_*.png"))
    if not (logits_paths and len(logits_paths) == len(msp_paths) == len(gt_paths)):
        raise FileNotFoundError("Expected matching logits, MSP, and ground-truth files")

    all_confidence = []
    all_correct = []
    corridor_confidence = []
    corridor_correct = []

    for logits_path, msp_path, gt_path in zip(logits_paths, msp_paths, gt_paths):
        gt_tags = np.asarray(Image.open(gt_path))[..., 0]
        gt = np.full(gt_tags.shape, 255, dtype=np.uint8)
        for carla_tag, cityscapes_id in CARLA_TO_CITYSCAPES.items():
            gt[gt_tags == carla_tag] = cityscapes_id

        prediction = load_prediction(logits_path, gt.shape)
        confidence = torch.load(msp_path, map_location="cpu", weights_only=False).numpy()
        valid = gt != 255
        correct = prediction == gt
        corridor = np.zeros_like(valid)
        height, width = gt.shape
        corridor[int(height * 0.40):, int(width * 0.25):int(width * 0.75)] = True
        corridor_valid = valid & corridor

        all_confidence.append(confidence[valid])
        all_correct.append(correct[valid])
        corridor_confidence.append(confidence[corridor_valid])
        corridor_correct.append(correct[corridor_valid])

    confidence = np.concatenate(all_confidence)
    correct = np.concatenate(all_correct)
    corridor_confidence = np.concatenate(corridor_confidence)
    corridor_correct = np.concatenate(corridor_correct)

    result = {
        "validation_sequence": args.ground_truth_sequence,
        "model_sequence": args.sequence,
        "target_error": args.target_error,
        "full_image": choose_threshold(confidence, correct, args.target_error),
        "driving_corridor": choose_threshold(
            corridor_confidence, corridor_correct, args.target_error
        ),
        "threshold_status": "selected on validation data; do not retune on final test data",
    }
    output_path = model_dir / "calibrated_thresholds.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Calibration: {output_path}")
    print("Full image:", result["full_image"])
    print("Driving corridor:", result["driving_corridor"])


if __name__ == "__main__":
    main()
