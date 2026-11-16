#!/usr/bin/env python3
"""Evaluate saved Cityscapes-label predictions against CARLA semantic tags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
NUM_CLASSES = 19
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
]

# CARLA 0.9.16 semantic tag -> Cityscapes class. Unsupported CARLA tags remain
# 255 and are excluded from metrics. This uses the actual CityObjectLabel enum
# values exposed by the installed CARLA Python API.
CARLA_TO_CITYSCAPES = {
    3: 2,   # Buildings
    5: 4,   # Fences
    12: 11, # Pedestrians
    6: 5,   # Poles
    1: 0,   # Roads
    2: 1,   # Sidewalks
    9: 8,   # Vegetation
    14: 13, # Car
    15: 14, # Truck
    16: 15, # Bus
    17: 16, # Train
    18: 17, # Motorcycle
    19: 18, # Bicycle
    4: 3,  # Walls
    8: 7,  # TrafficSigns
    11: 10, # Sky
    25: 9,  # Ground -> terrain
    7: 6,  # TrafficLight
    13: 12, # Rider
    10: 9,  # Terrain
}


def load_prediction(logit_path: Path, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(logit_path, map_location="cpu", weights_only=False)
    logits = payload["native_logits"]
    logits = torch.nn.functional.interpolate(
        logits, size=(height, width), mode="bilinear", align_corners=False
    )
    probabilities = torch.softmax(logits, dim=1)[0]
    confidence, prediction = probabilities.max(dim=0)
    return prediction.numpy(), confidence.numpy()


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray) -> float:
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            confidence <= upper if upper >= 1.0 else confidence < upper
        )
        if np.any(mask):
            ece += np.mean(mask) * abs(float(np.mean(confidence[mask])) - float(np.mean(correct[mask])))
    return float(ece)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="Model output sequence")
    parser.add_argument("--ground-truth-sequence", default="eval_normal_001")
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / "outputs" / "sequences" / args.sequence
    gt_dir = PROJECT_ROOT / "outputs" / "sequences" / args.ground_truth_sequence / "ground_truth"
    logit_paths = sorted((model_dir / "logits").glob("*_logits.pt"))
    gt_paths = sorted(gt_dir.glob("frame_*.png"))
    if len(logit_paths) != len(gt_paths) or not logit_paths:
        raise FileNotFoundError("Expected matching model logits and ground-truth frames")

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    all_confidence = []
    all_correct = []
    corridor_correct = []
    frame_records = []

    for logit_path, gt_path in zip(logit_paths, gt_paths):
        gt_tags = np.asarray(Image.open(gt_path))[..., 0]
        gt = np.full(gt_tags.shape, 255, dtype=np.uint8)
        for carla_tag, cityscapes_id in CARLA_TO_CITYSCAPES.items():
            gt[gt_tags == carla_tag] = cityscapes_id

        prediction, confidence = load_prediction(logit_path, *gt.shape)
        valid = gt != 255
        correct = prediction == gt
        valid_confidence = confidence[valid]
        valid_correct = correct[valid]
        all_confidence.append(valid_confidence)
        all_correct.append(valid_correct)

        height, width = gt.shape
        corridor = np.zeros_like(valid)
        corridor[int(height * 0.40):, int(width * 0.25):int(width * 0.75)] = True
        corridor_valid = valid & corridor
        corridor_correct.append(correct[corridor_valid])

        for target in range(NUM_CLASSES):
            for predicted in range(NUM_CLASSES):
                confusion[target, predicted] += np.sum(
                    (gt == target) & (prediction == predicted)
                )
        frame_records.append(
            {
                "frame": logit_path.stem.replace("_logits", ""),
                "valid_pixel_fraction": float(np.mean(valid)),
                "accuracy": float(np.mean(valid_correct)),
                "corridor_accuracy": float(np.mean(correct[corridor_valid])),
            }
        )

    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.full(NUM_CLASSES, np.nan), where=union > 0)
    confidence = np.concatenate(all_confidence)
    correct = np.concatenate(all_correct).astype(np.float32)
    corridor = np.concatenate(corridor_correct).astype(np.float32)

    result = {
        "model_sequence": args.sequence,
        "ground_truth_sequence": args.ground_truth_sequence,
        "frames": len(frame_records),
        "label_mapping": "CARLA 0.9.16 CityObjectLabel enum values mapped to Cityscapes; unsupported tags ignored",
        "valid_pixel_fraction": float(np.mean([r["valid_pixel_fraction"] for r in frame_records])),
        "pixel_accuracy": float(np.mean(correct)),
        "driving_corridor_accuracy": float(np.mean(corridor)),
        "mean_iou": float(np.nanmean(iou)),
        "expected_calibration_error": expected_calibration_error(confidence, correct),
        "mean_confidence_correct": float(np.mean(confidence[correct == 1])),
        "mean_confidence_incorrect": float(np.mean(confidence[correct == 0])) if np.any(correct == 0) else None,
        "per_class_iou": {
            CLASS_NAMES[i]: (None if np.isnan(iou[i]) else float(iou[i]))
            for i in range(NUM_CLASSES)
        },
        "frames_summary": frame_records,
    }
    output_path = model_dir / "ground_truth_evaluation.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Evaluated {len(frame_records)} frames")
    print(f"Evaluation: {output_path}")
    print(f"mIoU: {result['mean_iou']:.4f}")
    print(f"Pixel accuracy: {result['pixel_accuracy']:.4f}")
    print(f"Driving-corridor accuracy: {result['driving_corridor_accuracy']:.4f}")
    print(f"ECE: {result['expected_calibration_error']:.4f}")


if __name__ == "__main__":
    main()
