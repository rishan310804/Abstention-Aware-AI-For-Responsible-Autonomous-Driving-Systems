#!/usr/bin/env python3
"""Analyze uncertainty and temporal instability for a saved sequence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent


def contiguous_runs(values: list[bool]) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="normal_001")
    parser.add_argument("--msp-threshold", type=float, default=0.50)
    parser.add_argument("--low-pixel-fraction-threshold", type=float, default=0.10)
    parser.add_argument("--persistence-frames", type=int, default=3)
    args = parser.parse_args()

    sequence_dir = PROJECT_ROOT / "outputs" / "sequences" / args.sequence
    logits_paths = sorted((sequence_dir / "logits").glob("*_logits.pt"))
    msp_paths = sorted((sequence_dir / "uncertainty").glob("*_msp.pt"))
    maxlogit_paths = sorted((sequence_dir / "uncertainty").glob("*_maxlogit.pt"))

    if not logits_paths or len(logits_paths) != len(msp_paths):
        raise FileNotFoundError("Expected matching logits and MSP files")
    if len(logits_paths) != len(maxlogit_paths):
        raise FileNotFoundError("Expected matching logits and MaxLogit files")

    records: list[dict] = []
    previous_prediction = None
    roi_low_confidence: list[bool] = []

    for index, (logits_path, msp_path, maxlogit_path) in enumerate(
        zip(logits_paths, msp_paths, maxlogit_paths)
    ):
        payload = torch.load(logits_path, map_location="cpu", weights_only=False)
        native_logits = payload["native_logits"][0]
        native_probabilities = torch.softmax(native_logits, dim=0)
        native_entropy = -(
            native_probabilities
            * torch.log(native_probabilities.clamp_min(1e-8))
        ).sum(dim=0)
        native_prediction = native_probabilities.argmax(dim=0).numpy()

        msp = torch.load(msp_path, map_location="cpu", weights_only=False).numpy()
        maxlogit = torch.load(maxlogit_path, map_location="cpu", weights_only=False).numpy()

        height, width = msp.shape
        entropy = torch.nn.functional.interpolate(
            native_entropy.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        y0, y1 = int(height * 0.40), height
        x0, x1 = int(width * 0.25), int(width * 0.75)
        roi_msp = msp[y0:y1, x0:x1]
        roi_maxlogit = maxlogit[y0:y1, x0:x1]
        roi_entropy = entropy[y0:y1, x0:x1]

        native_height, native_width = native_prediction.shape
        native_y0, native_y1 = int(native_height * 0.40), native_height
        native_x0, native_x1 = int(native_width * 0.25), int(native_width * 0.75)
        roi_prediction = native_prediction[native_y0:native_y1, native_x0:native_x1]

        if previous_prediction is None:
            instability_full = None
            instability_roi = None
        else:
            instability_full = float(np.mean(native_prediction != previous_prediction))
            previous_roi = previous_prediction[native_y0:native_y1, native_x0:native_x1]
            instability_roi = float(np.mean(roi_prediction != previous_roi))

        low_pixel_fraction_roi = float(np.mean(roi_msp < args.msp_threshold))
        low_confidence_roi = bool(
            low_pixel_fraction_roi >= args.low_pixel_fraction_threshold
        )
        roi_low_confidence.append(low_confidence_roi)

        records.append(
            {
                "index": index,
                "frame": logits_path.stem.replace("_logits", ""),
                "mean_msp_full": float(np.mean(msp)),
                "p10_msp_full": float(np.percentile(msp, 10)),
                "low_msp_fraction_full": float(np.mean(msp < args.msp_threshold)),
                "mean_maxlogit_full": float(np.mean(maxlogit)),
                "mean_entropy_full": float(np.mean(entropy)),
                "mean_msp_driving_corridor": float(np.mean(roi_msp)),
                "p10_msp_driving_corridor": float(np.percentile(roi_msp, 10)),
                "low_msp_fraction_driving_corridor": float(
                    np.mean(roi_msp < args.msp_threshold)
                ),
                "mean_maxlogit_driving_corridor": float(np.mean(roi_maxlogit)),
                "mean_entropy_driving_corridor": float(np.mean(roi_entropy)),
                "prediction_instability_full": instability_full,
                "prediction_instability_driving_corridor": instability_roi,
            }
        )
        previous_prediction = native_prediction

    csv_path = sequence_dir / "temporal_uncertainty.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    runs = contiguous_runs(roi_low_confidence)
    persistent_runs = [run for run in runs if run >= args.persistence_frames]
    full_instability = [
        r["prediction_instability_full"]
        for r in records
        if r["prediction_instability_full"] is not None
    ]
    corridor_instability = [
        r["prediction_instability_driving_corridor"]
        for r in records
        if r["prediction_instability_driving_corridor"] is not None
    ]
    summary = {
        "sequence": args.sequence,
        "frames": len(records),
        "msp_threshold": args.msp_threshold,
        "low_pixel_fraction_threshold": args.low_pixel_fraction_threshold,
        "persistence_frames": args.persistence_frames,
        "driving_corridor_definition": "lower 60 percent of image and center 50 percent of width",
        "threshold_status": "exploratory; select final thresholds on validation data",
        "mean_msp_full": float(np.mean([r["mean_msp_full"] for r in records])),
        "mean_msp_driving_corridor": float(
            np.mean([r["mean_msp_driving_corridor"] for r in records])
        ),
        "mean_entropy_full": float(np.mean([r["mean_entropy_full"] for r in records])),
        "mean_entropy_driving_corridor": float(
            np.mean([r["mean_entropy_driving_corridor"] for r in records])
        ),
        "mean_prediction_instability_full": float(np.mean(full_instability)),
        "mean_prediction_instability_driving_corridor": float(
            np.mean(corridor_instability)
        ),
        "low_confidence_corridor_runs": runs,
        "persistent_low_confidence_corridor_runs": persistent_runs,
    }
    summary_path = sequence_dir / "temporal_uncertainty_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Analyzed {len(records)} frames")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Mean driving-corridor MSP: {summary['mean_msp_driving_corridor']:.4f}")
    print(
        "Mean driving-corridor instability: "
        f"{summary['mean_prediction_instability_driving_corridor']:.4f}"
    )


if __name__ == "__main__":
    main()
