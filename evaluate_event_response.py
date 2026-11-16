#!/usr/bin/env python3
"""Measure detection and recovery delay for a known temporal disturbance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def first_persistent(values: list[bool], start: int, run_length: int) -> int | None:
    for index in range(start, len(values) - run_length + 1):
        if all(values[index:index + run_length]):
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--disturbance-start", type=int, required=True)
    parser.add_argument("--disturbance-end", type=int, required=True)
    parser.add_argument("--msp-threshold", type=float, default=0.50)
    parser.add_argument("--low-pixel-fraction-threshold", type=float, default=0.10)
    parser.add_argument("--persistence-frames", type=int, default=3)
    parser.add_argument("--severe-pixel-fraction-threshold", type=float, default=0.25)
    args = parser.parse_args()

    sequence_dir = PROJECT_ROOT / "outputs" / "sequences" / args.sequence
    csv_path = sequence_dir / "temporal_uncertainty.csv"
    rows = list(csv.DictReader(csv_path.open()))
    low_confidence = []
    severe = []
    for row in rows:
        fraction = float(row["low_msp_fraction_driving_corridor"])
        low_confidence.append(fraction >= args.low_pixel_fraction_threshold)
        severe.append(fraction >= args.severe_pixel_fraction_threshold)

    detection_frame = first_persistent(
        low_confidence, args.disturbance_start, args.persistence_frames
    )
    severe_frame = first_persistent(
        severe, args.disturbance_start, args.persistence_frames
    )
    recovery_frame = None
    recovery_run = args.persistence_frames
    for index in range(args.disturbance_end + 1, len(rows) - recovery_run + 1):
        if all(not value for value in low_confidence[index:index + recovery_run]):
            recovery_frame = index
            break

    result = {
        "sequence": args.sequence,
        "disturbance_start": args.disturbance_start,
        "disturbance_end": args.disturbance_end,
        "persistence_frames": args.persistence_frames,
        "low_pixel_fraction_threshold": args.low_pixel_fraction_threshold,
        "severe_pixel_fraction_threshold": args.severe_pixel_fraction_threshold,
        "detection_frame": detection_frame,
        "detection_delay_frames": (
            None if detection_frame is None else detection_frame - args.disturbance_start
        ),
        "severe_detection_frame": severe_frame,
        "severe_detection_delay_frames": (
            None if severe_frame is None else severe_frame - args.disturbance_start
        ),
        "recovery_frame": recovery_frame,
        "recovery_delay_frames": (
            None if recovery_frame is None else recovery_frame - args.disturbance_end - 1
        ),
    }
    output_path = sequence_dir / "event_response.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Event response: {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
