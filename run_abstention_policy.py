#!/usr/bin/env python3
"""Apply the four-state abstention policy to temporal uncertainty records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
STATES = ("Normal", "Caution", "Abstention", "Escalation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--threshold-sequence", default=None)
    parser.add_argument("--mode", choices=("full", "confidence_only", "no_persistence", "no_corridor"), default="full")
    parser.add_argument("--persistence-frames", type=int, default=3)
    parser.add_argument("--recovery-frames", type=int, default=3)
    parser.add_argument("--instability-threshold", type=float, default=0.05)
    parser.add_argument("--severe-instability-threshold", type=float, default=0.15)
    parser.add_argument("--low-pixel-fraction-threshold", type=float, default=0.10)
    parser.add_argument("--severe-pixel-fraction-threshold", type=float, default=0.25)
    args = parser.parse_args()

    sequence_dir = PROJECT_ROOT / "outputs" / "sequences" / args.sequence
    threshold_sequence = args.threshold_sequence or args.sequence.replace(
        "eval_test_001", "eval_val_001"
    )
    threshold_path = PROJECT_ROOT / "outputs" / "sequences" / threshold_sequence / "calibrated_thresholds.json"
    threshold_data = json.loads(threshold_path.read_text())
    use_corridor = args.mode != "no_corridor"
    region = "driving_corridor" if use_corridor else "full_image"
    confidence_threshold = threshold_data[region]["threshold"]
    if confidence_threshold is None:
        raise RuntimeError("No calibrated confidence threshold available")
    persistence = 1 if args.mode == "no_persistence" else args.persistence_frames

    rows = list(csv.DictReader((sequence_dir / "temporal_uncertainty.csv").open()))
    decisions = []
    warning_run = 0
    severe_run = 0
    stable_run = 0
    previous_state = "Normal"

    for row in rows:
        confidence = float(row[
            "mean_msp_driving_corridor" if use_corridor else "mean_msp_full"
        ])
        low_fraction = float(row[
            "low_msp_fraction_driving_corridor" if use_corridor else "low_msp_fraction_full"
        ])
        instability = float(row[
            "prediction_instability_driving_corridor"
            if use_corridor else "prediction_instability_full"
        ] or 0.0)
        severe_evidence = (
            low_fraction >= args.severe_pixel_fraction_threshold
            or (
                confidence < 0.35
                and instability >= args.severe_instability_threshold
            )
        )
        warning_evidence = (
            confidence < confidence_threshold
            or low_fraction >= args.low_pixel_fraction_threshold
            or (
                args.mode not in ("confidence_only", "no_corridor")
                and instability >= args.instability_threshold
            )
        )
        warning_run = warning_run + 1 if warning_evidence else 0
        severe_run = severe_run + 1 if severe_evidence else 0
        stable_run = stable_run + 1 if not warning_evidence else 0

        if severe_run >= persistence:
            state = "Escalation"
            action = "request_human_supervisor"
        elif warning_run >= persistence:
            state = "Abstention"
            action = "withhold_decision"
        elif warning_evidence:
            state = "Caution"
            action = "continue_with_monitoring"
        elif previous_state in ("Abstention", "Escalation") and stable_run < args.recovery_frames:
            state = "Caution"
            action = "continue_with_monitoring"
        else:
            state = "Normal"
            action = "proceed"

        decisions.append({
            "index": int(row["index"]),
            "frame": row["frame"],
            "state": state,
            "action": action,
            "confidence": confidence,
            "low_pixel_fraction": low_fraction,
            "instability": instability,
            "warning_run": warning_run,
            "severe_run": severe_run,
            "stable_run": stable_run,
        })
        previous_state = state

    counts = Counter(row["state"] for row in decisions)
    supervisor_queue = [
        {
            "sequence": args.sequence,
            "frame": row["frame"],
            "reason": "persistent severe uncertainty",
            "confidence": row["confidence"],
            "low_pixel_fraction": row["low_pixel_fraction"],
            "instability": row["instability"],
            "status": "pending_review",
        }
        for row in decisions
        if row["state"] == "Escalation"
    ]
    summary = {
        "sequence": args.sequence,
        "threshold_sequence": threshold_sequence,
        "mode": args.mode,
        "region": region,
        "confidence_threshold": confidence_threshold,
        "persistence_frames": persistence,
        "recovery_frames": args.recovery_frames,
        "frames": len(decisions),
        "state_counts": {state: counts.get(state, 0) for state in STATES},
        "state_fractions": {state: counts.get(state, 0) / len(decisions) for state in STATES},
        "transition_count": sum(
            decisions[i]["state"] != decisions[i - 1]["state"]
            for i in range(1, len(decisions))
        ),
        "supervisor_queue_count": len(supervisor_queue),
    }
    csv_path = sequence_dir / f"policy_decisions_{args.mode}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=decisions[0].keys())
        writer.writeheader()
        writer.writerows(decisions)
    summary_path = sequence_dir / f"policy_summary_{args.mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    queue_path = sequence_dir / f"supervisor_queue_{args.mode}.json"
    queue_path.write_text(json.dumps(supervisor_queue, indent=2) + "\n")
    print(f"Policy CSV: {csv_path}")
    print(f"Policy summary: {summary_path}")
    print(f"Supervisor queue: {queue_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
