#!/usr/bin/env python3
"""Translate uncertainty-policy states into explicit driving actions.

This is the no-RL action layer for the project.  It does not invent a learned
driving policy.  Instead, it turns the already validated four-state policy
evidence into a conservative, auditable action plan that can later be connected
to a CARLA vehicle controller.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

ACTION_DEFINITIONS = {
    "Normal": {
        "action": "nominal_drive",
        "control_mode": "nominal_controller",
        "speed_scale": 1.0,
        "brake_command": 0.0,
        "supervisor_required": False,
    },
    "Caution": {
        "action": "cautious_drive",
        "control_mode": "nominal_controller_speed_limited",
        "speed_scale": 0.60,
        "brake_command": 0.0,
        "supervisor_required": False,
    },
    "Abstention": {
        "action": "controlled_stop",
        "control_mode": "fallback_controller",
        "speed_scale": 0.0,
        "brake_command": 1.0,
        "supervisor_required": False,
    },
    "Escalation": {
        "action": "emergency_stop_and_supervise",
        "control_mode": "fallback_controller_supervisor",
        "speed_scale": 0.0,
        "brake_command": 1.0,
        "supervisor_required": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--mode", default="full")
    parser.add_argument("--nominal-speed-mps", type=float, default=8.0)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    sequence_dir = PROJECT_ROOT / "outputs" / "sequences" / args.sequence
    input_path = sequence_dir / f"policy_decisions_{args.mode}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Policy decisions not found: {input_path}")

    decisions = []
    previous_state = None
    with input_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            state = row["state"]
            if state not in ACTION_DEFINITIONS:
                raise ValueError(f"Unknown policy state: {state}")
            definition = ACTION_DEFINITIONS[state]
            decisions.append({
                **row,
                "action": definition["action"],
                "control_mode": definition["control_mode"],
                "target_speed_mps": round(
                    args.nominal_speed_mps * definition["speed_scale"], 4
                ),
                "speed_scale": definition["speed_scale"],
                "brake_command": definition["brake_command"],
                "supervisor_required": definition["supervisor_required"],
                "state_transition": previous_state is not None and state != previous_state,
                "previous_state": previous_state or "",
                "action_policy_note": {
                    "Normal": "trust nominal driving policy",
                    "Caution": "continue with speed limitation and monitoring",
                    "Abstention": "do not rely on uncertain perception decision; brake to safe stop",
                    "Escalation": "maintain safe stop and request supervisory intervention",
                }[state],
            })
            previous_state = state

    counts = Counter(row["state"] for row in decisions)
    transitions = sum(row["state_transition"] for row in decisions)
    summary = {
        "sequence": args.sequence,
        "policy_mode": args.mode,
        "nominal_speed_mps": args.nominal_speed_mps,
        "frames": len(decisions),
        "state_counts": {state: counts.get(state, 0) for state in ACTION_DEFINITIONS},
        "action_counts": dict(Counter(row["action"] for row in decisions)),
        "state_transitions": transitions,
        "controlled_stop_frames": sum(
            row["action"] == "controlled_stop" for row in decisions
        ),
        "supervisor_required_frames": sum(
            row["supervisor_required"] for row in decisions
        ),
        "closed_loop_control_executed": False,
        "interpretation": (
            "Offline action plan generated from uncertainty-policy evidence. "
            "The plan is not a physical CARLA control execution until connected "
            "to a vehicle controller."
        ),
    }

    output_stem = args.output_name or f"action_policy_{args.mode}"
    decisions_path = sequence_dir / f"{output_stem}.csv"
    summary_path = sequence_dir / f"{output_stem}_summary.json"
    with decisions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=decisions[0].keys())
        writer.writeheader()
        writer.writerows(decisions)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Action plan: {decisions_path}")
    print(f"Action summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
