#!/usr/bin/env python3
"""Audit experiment completeness without running GPU workloads."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SEQUENCES = PROJECT_ROOT / "outputs" / "sequences"
MODELS = {
    "segformer_b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "segformer_b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "segformer_b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    "mask2former": "facebook/mask2former-swin-small-cityscapes-semantic",
}


def count_frames(path: Path) -> int:
    return len(list(path.glob("frame_*.png")))


def model_status(sequence: str, model: str) -> dict:
    path = SEQUENCES / f"{sequence}_{model}"
    metadata_name = "mask2former_metadata.json" if model == "mask2former" else "segformer_metadata.json"
    metadata_path = path / metadata_name
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "frames_processed": metadata.get("frames_processed", 0),
        "logits": len(list((path / "logits").glob("*_logits.pt"))),
        "predictions": len(list((path / "predictions").glob("*.png"))),
        "uncertainty": len(list((path / "uncertainty").glob("*.pt"))),
        "temporal_summary": (path / "temporal_uncertainty_summary.json").exists(),
        "ground_truth_evaluation": (path / "ground_truth_evaluation.json").exists(),
    }


def main() -> None:
    audit = {
        "project": str(PROJECT_ROOT),
        "models": MODELS,
        "native_weather_scenarios": {},
        "stress_scenarios": {},
        "evaluation_splits": {},
        "model_outputs": {},
        "calibration": {},
        "event_response_count": len(list(SEQUENCES.glob("*/event_response.json"))),
        "policy_summary_count": len(list(SEQUENCES.glob("*/policy_summary_*.json"))),
    }

    for scenario in ("carla_wet_001", "carla_rain_001", "carla_fog_001", "carla_night_001"):
        path = SEQUENCES / scenario
        audit["native_weather_scenarios"][scenario] = {
            "exists": path.exists(),
            "rgb_frames": count_frames(path),
            "ground_truth_frames": count_frames(path / "ground_truth"),
            "model_outputs": {
                model: model_status(scenario, model) for model in MODELS
            },
        }

    for split in ("eval_normal_001", "eval_val_001", "eval_test_001"):
        path = SEQUENCES / split
        audit["evaluation_splits"][split] = {
            "rgb_frames": count_frames(path),
            "ground_truth_frames": count_frames(path / "ground_truth"),
            "metadata": (path / "sequence_metadata.json").exists(),
        }
        audit["model_outputs"][split] = {
            model: model_status(split, model) for model in MODELS
        }

    for scenario in ("occlusion_001", "unknown_object_001"):
        path = SEQUENCES / scenario
        audit["stress_scenarios"][scenario] = {
            "exists": path.exists(),
            "rgb_frames": count_frames(path),
            "ground_truth_frames": count_frames(path / "ground_truth"),
            "model_outputs": {
                model: model_status(scenario, model) for model in MODELS
            },
        }

    for model in MODELS:
        path = SEQUENCES / f"eval_val_001_{model}" / "calibrated_thresholds.json"
        audit["calibration"][model] = json.loads(path.read_text()) if path.exists() else None

    output_path = PROJECT_ROOT / "experiment_audit.json"
    output_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Audit: {output_path}")
    for split, data in audit["evaluation_splits"].items():
        print(split, data["rgb_frames"], "RGB /", data["ground_truth_frames"], "GT")
    print("Native weather scenarios present:", sum(x["exists"] for x in audit["native_weather_scenarios"].values()), "/ 4")
    print("Stress scenarios present:", sum(x["exists"] for x in audit["stress_scenarios"].values()), "/ 2")
    print("Event reports:", audit["event_response_count"])
    print("Policy summaries:", audit["policy_summary_count"])


if __name__ == "__main__":
    main()
