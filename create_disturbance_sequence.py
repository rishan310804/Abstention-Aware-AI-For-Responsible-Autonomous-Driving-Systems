#!/usr/bin/env python3
"""Create deterministic image-corruption sequences from a CARLA sequence."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="normal_001")
    parser.add_argument("--scenario", default="blur_001")
    parser.add_argument(
        "--kind",
        choices=("blur", "noise", "exposure", "occlusion", "unknown_object"),
        default="blur",
    )
    parser.add_argument("--severity", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--occlusion-x", type=int, default=220)
    parser.add_argument("--occlusion-y", type=int, default=110)
    parser.add_argument("--occlusion-width", type=int, default=200)
    parser.add_argument("--occlusion-height", type=int, default=180)
    args = parser.parse_args()

    source_dir = PROJECT_ROOT / "outputs" / "sequences" / args.source
    output_dir = PROJECT_ROOT / "outputs" / "sequences" / args.scenario
    frame_paths = sorted(source_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No source frames found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    end_frame = len(frame_paths) - 1 if args.end_frame is None else args.end_frame
    if end_frame < args.start_frame or end_frame >= len(frame_paths):
        raise ValueError(
            f"Invalid disturbance window: {args.start_frame}..{end_frame} "
            f"for {len(frame_paths)} frames"
        )

    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        frame_number = int(frame_path.stem.split("_")[-1])

        if not args.start_frame <= frame_number <= end_frame:
            image.save(output_dir / frame_path.name)
            continue

        if args.kind == "blur":
            transformed = image.filter(ImageFilter.GaussianBlur(radius=args.severity))
        elif args.kind == "noise":
            array = np.asarray(image).astype(np.float32)
            noise = rng.normal(0.0, args.severity, array.shape)
            transformed = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
        elif args.kind == "exposure":
            transformed = ImageEnhance.Brightness(image).enhance(args.severity)
        elif args.kind == "occlusion":
            array = np.asarray(image).copy()
            height, width = array.shape[:2]
            x0 = max(0, min(args.occlusion_x, width))
            y0 = max(0, min(args.occlusion_y, height))
            x1 = max(x0, min(args.occlusion_x + args.occlusion_width, width))
            y1 = max(y0, min(args.occlusion_y + args.occlusion_height, height))
            array[y0:y1, x0:x1] = np.array([20, 20, 20], dtype=np.uint8)
            transformed = Image.fromarray(array)
        else:
            transformed = image.copy()
            draw = ImageDraw.Draw(transformed)
            x0 = args.occlusion_x
            y0 = args.occlusion_y
            x1 = x0 + args.occlusion_width
            y1 = y0 + args.occlusion_height
            draw.rounded_rectangle((x0, y0, x1, y1), radius=20, fill=(220, 30, 180))
            draw.rectangle((x0 + 25, y0 + 25, x1 - 25, y1 - 25), outline=(20, 240, 240), width=8)
            draw.ellipse((x0 + 55, y0 + 45, x0 + 105, y0 + 95), fill=(240, 240, 30))

        transformed.save(output_dir / frame_path.name)

    metadata = {
        "source_sequence": args.source,
        "scenario": args.scenario,
        "kind": args.kind,
        "severity": args.severity,
        "seed": args.seed,
        "start_frame": args.start_frame,
        "end_frame": end_frame,
        "clean_before_and_after_window": True,
        "occlusion_box": (
            {
                "x": args.occlusion_x,
                "y": args.occlusion_y,
                "width": args.occlusion_width,
                "height": args.occlusion_height,
            }
            if args.kind in ("occlusion", "unknown_object")
            else None
        ),
        "frames": len(frame_paths),
        "source_is_unchanged": True,
    }
    (output_dir / "scenario_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    source_ground_truth = source_dir / "ground_truth"
    if source_ground_truth.exists():
        output_ground_truth = output_dir / "ground_truth"
        output_ground_truth.mkdir(parents=True, exist_ok=True)
        for ground_truth_path in source_ground_truth.glob("frame_*.png"):
            shutil.copy2(ground_truth_path, output_ground_truth / ground_truth_path.name)
        metadata["ground_truth_copied_from"] = str(source_ground_truth)
        (output_dir / "scenario_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    print(f"Created {len(frame_paths)} frames")
    print(f"Scenario directory: {output_dir}")
    print(f"Metadata: {output_dir / 'scenario_metadata.json'}")


if __name__ == "__main__":
    main()
