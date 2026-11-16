#!/usr/bin/env python3
"""Capture one RGB frame from a running CARLA server."""

from __future__ import annotations

import json
import queue
from pathlib import Path

import carla


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "frames"
OUTPUT_PATH = OUTPUT_DIR / "frame_000001.png"
METADATA_PATH = OUTPUT_DIR / "frame_000001.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle = None
    camera = None
    original_settings = world.get_settings()

    try:
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA returned no vehicle spawn points")

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
        if vehicle is None:
            raise RuntimeError("Could not spawn the vehicle at the first spawn point")

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "360")
        camera_bp.set_attribute("fov", "90")

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-8.0),
        )
        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle,
        )

        image_queue: queue.Queue[carla.Image] = queue.Queue()
        camera.listen(image_queue.put)

        synchronous_settings = world.get_settings()
        synchronous_settings.synchronous_mode = True
        synchronous_settings.fixed_delta_seconds = 0.05
        world.apply_settings(synchronous_settings)

        world.tick()
        image = image_queue.get(timeout=20.0)
        image.save_to_disk(str(OUTPUT_PATH))

        metadata = {
            "frame": image.frame,
            "timestamp": image.timestamp,
            "width": image.width,
            "height": image.height,
            "map": world.get_map().name,
            "output": str(OUTPUT_PATH),
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

        print(f"Saved RGB frame: {OUTPUT_PATH}")
        print(f"Saved metadata: {METADATA_PATH}")

    finally:
        if camera is not None:
            camera.stop()
        if camera is not None:
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
