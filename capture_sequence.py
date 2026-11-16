#!/usr/bin/env python3
"""Capture a reproducible RGB sequence from CARLA in synchronous mode."""

from __future__ import annotations

import argparse
import json
import queue
from pathlib import Path

import carla


PROJECT_ROOT = Path(__file__).resolve().parent


def weather_from_preset(name: str) -> carla.WeatherParameters:
    if name == "clear":
        return carla.WeatherParameters.ClearNoon
    if name == "cloudy":
        return carla.WeatherParameters.CloudyNoon
    if name == "wet":
        return carla.WeatherParameters.WetNoon
    if name == "rain":
        return carla.WeatherParameters.HardRainNoon
    if name == "fog":
        return carla.WeatherParameters(
            cloudiness=70.0,
            precipitation=20.0,
            precipitation_deposits=30.0,
            wind_intensity=10.0,
            fog_density=70.0,
            fog_distance=15.0,
            wetness=70.0,
            sun_altitude_angle=35.0,
        )
    if name == "night":
        return carla.WeatherParameters(
            cloudiness=40.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=5.0,
            fog_density=0.0,
            fog_distance=0.0,
            wetness=0.0,
            sun_altitude_angle=-35.0,
        )
    raise ValueError(f"Unknown weather preset: {name}")


def transform_to_dict(transform: carla.Transform) -> dict[str, dict[str, float]]:
    return {
        "location": {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
        },
        "rotation": {
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--sequence", default="normal_001")
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument(
        "--weather-preset",
        choices=("clear", "cloudy", "wet", "rain", "fog", "night"),
        default="clear",
    )
    parser.add_argument(
        "--with-ground-truth",
        action="store_true",
        help="Also save CARLA semantic-segmentation ground-truth images",
    )
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.spawn_index < 0:
        raise ValueError("--spawn-index must be non-negative")

    output_dir = PROJECT_ROOT / "outputs" / "sequences" / args.sequence
    output_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    world.set_weather(weather_from_preset(args.weather_preset))
    blueprints = world.get_blueprint_library()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(args.seed)

    vehicle = None
    camera = None
    semantic_camera = None
    original_settings = world.get_settings()
    camera_queue: queue.Queue[carla.Image] = queue.Queue()
    semantic_queue: queue.Queue[carla.Image] = queue.Queue()

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA returned no spawn points")

        spawn_point = spawn_points[args.spawn_index % len(spawn_points)]
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            raise RuntimeError("Could not spawn the ego vehicle")

        vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.vehicle_percentage_speed_difference(vehicle, 0.0)

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "360")
        camera_bp.set_attribute("fov", "90")

        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(
                carla.Location(x=1.5, z=2.4),
                carla.Rotation(pitch=-8.0),
            ),
            attach_to=vehicle,
        )
        camera.listen(camera_queue.put)

        if args.with_ground_truth:
            semantic_camera_bp = blueprints.find("sensor.camera.semantic_segmentation")
            semantic_camera_bp.set_attribute("image_size_x", "640")
            semantic_camera_bp.set_attribute("image_size_y", "360")
            semantic_camera_bp.set_attribute("fov", "90")
            semantic_camera = world.spawn_actor(
                semantic_camera_bp,
                carla.Transform(
                    carla.Location(x=1.5, z=2.4),
                    carla.Rotation(pitch=-8.0),
                ),
                attach_to=vehicle,
            )
            semantic_camera.listen(semantic_queue.put)

        records = []
        for index in range(args.frames):
            world.tick()
            image = camera_queue.get(timeout=20.0)
            semantic_image = (
                semantic_queue.get(timeout=20.0) if args.with_ground_truth else None
            )
            frame_name = f"frame_{index:06d}.png"
            image_path = output_dir / frame_name
            image.save_to_disk(str(image_path))

            ground_truth_path = None
            if semantic_image is not None:
                ground_truth_path = output_dir / "ground_truth" / frame_name
                ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
                # Save raw semantic tags. The red channel contains the CARLA
                # semantic tag ID; no Cityscapes color conversion is applied.
                semantic_image.save_to_disk(str(ground_truth_path))

            records.append(
                {
                    "index": index,
                    "carla_frame": image.frame,
                    "timestamp": image.timestamp,
                    "image": str(image_path),
                    "ground_truth": str(ground_truth_path)
                    if ground_truth_path is not None
                    else None,
                    "vehicle_transform": transform_to_dict(vehicle.get_transform()),
                }
            )

        metadata = {
            "sequence": args.sequence,
            "frames": args.frames,
            "map": world.get_map().name,
            "image_width": 640,
            "image_height": 360,
            "fixed_delta_seconds": 0.1,
            "traffic_manager_port": args.tm_port,
            "seed": args.seed,
            "spawn_index": args.spawn_index,
            "weather_preset": args.weather_preset,
            "with_ground_truth": args.with_ground_truth,
            "ground_truth_definition": (
                "raw CARLA semantic tag IDs in the red channel"
                if args.with_ground_truth
                else None
            ),
            "records": records,
        }
        (output_dir / "sequence_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        print(f"Captured {args.frames} frames")
        print(f"Sequence directory: {output_dir}")
        print(f"Metadata: {output_dir / 'sequence_metadata.json'}")

    finally:
        # Stop Traffic Manager synchronization before destroying its vehicle.
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass

        if vehicle is not None:
            try:
                vehicle.set_autopilot(False, traffic_manager.get_port())
            except RuntimeError:
                pass
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass
            try:
                camera.destroy()
            except RuntimeError:
                pass
        if semantic_camera is not None:
            try:
                semantic_camera.stop()
            except RuntimeError:
                pass
            try:
                semantic_camera.destroy()
            except RuntimeError:
                pass
        if vehicle is not None:
            try:
                vehicle.destroy()
            except RuntimeError:
                pass
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
