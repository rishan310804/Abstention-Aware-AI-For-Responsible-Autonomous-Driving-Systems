#!/usr/bin/env python3
"""Capture a CARLA sequence with an independent physical obstacle in-lane."""

from __future__ import annotations

import argparse
import json
import math
import queue
from pathlib import Path

import carla  # type: ignore

from capture_sequence import transform_to_dict, weather_from_preset

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="physical_debris_001")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=411)
    parser.add_argument("--spawn-index", type=int, default=20)
    parser.add_argument("--weather-preset", default="clear")
    parser.add_argument("--port", type=int, default=2200)
    parser.add_argument("--tm-port", type=int, default=8260)
    parser.add_argument("--hazard-distance-m", type=float, default=12.0)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "outputs" / "sequences" / args.scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    original_settings = world.get_settings()
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_random_device_seed(args.seed)
    tm.set_synchronous_mode(True)
    vehicle = camera = semantic_camera = hazard = collision_sensor = None
    rgb_queue: queue.Queue = queue.Queue()
    gt_queue: queue.Queue = queue.Queue()
    collision_queue: queue.Queue = queue.Queue()
    collisions = []
    try:
        world.set_weather(weather_from_preset(args.weather_preset))
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        blueprints = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(
            blueprints.filter("vehicle.tesla.model3")[0],
            spawn_points[args.spawn_index % len(spawn_points)],
        )
        if vehicle is None:
            raise RuntimeError("Could not spawn ego vehicle")
        vehicle.set_autopilot(True, args.tm_port)
        tm.vehicle_percentage_speed_difference(vehicle, 0.0)

        hazard_bp = None
        for name in ("static.prop.container", "static.prop.box01", "static.prop.barrel"):
            matches = blueprints.filter(name)
            if matches:
                hazard_bp = matches[0]
                break
        if hazard_bp is None:
            raise RuntimeError("No independent hazard blueprint found")
        # Use the selected map spawn transform directly.  Immediately after
        # spawning, CARLA may report a not-yet-ticked actor transform.
        ego_transform = spawn_points[args.spawn_index % len(spawn_points)]
        forward = ego_transform.get_forward_vector()
        hazard_transform = carla.Transform(
            carla.Location(
                x=ego_transform.location.x + forward.x * args.hazard_distance_m,
                y=ego_transform.location.y + forward.y * args.hazard_distance_m,
                z=ego_transform.location.z + 0.6,
            ),
            carla.Rotation(yaw=ego_transform.rotation.yaw),
        )
        # Preserve the absolute transform before CARLA receives the object.
        # The Python API may mutate/reuse transform objects during spawning.
        hazard_metadata_transform = transform_to_dict(hazard_transform)
        hazard = world.spawn_actor(hazard_bp, hazard_transform)
        if hazard is None:
            raise RuntimeError("Could not spawn independent hazard")
        hazard.set_simulate_physics(False)

        collision_sensor = world.spawn_actor(
            blueprints.find("sensor.other.collision"), carla.Transform(), attach_to=vehicle
        )
        collision_sensor.listen(collision_queue.put)
        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "360")
        camera_bp.set_attribute("fov", "90")
        camera = world.spawn_actor(
            camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-8.0)), attach_to=vehicle
        )
        camera.listen(rgb_queue.put)
        gt_bp = blueprints.find("sensor.camera.semantic_segmentation")
        gt_bp.set_attribute("image_size_x", "640")
        gt_bp.set_attribute("image_size_y", "360")
        gt_bp.set_attribute("fov", "90")
        semantic_camera = world.spawn_actor(
            gt_bp, carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-8.0)), attach_to=vehicle
        )
        semantic_camera.listen(gt_queue.put)

        records = []
        for index in range(args.frames):
            world.tick()
            image = rgb_queue.get(timeout=20.0)
            gt = gt_queue.get(timeout=20.0)
            name = f"frame_{index:06d}.png"
            image.save_to_disk(str(output_dir / name))
            gt_path = output_dir / "ground_truth" / name
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt.save_to_disk(str(gt_path))
            frame_collisions = []
            while True:
                try:
                    event = collision_queue.get_nowait()
                except queue.Empty:
                    break
                frame_collisions.append(event.other_actor.type_id)
                collisions.append(index)
            records.append({
                "index": index,
                "carla_frame": image.frame,
                "timestamp": image.timestamp,
                "image": str(output_dir / name),
                "ground_truth": str(gt_path),
                "vehicle_transform": transform_to_dict(vehicle.get_transform()),
                "collision_events": frame_collisions,
            })

        metadata = {
            "sequence": args.scenario,
            "frames": args.frames,
            "map": world.get_map().name,
            "image_width": 640,
            "image_height": 360,
            "fixed_delta_seconds": 0.1,
            "traffic_manager_port": args.tm_port,
            "seed": args.seed,
            "spawn_index": args.spawn_index,
            "weather_preset": args.weather_preset,
            "with_ground_truth": True,
            "ground_truth_definition": "raw CARLA semantic tag IDs in the red channel",
            "independent_hazard": {
                "blueprint": hazard_bp.id,
                "world_transform": hazard_metadata_transform,
                "distance_from_initial_ego_m": args.hazard_distance_m,
            },
            "capture_collision_frames": collisions,
            "records": records,
        }
        (output_dir / "sequence_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({"scenario": args.scenario, "capture_collisions": collisions}, indent=2))
    finally:
        tm.set_synchronous_mode(False)
        for actor in (collision_sensor, camera, semantic_camera, hazard, vehicle):
            if actor is not None:
                try:
                    actor.stop()
                except (AttributeError, RuntimeError):
                    pass
                try:
                    actor.destroy()
                except RuntimeError:
                    pass
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
