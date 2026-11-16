#!/usr/bin/env python3
"""Capture physically rendered CARLA occlusion and unfamiliar-prop scenarios."""

from __future__ import annotations

import argparse
import json
import queue
from pathlib import Path

import carla  # type: ignore

from capture_sequence import transform_to_dict, weather_from_preset

PROJECT_ROOT = Path(__file__).resolve().parent


def choose_prop(blueprints: carla.BlueprintLibrary, kind: str) -> carla.ActorBlueprint:
    candidates = (
        ("static.prop.container", "static.prop.box01", "static.prop.box02")
        if kind == "occlusion"
        else ("static.prop.barrel", "static.prop.box01", "static.prop.container")
    )
    for name in candidates:
        matches = blueprints.filter(name)
        if matches:
            return matches[0]
    raise RuntimeError(f"No suitable physical prop blueprint found for {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--kind", choices=("occlusion", "unknown_object"), required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--spawn-index", type=int, default=20)
    parser.add_argument("--weather-preset", default="clear")
    parser.add_argument("--tm-port", type=int, default=8260)
    parser.add_argument("--port", type=int, default=2200)
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
    vehicle = camera = semantic_camera = stress_prop = None
    rgb_queue: queue.Queue = queue.Queue()
    gt_queue: queue.Queue = queue.Queue()
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

        prop_bp = choose_prop(blueprints, args.kind)
        if prop_bp.has_attribute("role_name"):
            prop_bp.set_attribute("role_name", f"physical_{args.kind}")
        # The prop is physically rendered and moves with the ego vehicle so
        # the stress condition remains reproducible during the capture.
        prop_transform = (
            carla.Transform(carla.Location(x=3.2, z=1.8), carla.Rotation())
            if args.kind == "occlusion"
            else carla.Transform(carla.Location(x=7.0, z=0.7), carla.Rotation())
        )
        stress_prop = world.spawn_actor(prop_bp, prop_transform, attach_to=vehicle)
        if stress_prop is None:
            raise RuntimeError("Could not spawn physical stress prop")
        stress_prop.set_simulate_physics(False)

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
            records.append({
                "index": index,
                "carla_frame": image.frame,
                "timestamp": image.timestamp,
                "image": str(output_dir / name),
                "ground_truth": str(gt_path),
                "vehicle_transform": transform_to_dict(vehicle.get_transform()),
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
            "physical_stress": {
                "kind": args.kind,
                "prop_blueprint": prop_bp.id,
                "attached_transform": transform_to_dict(prop_transform),
            },
            "records": records,
        }
        (output_dir / "sequence_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Captured physical {args.kind} scenario: {output_dir}")
    finally:
        tm.set_synchronous_mode(False)
        for actor in (camera, semantic_camera, stress_prop, vehicle):
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
