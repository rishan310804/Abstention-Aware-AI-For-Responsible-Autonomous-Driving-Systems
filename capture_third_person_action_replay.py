#!/usr/bin/env python3
"""Capture a third-person CARLA view while replaying the action policy."""

from __future__ import annotations

import argparse
import json
import queue
from pathlib import Path

import carla  # type: ignore

from capture_sequence import weather_from_preset
from replay_action_policy_in_carla import load_plan, scenario_metadata, transform_from_dict

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="physical_debris_001_segformer_b0")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--port", type=int, default=2200)
    parser.add_argument("--tm-port", type=int, default=8260)
    parser.add_argument("--output", default="third_person_replay")
    args = parser.parse_args()

    plan, _ = load_plan(args.sequence, args.mode)
    metadata = scenario_metadata(args.sequence, "physical_debris_001")
    output = PROJECT_ROOT / "outputs" / "visualizations" / args.output
    output.mkdir(parents=True, exist_ok=True)
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    original = world.get_settings()
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_random_device_seed(int(metadata["seed"]))
    tm.set_synchronous_mode(True)
    vehicle = hazard = camera = None
    image_queue: queue.Queue = queue.Queue()
    fallback = False
    autopilot = False
    try:
        world.set_weather(weather_from_preset(metadata["weather_preset"]))
        settings = world.get_settings(); settings.synchronous_mode = True; settings.fixed_delta_seconds = .1; world.apply_settings(settings)
        blueprints = world.get_blueprint_library(); spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(blueprints.filter("vehicle.tesla.model3")[0], spawn_points[int(metadata["spawn_index"]) % len(spawn_points)])
        if vehicle is None: raise RuntimeError("Could not spawn ego vehicle")
        independent = metadata["independent_hazard"]
        hazard = world.spawn_actor(blueprints.find(independent["blueprint"]), transform_from_dict(independent["world_transform"]))
        if hazard is None: raise RuntimeError("Could not spawn debris")
        hazard.set_simulate_physics(False)
        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "960"); camera_bp.set_attribute("image_size_y", "540"); camera_bp.set_attribute("fov", "90")
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=-8.0, z=4.2), carla.Rotation(pitch=-16.0)),
            attach_to=vehicle,
        )
        camera.listen(image_queue.put)
        for row in plan:
            action = row["action"]
            if action in ("controlled_stop", "emergency_stop_and_supervise"):
                fallback = True
            if fallback:
                if autopilot:
                    vehicle.set_autopilot(False, args.tm_port); autopilot = False
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            else:
                if not autopilot:
                    vehicle.set_autopilot(True, args.tm_port); autopilot = True
                tm.vehicle_percentage_speed_difference(vehicle, 40.0 if action == "cautious_drive" else 0.0)
            world.tick(); image = image_queue.get(timeout=20.0)
            image.save_to_disk(str(output / f"frame_{int(row['index']):06d}.png"))
        (output / "metadata.json").write_text(json.dumps({"sequence": args.sequence, "camera": "third_person", "frames": len(plan)}, indent=2) + "\n")
        print(f"Third-person frames: {output}")
    finally:
        tm.set_synchronous_mode(False)
        for actor in (camera, hazard, vehicle):
            if actor is not None:
                try: actor.stop()
                except (AttributeError, RuntimeError): pass
                try: actor.destroy()
                except RuntimeError: pass
        world.apply_settings(original)


if __name__ == "__main__":
    main()
