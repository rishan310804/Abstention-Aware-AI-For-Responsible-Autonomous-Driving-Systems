#!/usr/bin/env python3
"""Execute a saved uncertainty action plan in CARLA and measure outcomes.

The action plan is generated from recorded perception evidence.  This script
replays those decisions in the matching CARLA scenario and physically applies
nominal, speed-limited, or braking control to the ego vehicle.  It is an
action-policy evaluation, not a claim of live end-to-end neural inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import carla  # type: ignore

from capture_sequence import weather_from_preset


def speed_mps(vehicle: carla.Vehicle) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def transform_from_dict(payload: dict) -> carla.Transform:
    location = payload["location"]
    rotation = payload["rotation"]
    return carla.Transform(
        carla.Location(**location),
        carla.Rotation(**rotation),
    )


def load_plan(sequence: str, mode: str) -> tuple[list[dict], Path]:
    sequence_dir = PROJECT_ROOT / "outputs" / "sequences" / sequence
    path = sequence_dir / f"action_policy_{mode}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Action plan not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Action plan is empty: {path}")
    return rows, sequence_dir


def scenario_metadata(sequence: str, explicit: str | None) -> dict:
    source = explicit or sequence
    path = PROJECT_ROOT / "outputs" / "sequences" / source / "sequence_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario metadata not found: {path}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="Model/action-plan sequence")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--scenario-sequence", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--stop-speed-mps", type=float, default=0.15)
    parser.add_argument("--max-wait-after-plan", type=int, default=30)
    parser.add_argument(
        "--baseline",
        choices=("policy", "nominal"),
        default="policy",
        help="Use the saved uncertainty policy or ungated nominal driving",
    )
    args = parser.parse_args()

    plan, output_dir = load_plan(args.sequence, args.mode)
    metadata = scenario_metadata(args.sequence, args.scenario_sequence)
    tm_port = args.tm_port or int(metadata.get("traffic_manager_port", 8060))
    seed = int(metadata.get("seed", 42))
    spawn_index = int(metadata.get("spawn_index", 0))
    weather = metadata.get("weather_preset", "clear")
    delta = float(metadata.get("fixed_delta_seconds", 0.1))

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_random_device_seed(seed)
    traffic_manager.set_synchronous_mode(True)

    vehicle = None
    stress_prop = None
    independent_hazard = None
    collision_sensor = None
    collision_queue: queue.Queue = queue.Queue()
    records: list[dict] = []
    collision_frames: list[int] = []
    stop_start_time = None
    stop_time = None
    braking_start_location = None
    stop_location = None
    previous_action = None
    autopilot_enabled = False
    fallback_latched = False

    try:
        world.set_weather(weather_from_preset(weather))
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta
        world.apply_settings(settings)

        blueprints = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA returned no spawn points")
        vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]
        vehicle = world.try_spawn_actor(
            vehicle_bp, spawn_points[spawn_index % len(spawn_points)]
        )
        if vehicle is None:
            raise RuntimeError("Could not spawn ego vehicle")

        physical_stress = metadata.get("physical_stress")
        if physical_stress:
            prop_bp = blueprints.find(physical_stress["prop_blueprint"])
            stress_prop = world.spawn_actor(
                prop_bp,
                transform_from_dict(physical_stress["attached_transform"]),
                attach_to=vehicle,
            )
            if stress_prop is None:
                raise RuntimeError("Could not recreate physical stress prop")
            stress_prop.set_simulate_physics(False)
        independent = metadata.get("independent_hazard")
        if independent:
            hazard_bp = blueprints.find(independent["blueprint"])
            independent_hazard = world.spawn_actor(
                hazard_bp,
                transform_from_dict(independent["world_transform"]),
            )
            if independent_hazard is None:
                raise RuntimeError("Could not recreate independent hazard")
            independent_hazard.set_simulate_physics(False)

        collision_bp = blueprints.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=vehicle
        )
        collision_sensor.listen(collision_queue.put)

        for row in plan:
            index = int(row["index"])
            action = "nominal_drive" if args.baseline == "nominal" else row["action"]
            state = "Normal" if args.baseline == "nominal" else row["state"]
            is_braking = action in ("controlled_stop", "emergency_stop_and_supervise")
            if is_braking:
                fallback_latched = True
            # A safety fallback is absorbing during this replay.  Once the
            # vehicle has been commanded to abstain, later Normal/Caution
            # evidence cannot immediately re-enable motion before standstill.
            effective_action = "controlled_stop" if fallback_latched else action
            is_braking = fallback_latched
            is_cautious = effective_action == "cautious_drive"

            if is_braking:
                if autopilot_enabled:
                    vehicle.set_autopilot(False, tm_port)
                    autopilot_enabled = False
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                if stop_start_time is None:
                    stop_start_time = float(index * delta)
                    braking_start_location = vehicle.get_location()
            else:
                if not autopilot_enabled:
                    vehicle.set_autopilot(True, tm_port)
                    autopilot_enabled = True
                # Traffic Manager speed difference is relative to its nominal
                # target.  40% reduction approximates the action-policy 0.60 scale.
                traffic_manager.vehicle_percentage_speed_difference(
                    vehicle, 40.0 if is_cautious else 0.0
                )

            world.tick()
            current_speed = speed_mps(vehicle)
            if stop_start_time is not None and stop_time is None and current_speed <= args.stop_speed_mps:
                stop_time = float(index * delta)
                stop_location = vehicle.get_location()

            frame_collisions = []
            while True:
                try:
                    event = collision_queue.get_nowait()
                except queue.Empty:
                    break
                frame_collisions.append({
                    "other_actor": event.other_actor.type_id,
                    "intensity": {
                        "x": event.normal_impulse.x,
                        "y": event.normal_impulse.y,
                        "z": event.normal_impulse.z,
                    },
                })
                collision_frames.append(index)

            records.append({
                "index": index,
                "state": state,
                "action": action,
                "effective_action": effective_action,
                "control_executed": "brake" if is_braking else "traffic_manager",
                "autopilot_enabled": autopilot_enabled,
                "speed_mps": current_speed,
                "actual_throttle": vehicle.get_control().throttle,
                "actual_brake": vehicle.get_control().brake,
                "actual_steer": vehicle.get_control().steer,
                "actual_hand_brake": vehicle.get_control().hand_brake,
                "collision_events": frame_collisions,
                "state_transition": previous_action is not None and action != previous_action,
            })
            previous_action = action

        # Continue braking after the plan until the vehicle is stationary or
        # the bounded post-plan window is exhausted.
        if stop_start_time is not None and stop_time is None:
            if autopilot_enabled:
                vehicle.set_autopilot(False, tm_port)
                autopilot_enabled = False
            for offset in range(args.max_wait_after_plan):
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                world.tick()
                current_speed = speed_mps(vehicle)
                if stop_time is None and current_speed <= args.stop_speed_mps:
                    stop_time = float((len(plan) + offset + 1) * delta)
                    stop_location = vehicle.get_location()
                    break

        summary = {
            "sequence": args.sequence,
            "scenario_sequence": args.scenario_sequence or args.sequence,
            "policy_mode": args.mode,
            "baseline": args.baseline,
            "frames_executed": len(records),
            "state_counts": dict(Counter(row["state"] for row in records)),
            "action_counts": dict(Counter(row["action"] for row in records)),
            "effective_action_counts": dict(
                Counter(row["effective_action"] for row in records)
            ),
            "state_transitions": sum(row["state_transition"] for row in records),
            "collision_count": len(collision_frames),
            "collision_frames": collision_frames,
            "stop_requested": stop_start_time is not None,
            "stop_time_seconds": stop_time,
            "stopping_time_seconds": (
                stop_time - stop_start_time
                if stop_time is not None and stop_start_time is not None
                else None
            ),
            "stopping_distance_m": (
                math.sqrt(
                    (stop_location.x - braking_start_location.x) ** 2
                    + (stop_location.y - braking_start_location.y) ** 2
                    + (stop_location.z - braking_start_location.z) ** 2
                )
                if stop_location is not None and braking_start_location is not None
                else None
            ),
            "maximum_actual_throttle": max(row["actual_throttle"] for row in records),
            "maximum_actual_brake": max(row["actual_brake"] for row in records),
            "maximum_absolute_actual_steer": max(
                abs(row["actual_steer"]) for row in records
            ),
            "minimum_speed_mps": min(row["speed_mps"] for row in records),
            "physical_control_executed": True,
            "fallback_latched": any(
                row["effective_action"] == "controlled_stop" for row in records
            ),
            "control_scope": "replayed saved action plan in synchronized CARLA",
            "live_neural_inference": False,
        }
        stem = f"carla_action_execution_{args.mode}"
        if args.baseline == "nominal":
            stem += "_nominal_baseline"
        (output_dir / f"{stem}.json").write_text(
            json.dumps({"summary": summary, "records": records}, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2))

    finally:
        if collision_sensor is not None:
            collision_sensor.stop()
            collision_sensor.destroy()
        if stress_prop is not None:
            stress_prop.destroy()
        if independent_hazard is not None:
            independent_hazard.destroy()
        if vehicle is not None:
            try:
                vehicle.set_autopilot(False, tm_port)
            except RuntimeError:
                pass
            vehicle.destroy()
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
