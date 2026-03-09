"""
Entry point for the Autonomous Driving System.

Connects to a running CARLA 0.9.14 server, spawns the ego vehicle with the
full sensor suite, and runs the AutonomousAgent control loop until the
destination is reached or the user presses Ctrl+C.

Usage:
    # 1. Start CARLA: ./CarlaUE4.sh (Linux) or CarlaUE4.exe (Windows)
    # 2. Run the agent:
    python main.py [--config config/config.yaml] [--no-hud] [--steps N]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import random
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Driving Agent in CARLA 0.9.14")
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--no-hud", action="store_true",
        help="Disable the pygame visualisation window",
    )
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Maximum number of simulation steps (0 = unlimited)",
    )
    parser.add_argument(
        "--map", default=None,
        help="Override the CARLA map (e.g. Town01, Town02)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    carla_cfg = cfg.get("carla", {})
    ego_cfg   = cfg.get("ego", {})

    if args.map:
        carla_cfg["map"] = args.map

    try:
        import carla
    except ImportError:
        print(
            "ERROR: The CARLA Python API is not installed.\n"
            "Download CARLA 0.9.14 from https://github.com/carla-simulator/carla/releases/tag/0.9.14\n"
            "Then install the .egg or .whl file:\n"
            "  pip install <path>/carla-0.9.14-*.whl"
        )
        sys.exit(1)

    client = carla.Client(carla_cfg.get("host", "localhost"), carla_cfg.get("port", 2000))
    client.set_timeout(carla_cfg.get("timeout", 10.0))

    print(f"[Main] Connecting to CARLA server at "
          f"{carla_cfg['host']}:{carla_cfg['port']} ...")
    world = client.load_world(carla_cfg.get("map", "Town02"))
    print(f"[Main] Loaded map: {world.get_map().name}")

    # Apply synchronous mode settings
    settings = world.get_settings()
    settings.synchronous_mode = carla_cfg.get("synchronous", True)
    settings.fixed_delta_seconds = carla_cfg.get("fixed_delta_seconds", 0.05)
    settings.no_rendering_mode = carla_cfg.get("no_rendering", False)
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    # Spawn ego vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter(ego_cfg.get("blueprint", "vehicle.lincoln.mkz_2017"))
    if not vehicle_bps:
        print(f"ERROR: Blueprint '{ego_cfg['blueprint']}' not found.")
        sys.exit(1)
    vehicle_bp = vehicle_bps[0]

    spawn_points = world.get_map().get_spawn_points()
    spawn_index  = ego_cfg.get("spawn_index", 0)
    spawn_point  = spawn_points[min(spawn_index, len(spawn_points) - 1)]

    vehicle = None
    for attempt, sp in enumerate([spawn_point] + random.sample(spawn_points, min(5, len(spawn_points)))):
        try:
            vehicle = world.spawn_actor(vehicle_bp, sp)
            print(f"[Main] Ego vehicle spawned at spawn point {attempt}: {sp.location}")
            break
        except Exception:
            continue

    if vehicle is None:
        print("ERROR: Could not spawn ego vehicle at any spawn point.")
        _restore_settings(world, settings)
        sys.exit(1)

    # Pick a random destination different from spawn
    destination = random.choice(
        [sp.location for sp in spawn_points if sp.location.distance(spawn_point.location) > 50]
    )
    print(f"[Main] Destination: {destination}")

    from agent import AutonomousAgent
    agent = AutonomousAgent(
        world=world,
        vehicle=vehicle,
        cfg=cfg,
        destination=destination,
        enable_hud=not args.no_hud,
    )

    print("[Main] Starting agent loop. Press Ctrl+C to stop.")
    step = 0
    try:
        while True:
            keep_running = agent.tick()
            if not keep_running:
                break
            step += 1
            if args.steps > 0 and step >= args.steps:
                print(f"[Main] Reached max steps ({args.steps}).")
                break
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    except Exception:
        print("[Main] Unhandled exception:")
        traceback.print_exc()
    finally:
        print("[Main] Cleaning up...")
        agent.destroy()
        vehicle.destroy()
        _restore_settings(world, settings)
        print("[Main] Done.")


def _restore_settings(world, original_settings) -> None:
    original_settings.synchronous_mode = False
    world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
