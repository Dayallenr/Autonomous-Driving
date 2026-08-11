#!/usr/bin/env python
"""
Probe a live CARLA server and report what its API actually offers.

    # terminal 1: start CARLA
    CarlaUE4.exe
    # terminal 2:
    python scripts/probe_carla.py

Writes ``results/carla/probe.json`` and prints a summary. Read-only apart from
spawning and destroying a handful of actors, which it cleans up.

This exists because the CARLA backend in this repo was written without a server
to test against. Rather than guess at API shapes across CARLA versions, measure
them once and build against the answer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder.sim.carla_paths import CARLA_HINT, ensure_agents_importable  # noqa: E402

REQUIRED_BLUEPRINTS = (
    "vehicle.tesla.model3",
    "sensor.camera.rgb",
    "sensor.other.collision",
    "sensor.other.lane_invasion",
    "sensor.other.obstacle",
    "walker.pedestrian.0001",
    "controller.ai.walker",
)


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n {title}\n{'=' * 68}")


def probe_versions(carla, client) -> dict:
    section("versions")
    info = {
        "client_version": client.get_client_version(),
        "server_version": client.get_server_version(),
        "python_version": sys.version.split()[0],
        "carla_module": getattr(carla, "__file__", "?"),
    }
    info["version_match"] = info["client_version"] == info["server_version"]
    for key, value in info.items():
        print(f"  {key:20} {value}")
    if not info["version_match"]:
        print("  !! client and server versions differ — this causes subtle API failures")
    return info


def probe_maps(client) -> dict:
    section("maps")
    maps = [name.split("/")[-1] for name in client.get_available_maps()]
    print(f"  {len(maps)} available: {', '.join(sorted(set(maps)))}")
    return {"available_maps": sorted(set(maps))}


def probe_blueprints(world) -> dict:
    section("blueprints")
    library = world.get_blueprint_library()
    found = {}
    for name in REQUIRED_BLUEPRINTS:
        try:
            library.find(name)
            found[name] = True
        except Exception:
            found[name] = False
        print(f"  {'OK ' if found[name] else 'MISSING'} {name}")

    vehicles = len(library.filter("vehicle.*"))
    walkers = len(library.filter("walker.pedestrian.*"))
    print(f"\n  vehicle blueprints: {vehicles}   walker blueprints: {walkers}")
    return {"blueprints": found, "vehicle_count": vehicles, "walker_count": walkers}


def probe_settings(world) -> dict:
    section("world settings fields")
    settings = world.get_settings()
    fields = [a for a in dir(settings) if not a.startswith("_")]
    interesting = [
        "synchronous_mode", "fixed_delta_seconds", "substepping",
        "max_substep_delta_time", "max_substeps", "deterministic_ragdolls",
        "no_rendering_mode", "spectator_as_ego",
    ]
    present = {}
    for name in interesting:
        present[name] = name in fields
        value = getattr(settings, name, None) if present[name] else None
        print(f"  {'OK ' if present[name] else '-- '} {name:26} {value}")
    return {"settings_fields": present, "all_settings_fields": fields}


def probe_traffic_manager(client) -> dict:
    section("traffic manager")
    result = {}
    try:
        manager = client.get_trafficmanager()
        methods = [a for a in dir(manager) if not a.startswith("_")]
        for name in (
            "set_random_device_seed", "set_synchronous_mode",
            "set_hybrid_physics_mode", "global_percentage_speed_difference",
        ):
            result[name] = name in methods
            print(f"  {'OK ' if result[name] else '-- '} {name}")
        result["port"] = manager.get_port()
        print(f"  port: {result['port']}")
    except Exception as error:
        result["error"] = str(error)
        print(f"  !! traffic manager unavailable: {error}")
    return {"traffic_manager": result}


def probe_route_planner() -> dict:
    """The ``agents`` package ships in the CARLA release, not the pip wheel."""
    section("route planner (agents package)")
    result = {}

    located = ensure_agents_importable()
    if located is not None:
        result["discovered_at"] = str(located)
        print(f"  discovered CARLA PythonAPI at: {located}")

    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: F401

        result["importable"] = True
        print("  OK  agents.navigation.global_route_planner")
        try:
            from agents.navigation.local_planner import RoadOption  # noqa: F401

            result["road_option"] = True
            print("  OK  agents.navigation.local_planner.RoadOption")
        except Exception as error:
            result["road_option"] = False
            print(f"  !! RoadOption import failed: {error}")
    except Exception as error:
        result["importable"] = False
        result["error"] = str(error)
        print(f"  !! agents package NOT importable: {error}\n")
        for line in CARLA_HINT.splitlines():
            print(f"     {line}")
    return {"route_planner": result}


def probe_spawn_points(world) -> dict:
    section("spawn points and topology")
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    topology = carla_map.get_topology()
    print(f"  map: {carla_map.name}")
    print(f"  spawn points: {len(spawn_points)}")
    print(f"  topology edges: {len(topology)}")
    landmarks = []
    try:
        landmarks = world.get_actors().filter("traffic.traffic_light")
        print(f"  traffic lights in world: {len(landmarks)}")
    except Exception as error:
        print(f"  !! traffic light query failed: {error}")
    return {
        "map": str(carla_map.name),
        "spawn_points": len(spawn_points),
        "topology_edges": len(topology),
        "traffic_lights": len(landmarks),
    }


def probe_drive(carla, client, world, steps: int, delta: float) -> dict:
    """Drive straight in synchronous mode twice from one seed; compare paths.

    Determinism is the property the whole distributed benchmark rests on. If two
    identical runs diverge, "replay any result bit-identically" is not a claim
    this project can make about CARLA, and it is far better to learn that here
    than after building a benchmark on top of it.
    """
    section(f"synchronous drive x2 ({steps} steps @ {delta}s)")
    original = world.get_settings()
    trajectories = []
    timings = []

    try:
        for run in range(2):
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = delta
            world.apply_settings(settings)

            library = world.get_blueprint_library()
            vehicle_bp = library.filter("vehicle.tesla.model3")[0]
            spawn = world.get_map().get_spawn_points()[0]

            vehicle = None
            for _ in range(10):
                vehicle = world.try_spawn_actor(vehicle_bp, spawn)
                if vehicle is not None:
                    break
                world.tick()
            if vehicle is None:
                raise RuntimeError("could not spawn a vehicle after 10 attempts")

            collisions = []
            collision_bp = library.find("sensor.other.collision")
            collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
            # Bound as a default argument, not captured: `collisions` is rebound
            # each run, and a late event from run 0's sensor would otherwise
            # land in run 1's list and corrupt the determinism comparison.
            collision.listen(lambda event, sink=collisions: sink.append(event))

            world.tick()
            path = []
            started = time.perf_counter()
            for _ in range(steps):
                vehicle.apply_control(
                    carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
                )
                world.tick()
                location = vehicle.get_location()
                path.append((round(location.x, 6), round(location.y, 6)))
            elapsed = time.perf_counter() - started

            trajectories.append(path)
            timings.append(elapsed)
            print(f"  run {run}: {steps} ticks in {elapsed:.2f}s "
                  f"({steps / elapsed:.1f} tick/s), collisions={len(collisions)}")

            collision.stop()
            collision.destroy()
            vehicle.destroy()
            world.tick()
    finally:
        world.apply_settings(original)

    identical = trajectories[0] == trajectories[1]
    max_drift = 0.0
    if not identical:
        for (x1, y1), (x2, y2) in zip(trajectories[0], trajectories[1], strict=False):
            max_drift = max(max_drift, math.hypot(x1 - x2, y1 - y2))

    print(f"\n  final position run 0: {trajectories[0][-1]}")
    print(f"  final position run 1: {trajectories[1][-1]}")
    print(f"  bit-identical trajectories: {identical}")
    if not identical:
        print(f"  max divergence: {max_drift:.6f} m")
        print("  -> CARLA physics is not bit-reproducible across runs on this build.")
        print("     Replay guarantees will need to be stated as seed+config capture,")
        print("     not bit-identical trajectories.")

    return {
        "drive": {
            "steps": steps,
            "delta_seconds": delta,
            "ticks_per_second": [round(steps / t, 2) for t in timings],
            "deterministic": identical,
            "max_divergence_m": round(max_drift, 6),
            "final_positions": [trajectories[0][-1], trajectories[1][-1]],
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe a live CARLA server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("results/carla/probe.json"))
    arguments = parser.parse_args()

    try:
        import carla
    except ImportError:
        print("the 'carla' package is not importable.\n"
              "Install the client matching your server: pip install carla==<version>",
              file=sys.stderr)
        return 1

    try:
        client = carla.Client(arguments.host, arguments.port)
        client.set_timeout(arguments.timeout)
        world = client.get_world()
    except Exception as error:
        print(f"could not reach a CARLA server at {arguments.host}:{arguments.port}\n"
              f"  {error}\nIs CarlaUE4.exe running?", file=sys.stderr)
        return 1

    report: dict = {}
    for name, probe in (
        ("versions", lambda: probe_versions(carla, client)),
        ("maps", lambda: probe_maps(client)),
        ("blueprints", lambda: probe_blueprints(world)),
        ("settings", lambda: probe_settings(world)),
        ("traffic_manager", lambda: probe_traffic_manager(client)),
        ("route_planner", probe_route_planner),
        ("spawn_points", lambda: probe_spawn_points(world)),
        ("drive", lambda: probe_drive(carla, client, world, arguments.steps, arguments.delta)),
    ):
        try:
            report.update(probe())
        except Exception as error:
            print(f"\n!! probe '{name}' failed: {error}")
            traceback.print_exc()
            report[f"{name}_error"] = str(error)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {arguments.out}")
    print("Commit and push it, or paste the summary above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
