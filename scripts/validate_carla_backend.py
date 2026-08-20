#!/usr/bin/env python
"""
Validate the CARLA backend against a live server — issue #5.

    # terminal 1: start CARLA
    CarlaUE4.exe
    # terminal 2:
    python scripts/validate_carla_backend.py

Runs one short routed episode twice with a driving policy that steers toward
the lookahead point (pure throttle+brake would never turn, so command variety
and route completion could not be exercised), and checks every acceptance
criterion from issue #5:

  - route completion advances monotonically
  - the navigation command visits all four branches, not just FOLLOW_LANE
  - the camera observation is an 88x200x3 array
  - traffic and pedestrians spawn at the configured densities
  - collisions and red-light violations register as infractions (best effort —
    a clean drive may legitimately hit neither; this is reported, not failed)
  - teardown restores the server to asynchronous mode
  - the same seed run twice produces identical results

Writes results/carla/backend_validation.json. Exits non-zero if any hard
criterion fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder.sim.base import Command, EpisodeSpec  # noqa: E402
from pathfinder.sim.carla_backend import build_simulator  # noqa: E402


def drive_toward_route(state, kp_steer: float = 1.2) -> tuple[float, float, float]:
    """A minimal proportional controller — enough to actually turn at
    intersections rather than drive straight into a curb, without pulling in
    the CIL model or ModularPolicy machinery this validation isn't about."""
    steer = max(-1.0, min(1.0, kp_steer * state.heading_error_rad + 0.5 * state.lateral_error_m / 10.0))
    if state.traffic_light_state == "red" and state.traffic_light_distance_m < 15.0:
        return 0.0, steer, 0.8
    target_speed = 6.0
    if state.speed_mps < target_speed:
        return 0.6, steer, 0.0
    return 0.2, steer, 0.0


def run_episode(spec: EpisodeSpec) -> dict:
    sim = build_simulator("carla", render_camera=True)
    completions: list[float] = []
    commands: set[int] = set()
    positions: list[tuple[float, float]] = []
    infractions_seen: list[str] = []
    image_shape = None

    try:
        state = sim.reset(spec)
        image_shape = None if state.image is None else tuple(state.image.shape)
        commands.add(int(state.command))
        completions.append(sim.route_completion)
        positions.append((round(state.x, 6), round(state.y, 6)))

        for _ in range(spec.max_steps):
            throttle, steer, brake = drive_toward_route(state)
            result = sim.step(throttle, steer, brake)
            state = result.state
            commands.add(int(state.command))
            completions.append(sim.route_completion)
            positions.append((round(state.x, 6), round(state.y, 6)))
            infractions_seen.extend(i.value for i in result.infractions)
            if result.done:
                break

        world = sim._world
        sync_during_run = bool(world.get_settings().synchronous_mode) if world else None
        traffic_spawned = len(sim._traffic)
        walkers_spawned = len(sim._walkers)
    finally:
        sim.close()
        # sim._world is None after close(); check the server directly instead.
        import carla

        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10.0)
        restored_async = not client.get_world().get_settings().synchronous_mode

    return {
        "final_completion": completions[-1],
        "monotonic": all(
            b >= a - 1e-9 for a, b in zip(completions, completions[1:], strict=False)
        ),
        "commands_seen": sorted(commands),
        "all_four_branches": {int(c) for c in Command} <= commands,
        "image_shape": image_shape,
        "image_shape_ok": image_shape == (88, 200, 3),
        "infractions_seen": sorted(set(infractions_seen)),
        "traffic_spawned": traffic_spawned,
        "walkers_spawned": walkers_spawned,
        "sync_during_run": sync_during_run,
        "restored_async_after_close": restored_async,
        "steps_run": len(positions) - 1,
        "final_position": positions[-1],
        "trajectory": positions,
    }


def main() -> int:
    spec = EpisodeSpec(
        episode_id="validate-carla-backend",
        town="Town05",
        route_length_m=250.0,
        seed=42,
        max_steps=600,
        traffic_density=0.3,
        pedestrian_density=0.15,
    )

    report: dict = {"spec": spec.to_dict()}
    hard_failures: list[str] = []

    print("=== run 1 ===")
    run1 = run_episode(spec)
    print(json.dumps({k: v for k, v in run1.items() if k != "trajectory"}, indent=2))

    print("\n=== run 2 (same seed) ===")
    run2 = run_episode(spec)
    print(json.dumps({k: v for k, v in run2.items() if k != "trajectory"}, indent=2))

    deterministic = run1["trajectory"] == run2["trajectory"]
    report["run1"] = run1
    report["run2"] = run2
    report["deterministic_repeat"] = deterministic

    if not run1["monotonic"] or not run2["monotonic"]:
        hard_failures.append("route completion was not monotonic")
    if not run1["all_four_branches"]:
        hard_failures.append(f"only saw commands {run1['commands_seen']}, not all four branches")
    if not run1["image_shape_ok"]:
        hard_failures.append(f"camera image shape was {run1['image_shape']}, expected (88, 200, 3)")
    if run1["traffic_spawned"] == 0:
        hard_failures.append("no traffic vehicles spawned (traffic_density=0.3)")
    if run1["walkers_spawned"] == 0:
        hard_failures.append("no pedestrians spawned (pedestrian_density=0.15)")
    if not run1["restored_async_after_close"] or not run2["restored_async_after_close"]:
        hard_failures.append("world was not restored to asynchronous mode after close()")
    if not deterministic:
        hard_failures.append("two runs of the same seed produced different trajectories")

    report["hard_failures"] = hard_failures
    report["infractions_note"] = (
        "collisions/red-light infractions are opportunistic on this route/seed; "
        "an empty list here does not fail validation, but if you want a positive "
        "check, look at infractions_seen and re-run with a seed/town that crosses "
        "a light or forces a close pass."
    )

    out = Path("results/carla/backend_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")

    if hard_failures:
        print("\nFAILED:")
        for failure in hard_failures:
            print(f"  - {failure}")
        return 1

    print("\nAll hard criteria passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
