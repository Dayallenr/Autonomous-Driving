#!/usr/bin/env python
"""
Validate the CARLA backend against a live server — issue #5.

    # terminal 1: start CARLA
    CarlaUE4.exe
    # terminal 2:
    python scripts/validate_carla_backend.py

Drives one short routed Episode twice with a driving policy that steers toward
the lookahead point (pure throttle+brake would never turn, so command variety
and route completion could not be exercised), and checks every acceptance
criterion from issue #5:

  - route completion advances monotonically
  - the navigation command visits all four branches, not just FOLLOW_LANE
  - the camera observation is an 88x200x3 array
  - traffic and pedestrians spawn at the configured densities (at least half
    of each requested count — try_spawn_actor legitimately drops some to
    spawn-point collisions)
  - collisions and red-light violations register as infractions (best effort —
    a clean drive may legitimately hit neither; this is reported, not failed)
  - teardown restores the server to asynchronous mode
  - the same seed driven twice produces identical trajectories, completions,
    commands, and infractions

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

# One place for the server address: the episode connects through these, and the
# post-teardown async check must probe the same server, not a hardcoded twin.
HOST = "127.0.0.1"
PORT = 2000


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
    sim = build_simulator("carla", host=HOST, port=PORT, render_camera=True)
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
        sync_during_episode = bool(world.get_settings().synchronous_mode) if world else None
        # Requested counts mirror _spawn_traffic's own arithmetic, so "at the
        # configured densities" is checked against what the backend was asked
        # for, not against a bare non-zero.
        spawn_points = len(world.get_map().get_spawn_points()) if world else 0
        vehicles_requested = int(spawn_points * spec.traffic_density)
        walkers_requested = int(60 * spec.pedestrian_density)
        traffic_spawned = len(sim._traffic)
        walkers_spawned = len(sim._walkers)
    finally:
        sim.close()
        # sim._world is None after close(); check the server directly instead.
        import carla

        client = carla.Client(HOST, PORT)
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
        "vehicles_requested": vehicles_requested,
        "walkers_requested": walkers_requested,
        "traffic_spawned": traffic_spawned,
        "walkers_spawned": walkers_spawned,
        "sync_during_episode": sync_during_episode,
        "restored_async_after_close": restored_async,
        "steps_run": len(positions) - 1,
        "final_position": positions[-1],
        "completions": completions,
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

    hidden_keys = ("trajectory", "completions")
    print("=== episode 1 ===")
    episode1 = run_episode(spec)
    print(json.dumps({k: v for k, v in episode1.items() if k not in hidden_keys}, indent=2))

    print("\n=== episode 2 (same seed) ===")
    episode2 = run_episode(spec)
    print(json.dumps({k: v for k, v in episode2.items() if k not in hidden_keys}, indent=2))

    # Determinism means everything observable repeats, not just positions:
    # matching trajectories with diverging infractions would still make
    # results irreproducible.
    diverged = [
        key
        for key in ("trajectory", "completions", "commands_seen", "infractions_seen")
        if episode1[key] != episode2[key]
    ]
    report["episode1"] = episode1
    report["episode2"] = episode2
    report["deterministic_repeat"] = not diverged

    if not episode1["monotonic"] or not episode2["monotonic"]:
        hard_failures.append("route completion was not monotonic")
    if not episode1["all_four_branches"]:
        hard_failures.append(
            f"only saw commands {episode1['commands_seen']}, not all four branches"
        )
    if not episode1["image_shape_ok"]:
        hard_failures.append(
            f"camera image shape was {episode1['image_shape']}, expected (88, 200, 3)"
        )
    if episode1["traffic_spawned"] < max(1, episode1["vehicles_requested"] // 2):
        hard_failures.append(
            f"{episode1['traffic_spawned']} of {episode1['vehicles_requested']} requested "
            "traffic vehicles spawned (traffic_density=0.3)"
        )
    if episode1["walkers_spawned"] < max(1, episode1["walkers_requested"] // 2):
        hard_failures.append(
            f"{episode1['walkers_spawned']} of {episode1['walkers_requested']} requested "
            "pedestrians spawned (pedestrian_density=0.15)"
        )
    if not episode1["restored_async_after_close"] or not episode2["restored_async_after_close"]:
        hard_failures.append("world was not restored to asynchronous mode after close()")
    if diverged:
        hard_failures.append(
            f"two episodes of the same seed diverged in: {', '.join(diverged)}"
        )

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
