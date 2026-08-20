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
  - the navigation command is real rather than hardcoded, checked in two parts:
    the planner and RoadOption mapping produce all four branches across a
    sample of routes, and the driven episode only ever surfaces commands its
    own route planned, reaching at least one junction manoeuvre. (Requiring all
    four *within one episode* is not a backend property — Town05 at seed 42
    plans a route containing no STRAIGHT manoeuvre at all.)
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
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder.runner import PurePursuitPolicy  # noqa: E402
from pathfinder.sim.base import Command, EpisodeSpec  # noqa: E402
from pathfinder.sim.carla_backend import build_simulator  # noqa: E402

# One place for the server address: the episode connects through these, and the
# post-teardown async check must probe the same server, not a hardcoded twin.
HOST = "127.0.0.1"
PORT = 2000

# The project's own geometric baseline drives this validation rather than a
# controller hand-rolled here. An earlier version of this script used a local
# proportional controller that omitted the negation on both error terms that
# PurePursuitPolicy applies (runner.py) — with the sign inverted the steering
# became positive feedback, so the ego turned away from the route within ~40
# steps and never reached a junction. Reusing the real controller keeps the
# convention in one place and means this script exercises the path the ablation
# actually drives.
_POLICY = PurePursuitPolicy(target_speed_mps=6.0)


def drive_toward_route(state) -> tuple[float, float, float]:
    control = _POLICY.plan(state)
    return control.throttle, control.steer, control.brake


def run_episode(spec: EpisodeSpec) -> dict:
    sim = build_simulator("carla", host=HOST, port=PORT, render_camera=True)
    completions: list[float] = []
    commands: set[int] = set()
    positions: list[tuple[float, float]] = []
    infractions_seen: list[str] = []
    image_shape = None

    try:
        state = sim.reset(spec)
        # What the planner laid out, to compare against what the ego was told.
        # A command the episode surfaces but the route never planned would mean
        # the plumbing invents commands rather than reporting them.
        planned_commands = sorted({int(p.command) for p in sim._route.points})
        route_length_m = sim._route.total_length_m
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
        "route_commands_planned": planned_commands,
        "route_length_m": route_length_m,
        # Plumbing, not topology: whether a single route happens to contain all
        # four branches is a property of the town, checked separately by
        # survey_navigation_commands(). What must hold here is that every
        # command surfaced was one the route actually planned, and that the ego
        # got far enough to be handed a junction manoeuvre at all.
        "commands_are_planned": commands <= set(planned_commands),
        "reached_a_junction_command": bool(commands - {int(Command.FOLLOW_LANE)}),
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


def survey_navigation_commands(town: str, *, samples: int = 40, seed: int = 0) -> dict:
    """Trace several routes and record which of the four branches the planner
    and :func:`road_option_to_command` actually produce between them.

    Issue #5 asked that the navigation command "vary across all four branches,
    not stay stuck on FOLLOW_LANE". Whether *one* route contains all four is a
    property of the town's topology, not of the backend — Town05 at seed 42
    plans a route with no STRAIGHT manoeuvre in it, and no amount of correct
    driving would make one appear. The property actually worth pinning is that
    the mapping from CARLA's real ``RoadOption`` values onto the policy's four
    branches is complete and not collapsing everything onto lane following.
    """
    import carla

    from pathfinder.sim.carla_paths import ensure_agents_importable
    from pathfinder.sim.route import road_option_to_command

    ensure_agents_importable(raise_on_missing=True)
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    if not world.get_map().name.endswith(town):
        world = client.load_world(town)

    carla_map = world.get_map()
    planner = GlobalRoutePlanner(carla_map, 2.0)
    spawn_points = carla_map.get_spawn_points()

    rng = random.Random(seed)
    road_options: Counter[str] = Counter()
    per_command: Counter[int] = Counter()
    routes_checked = 0

    for _ in range(samples):
        origin, destination = rng.sample(spawn_points, 2)
        # Short hops rarely contain a junction manoeuvre at all.
        if origin.location.distance(destination.location) < 150.0:
            continue
        try:
            plan = planner.trace_route(origin.location, destination.location)
        except Exception as error:  # topology gaps raise rather than return empty
            print(f"  (route planning failed for one pair: {error})")
            continue
        if len(plan) < 2:
            continue
        routes_checked += 1
        for _waypoint, option in plan:
            road_options[getattr(option, "name", str(option))] += 1
            per_command[int(road_option_to_command(option))] += 1

    return {
        "town": carla_map.name,
        "routes_checked": routes_checked,
        "road_options_seen": dict(road_options.most_common()),
        "waypoints_per_command": {Command(k).name: v for k, v in sorted(per_command.items())},
        "commands_produced": sorted(per_command),
        "all_four_branches": {int(c) for c in Command} <= set(per_command),
    }


def main() -> int:
    spec = EpisodeSpec(
        episode_id="validate-carla-backend",
        town="Town05",
        route_length_m=250.0,
        seed=42,
        # At 0.05 s per tick, 600 steps is 30 s — not enough to cover a 350 m
        # route through traffic and lights, and the first TURN_LEFT on this
        # route sits 225 m in. The backend's own default is 2000.
        max_steps=2000,
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

    print("\n=== navigation command survey ===")
    survey = survey_navigation_commands(spec.town)
    print(json.dumps(survey, indent=2))
    report["command_survey"] = survey

    if not episode1["monotonic"] or not episode2["monotonic"]:
        hard_failures.append("route completion was not monotonic")
    if not survey["all_four_branches"]:
        missing = [
            Command(c).name
            for c in sorted({int(c) for c in Command} - set(survey["commands_produced"]))
        ]
        hard_failures.append(
            f"the planner never produced {', '.join(missing)} across "
            f"{survey['routes_checked']} routes in {spec.town}"
        )
    if not episode1["commands_are_planned"]:
        hard_failures.append(
            f"episode surfaced commands {episode1['commands_seen']} but the route only "
            f"planned {episode1['route_commands_planned']}"
        )
    if not episode1["reached_a_junction_command"]:
        hard_failures.append(
            "the ego never received a junction command — it stayed on FOLLOW_LANE for the "
            f"whole episode (completion {episode1['final_completion']:.3f} of "
            f"{episode1['route_length_m']:.0f} m)"
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
