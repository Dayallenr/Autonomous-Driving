"""
Deterministic kinematic simulator — the backend that runs anywhere.

Vehicle model
-------------
A kinematic bicycle model. It is the standard low-speed approximation and it is
the right level of fidelity here: it reproduces the constraint that actually
shapes a policy's behaviour — that a car cannot translate sideways, and that
turn radius is bounded by steering angle and wheelbase — without pretending to
model tyre slip, weight transfer, or suspension.

    x'   = v·cos(ψ)
    y'   = v·sin(ψ)
    ψ'   = (v/L)·tan(δ)
    v'   = a

with wheelbase L, steering angle δ, and acceleration a from throttle/brake.

Above roughly 15 m/s on tight corners the no-slip assumption stops holding, so
the model is optimistic there. That is stated rather than hidden: this backend
measures the pipeline, not driving ability.

Determinism
-----------
Every stochastic element — obstacle placement, traffic light phases, pedestrian
crossings — is drawn from a ``random.Random`` seeded by ``EpisodeSpec.seed``. The
same spec produces a byte-identical episode on any machine. Without that, a
distributed benchmark cannot be reproduced and a regression cannot be bisected.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from pathfinder.sim.base import (
    Command,
    EpisodeSpec,
    FrameState,
    Infraction,
    SimulatorBackend,
    StepResult,
)
from pathfinder.sim.render import render_forward_view

__all__ = ["KinematicSimulator"]

# Vehicle constants, roughly a compact sedan.
WHEELBASE_M = 2.7
MAX_STEER_RADIANS = math.radians(35.0)
MAX_ACCELERATION_MPS2 = 3.0
MAX_BRAKE_MPS2 = 8.0
DRAG_COEFFICIENT = 0.02

# Lane geometry.
LANE_HALF_WIDTH_M = 1.75
OFF_ROAD_DISTANCE_M = 3.5

# An agent that has not moved for this long is stuck. CARLA's leaderboard uses
# the same idea: without it, a policy that simply stops would never terminate
# and would score a perfect zero-infraction run.
BLOCKED_SPEED_MPS = 0.1
BLOCKED_SECONDS = 20.0

COLLISION_DISTANCE_M = 2.0

# How far ahead traffic lights are perceivable.
TRAFFIC_LIGHT_SENSING_M = 60.0

# Range within which obstacles are reported in `detections`. The CARLA backend
# imports this so the two backends' privileged sensing windows cannot drift.
OBSTACLE_SENSING_RANGE_M = 50.0


@dataclass
class _Obstacle:
    """An object along the route.

    Vehicles and pedestrians **move**. That is not decoration: with static
    traffic, any obstacle in the ego's lane blocks it permanently, the ego
    brakes to a stop, and every episode ends in ``agent_blocked``. The benchmark
    then measures nothing except that the ego can brake. Moving traffic makes
    car-following the actual task, which is what the policy should be scored on.
    """

    distance_along_route_m: float
    lateral_offset_m: float
    kind: str  # "vehicle" | "pedestrian" | "static"
    #: Longitudinal speed along the route (vehicles) or lateral crossing speed
    #: (pedestrians). Static objects are 0.
    speed_mps: float = 0.0

    def advance(self, delta_seconds: float) -> None:
        if self.kind == "vehicle":
            self.distance_along_route_m += self.speed_mps * delta_seconds
        elif self.kind == "pedestrian":
            # Pedestrians cross the road and keep going, clearing the lane.
            self.lateral_offset_m += self.speed_mps * delta_seconds


@dataclass
class _TrafficLight:
    distance_along_route_m: float
    #: Cycle position in seconds; red for the first `red_seconds` of each cycle.
    cycle_seconds: float
    red_seconds: float
    phase_offset: float

    def is_red(self, simulation_time: float) -> bool:
        position = (simulation_time + self.phase_offset) % self.cycle_seconds
        return position < self.red_seconds


class KinematicSimulator(SimulatorBackend):
    """Bicycle-model simulator over a procedurally generated route.

    Args:
        render: When true, each :class:`FrameState` carries a synthetic forward
            view in ``.image``. Off by default because rendering costs ~10x a
            bare step, and orchestration benchmarks do not need pixels.
    """

    def __init__(self, *, render: bool = False) -> None:
        self.render = render
        self._spec: EpisodeSpec | None = None
        self._rng: random.Random | None = None
        self._obstacles: list[_Obstacle] = []
        self._lights: list[_TrafficLight] = []
        self._route_curvature: list[float] = []
        self._reference: list[tuple[float, float, float]] = []

        self._frame = 0
        self._time = 0.0
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._speed = 0.0
        self._distance = 0.0
        self._stopped_seconds = 0.0
        self._lights_passed: set[int] = set()
        self._collided = False

    @property
    def name(self) -> str:
        return "kinematic"

    def reset(self, spec: EpisodeSpec) -> FrameState:
        self._spec = spec
        self._rng = random.Random(spec.seed)

        self._frame = 0
        self._time = 0.0
        self._x = self._y = self._yaw = 0.0
        self._speed = 0.0
        self._distance = 0.0
        self._stopped_seconds = 0.0
        self._lights_passed = set()
        self._collided = False

        self._generate_route(spec)
        self._build_reference_path(spec)
        return self._state()

    def _generate_route(self, spec: EpisodeSpec) -> None:
        """Lay out curvature, obstacles, and lights for the whole route up front.

        Generating everything at reset (rather than lazily as the ego advances)
        is what makes the episode a pure function of the seed: lazy generation
        would couple the layout to the trajectory, so a policy change would
        silently change the scenario it is being evaluated on.
        """
        assert self._rng is not None
        rng = self._rng

        # Curvature profile in 10 m segments: mostly straight with occasional turns.
        segments = max(1, int(spec.route_length_m // 10))
        self._route_curvature = []
        for _ in range(segments):
            roll = rng.random()
            if roll < 0.70:
                self._route_curvature.append(0.0)                       # straight
            elif roll < 0.85:
                self._route_curvature.append(rng.uniform(0.01, 0.04))   # left
            else:
                self._route_curvature.append(-rng.uniform(0.01, 0.04))  # right

        self._obstacles = []
        vehicle_count = int(spec.route_length_m / 50 * spec.traffic_density * 3)
        for _ in range(vehicle_count):
            self._obstacles.append(
                _Obstacle(
                    distance_along_route_m=rng.uniform(20, max(21.0, spec.route_length_m)),
                    lateral_offset_m=rng.choice([-3.5, 0.0, 3.5]),
                    kind="vehicle",
                    # Traffic moves slower than the ego's target, so following
                    # is the task rather than an unavoidable wall.
                    speed_mps=rng.uniform(4.0, 7.5),
                )
            )
        pedestrian_count = int(spec.route_length_m / 50 * spec.pedestrian_density * 2)
        for _ in range(pedestrian_count):
            self._obstacles.append(
                _Obstacle(
                    distance_along_route_m=rng.uniform(20, max(21.0, spec.route_length_m)),
                    lateral_offset_m=rng.uniform(-2.0, 2.0),
                    kind="pedestrian",
                    # Crossing, so the lane clears rather than staying blocked.
                    speed_mps=rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 1.6),
                )
            )
        self._obstacles.sort(key=lambda obstacle: obstacle.distance_along_route_m)

        self._lights = [
            _TrafficLight(
                distance_along_route_m=distance,
                cycle_seconds=rng.uniform(20, 40),
                red_seconds=rng.uniform(8, 15),
                phase_offset=rng.uniform(0, 20),
            )
            for distance in range(80, int(spec.route_length_m), 150)
        ]

    def _command_for(self, distance: float) -> Command:
        """Derive the navigation command from upcoming curvature."""
        index = min(int(distance // 10), len(self._route_curvature) - 1)
        if index < 0 or not self._route_curvature:
            return Command.FOLLOW_LANE
        curvature = self._route_curvature[index]
        if curvature > 0.02:
            return Command.TURN_LEFT
        if curvature < -0.02:
            return Command.TURN_RIGHT
        if abs(curvature) < 0.005:
            return Command.GO_STRAIGHT
        return Command.FOLLOW_LANE

    def _build_reference_path(self, spec: EpisodeSpec) -> None:
        """Precompute the route centreline once, at 1 m resolution.

        Built up front rather than re-integrated per frame: the earlier version
        walked the whole path on every call, making the episode O(n^2) in route
        length. At 1 m spacing a 5 km route is 5000 points, which is nothing to
        store and turns tracking into an index lookup.
        """
        self._reference: list[tuple[float, float, float]] = []  # (x, y, yaw)
        x = y = yaw = 0.0
        self._reference.append((x, y, yaw))
        step = 1.0
        travelled = 0.0
        # Extend past the route end so lookahead near the finish still resolves.
        while travelled < spec.route_length_m + 50.0:
            index = min(int(travelled // 10), len(self._route_curvature) - 1)
            curvature = self._route_curvature[index] if self._route_curvature else 0.0
            yaw += curvature * step
            x += math.cos(yaw) * step
            y += math.sin(yaw) * step
            travelled += step
            self._reference.append((x, y, yaw))

    def _reference_at(self, distance: float) -> tuple[float, float, float]:
        """Reference pose at a given arc length along the route."""
        if not self._reference:
            return (0.0, 0.0, 0.0)
        index = min(max(int(distance), 0), len(self._reference) - 1)
        return self._reference[index]

    def _tracking_errors(self) -> tuple[float, float, float]:
        """Return ``(lateral_error, heading_error, lookahead_curvature)``.

        Cross-track error is the ego offset projected onto the reference path's
        normal at the ego's arc length. This is a genuine tracking error, so a
        policy that steers badly accumulates it and eventually leaves the lane.
        """
        reference_x, reference_y, reference_yaw = self._reference_at(self._distance)
        dx, dy = self._x - reference_x, self._y - reference_y
        lateral = -dx * math.sin(reference_yaw) + dy * math.cos(reference_yaw)

        heading_error = (self._yaw - reference_yaw + math.pi) % (2 * math.pi) - math.pi

        # Curvature at a speed-dependent lookahead: faster means look further,
        # which is what keeps a pure-pursuit controller stable across speeds.
        lookahead = max(5.0, self._speed * 1.5)
        index = min(
            int((self._distance + lookahead) // 10),
            max(0, len(self._route_curvature) - 1),
        )
        curvature = self._route_curvature[index] if self._route_curvature else 0.0
        return lateral, heading_error, curvature

    def _lateral_error(self) -> float:
        return self._tracking_errors()[0]

    def _nearest_obstacle(self) -> tuple[float, _Obstacle | None]:
        nearest_distance = float("inf")
        nearest: _Obstacle | None = None
        for obstacle in self._obstacles:
            ahead = obstacle.distance_along_route_m - self._distance
            if ahead < -2.0:
                continue  # already behind
            if ahead > 60.0:
                break     # sorted, so nothing closer follows
            lateral = abs(obstacle.lateral_offset_m - self._lateral_error())
            distance = math.hypot(ahead, lateral)
            if distance < nearest_distance:
                nearest_distance, nearest = distance, obstacle
        return nearest_distance, nearest

    def _next_light(self) -> tuple[str, float]:
        """State and range of the next light ahead, within sensing distance."""
        for index, light in enumerate(self._lights):
            if index in self._lights_passed:
                continue
            ahead = light.distance_along_route_m - self._distance
            if ahead < 0:
                continue
            if ahead > TRAFFIC_LIGHT_SENSING_M:
                break  # lights are in ascending order
            return ("red" if light.is_red(self._time) else "green", ahead)
        return ("", float("inf"))

    def _state(self) -> FrameState:
        nearest_distance, _ = self._nearest_obstacle()
        visible = sum(
            1
            for obstacle in self._obstacles
            if 0 <= obstacle.distance_along_route_m - self._distance <= OBSTACLE_SENSING_RANGE_M
        )
        lateral, heading_error, curvature = self._tracking_errors()
        light_state, light_distance = self._next_light()
        return FrameState(
            frame_index=self._frame,
            simulation_time=self._time,
            x=self._x,
            y=self._y,
            yaw_degrees=math.degrees(self._yaw),
            speed_mps=self._speed,
            command=self._command_for(self._distance),
            distance_travelled_m=self._distance,
            nearest_object_m=nearest_distance,
            detections=visible,
            lateral_error_m=lateral,
            heading_error_rad=heading_error,
            lookahead_curvature=curvature,
            traffic_light_state=light_state,
            traffic_light_distance_m=light_distance,
            image=self._render_image(lateral, heading_error, curvature, light_state, light_distance)
            if self.render
            else None,
        )

    def _render_image(self, lateral, heading_error, curvature, light_state, light_distance):
        """Render the forward view for the CIL policy."""
        relative = [
            (
                obstacle.distance_along_route_m - self._distance,
                obstacle.lateral_offset_m - lateral,
                obstacle.kind,
            )
            for obstacle in self._obstacles
            if 0 < obstacle.distance_along_route_m - self._distance < 60
        ]
        return render_forward_view(
            lateral_error=lateral,
            heading_error=heading_error,
            curvature=curvature,
            obstacles=relative,
            traffic_light=(light_state, light_distance),
        )

    def step(self, throttle: float, steer: float, brake: float) -> StepResult:
        if self._spec is None:
            raise RuntimeError("step() called before reset()")

        # Clamp rather than reject: a policy producing out-of-range controls is
        # a bug worth surviving, and the simulator saturating is what real
        # actuators do anyway.
        throttle = min(max(throttle, 0.0), 1.0)
        brake = min(max(brake, 0.0), 1.0)
        steer = min(max(steer, -1.0), 1.0)

        delta = self._spec.delta_seconds

        # Advance traffic before the ego moves, then re-sort: _nearest_obstacle
        # relies on ascending order for its early exit.
        for obstacle in self._obstacles:
            obstacle.advance(delta)
        self._obstacles.sort(key=lambda obstacle: obstacle.distance_along_route_m)

        acceleration = throttle * MAX_ACCELERATION_MPS2 - brake * MAX_BRAKE_MPS2
        acceleration -= DRAG_COEFFICIENT * self._speed * abs(self._speed)

        self._speed = max(0.0, self._speed + acceleration * delta)
        steering_angle = steer * MAX_STEER_RADIANS
        self._yaw += (self._speed / WHEELBASE_M) * math.tan(steering_angle) * delta
        self._x += self._speed * math.cos(self._yaw) * delta
        self._y += self._speed * math.sin(self._yaw) * delta
        self._distance += self._speed * delta
        self._time += delta
        self._frame += 1

        infractions: list[Infraction] = []
        done = False
        reason = ""

        # ── collisions ───────────────────────────────────────────────────────
        nearest_distance, nearest = self._nearest_obstacle()
        if nearest is not None and nearest_distance < COLLISION_DISTANCE_M and not self._collided:
            self._collided = True
            infractions.append(
                {
                    "pedestrian": Infraction.COLLISION_PEDESTRIAN,
                    "vehicle": Infraction.COLLISION_VEHICLE,
                }.get(nearest.kind, Infraction.COLLISION_STATIC)
            )
            # A collided obstacle is removed so one object cannot register a
            # collision on every subsequent frame and dominate the score.
            self._obstacles.remove(nearest)
            self._collided = False

        # ── lane keeping ─────────────────────────────────────────────────────
        lateral_error = abs(self._lateral_error())
        if lateral_error > OFF_ROAD_DISTANCE_M:
            infractions.append(Infraction.OFF_ROAD)
            done, reason = True, "left the drivable area"
        elif lateral_error > LANE_HALF_WIDTH_M:
            infractions.append(Infraction.LANE_INVASION)

        # ── traffic lights ───────────────────────────────────────────────────
        for index, light in enumerate(self._lights):
            if index in self._lights_passed:
                continue
            if self._distance >= light.distance_along_route_m:
                self._lights_passed.add(index)
                # Running a red requires actually moving through it; creeping at
                # near-zero speed is not a violation.
                if light.is_red(self._time) and self._speed > 1.0:
                    infractions.append(Infraction.RED_LIGHT)

        # ── blocked agent ────────────────────────────────────────────────────
        if self._speed < BLOCKED_SPEED_MPS:
            self._stopped_seconds += delta
            if self._stopped_seconds >= BLOCKED_SECONDS:
                infractions.append(Infraction.AGENT_BLOCKED)
                done, reason = True, "agent blocked"
        else:
            self._stopped_seconds = 0.0

        # ── termination ──────────────────────────────────────────────────────
        if self._distance >= self._spec.route_length_m:
            done, reason = True, "route completed"
        elif self._frame >= self._spec.max_steps:
            done, reason = True, "max steps reached"

        return StepResult(state=self._state(), done=done, infractions=infractions, reason=reason)

    def close(self) -> None:
        self._spec = None
        self._obstacles.clear()
        self._lights.clear()
