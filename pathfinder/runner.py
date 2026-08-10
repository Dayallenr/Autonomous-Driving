"""
Episode runner: drive one episode, emit telemetry, produce a score.

This is the seam between the driving stack and the distributed layer. A worker
does not know how to drive; it knows how to pull an :class:`EpisodeSpec` off a
queue and hand it here.

Telemetry is emitted **during** the episode, not collected and flushed at the
end. Two reasons, both operational: a crashed episode still yields the frames
leading up to the crash (which are the interesting ones), and memory stays
bounded on a 2000-step run instead of growing with episode length.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from pathfinder.metrics.driving_score import EpisodeScore, score_episode
from pathfinder.sim.base import EpisodeSpec, FrameState, Infraction, SimulatorBackend

logger = logging.getLogger(__name__)

__all__ = ["ControlOutput", "Planner", "PurePursuitPlanner", "run_episode"]


@dataclass(frozen=True)
class ControlOutput:
    """One control command."""

    throttle: float
    steer: float
    brake: float
    #: How long the planner took, carried into telemetry so planner latency is
    #: measured rather than estimated.
    latency_ms: float = 0.0


class Planner(Protocol):
    """What the runner needs from a planner. Deliberately structural: the CIL
    model, a classical controller, and a test stub all satisfy it without
    inheriting anything."""

    def plan(self, state: FrameState) -> ControlOutput: ...


class PurePursuitPlanner:
    """Geometric baseline: Stanley-style lateral control + speed governor.

    Exists so the whole pipeline is runnable and scoreable without trained
    weights, and so the CIL planner has something to be measured against. A
    learned planner that cannot beat a geometric controller has not demonstrated
    anything, and without a baseline in the repo that comparison never gets made.

    Lateral control combines three terms, which is what Stanley control is:

    * **feedforward** from path curvature — steers into the bend before error
      accumulates, rather than reacting after the car has already drifted;
    * **heading error** — aligns the car with the path tangent;
    * **cross-track error**, attenuated by speed via ``atan(k·e / v)`` — the
      characteristic Stanley term. Dividing by speed is what keeps it stable:
      a fixed cross-track gain that is comfortable at 3 m/s oscillates violently
      at 20 m/s, because the same steering angle produces far more lateral
      acceleration.
    """

    def __init__(
        self,
        *,
        target_speed_mps: float = 8.0,
        crosstrack_gain: float = 1.2,
        heading_gain: float = 1.0,
        curvature_gain: float = 8.0,
        brake_distance_m: float = 14.0,
    ) -> None:
        self.target_speed_mps = target_speed_mps
        self.crosstrack_gain = crosstrack_gain
        self.heading_gain = heading_gain
        self.curvature_gain = curvature_gain
        self.brake_distance_m = brake_distance_m

    def plan(self, state: FrameState) -> ControlOutput:
        started = time.perf_counter()

        # Speed floor in the denominator: at a standstill the cross-track term
        # would divide by zero and saturate steering to full lock.
        speed = max(state.speed_mps, 1.0)

        crosstrack_term = math.atan2(self.crosstrack_gain * -state.lateral_error_m, speed)
        heading_term = self.heading_gain * -state.heading_error_rad
        feedforward = self.curvature_gain * state.lookahead_curvature
        steer = max(-1.0, min(1.0, feedforward + heading_term + crosstrack_term))

        # Speed governor: slow for obstacles, for curvature, and for large error.
        target = self.target_speed_mps
        if state.nearest_object_m < self.brake_distance_m:
            proximity = max(0.0, (state.nearest_object_m - 3.0) / self.brake_distance_m)
            target *= proximity
        target *= max(0.35, 1.0 - abs(state.lookahead_curvature) * 12.0)
        # Recovering from a large cross-track error is easier slowly; this is
        # also what stops a small drift from compounding into an off-road.
        target *= max(0.4, 1.0 - abs(state.lateral_error_m) / 4.0)

        # Stop for a red light. The comfortable-deceleration stopping distance is
        # v^2/(2a); braking is initiated a little before that so the stop is not
        # at the limit of the actuator.
        if state.traffic_light_state == "red":
            stopping_distance = state.speed_mps**2 / (2 * 3.0) + 4.0
            if state.traffic_light_distance_m <= stopping_distance:
                target = 0.0

        error = target - state.speed_mps
        throttle = min(1.0, max(0.0, error * 0.6))
        brake = min(1.0, max(0.0, -error * 0.4))

        return ControlOutput(
            throttle=throttle,
            steer=steer,
            brake=brake,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def run_episode(
    simulator: SimulatorBackend,
    spec: EpisodeSpec,
    planner: Planner,
    *,
    telemetry_sink: Callable[[dict], None] | None = None,
    worker_id: str = "local",
    model_version: str = "pure_pursuit",
    dataset_version: str = "",
) -> EpisodeScore:
    """Drive one episode to termination and score it.

    Args:
        simulator: Backend to drive in.
        spec: The episode to run.
        planner: Anything satisfying :class:`Planner`.
        telemetry_sink: Called once per frame with a warehouse-shaped row.
            Exceptions from it are swallowed — telemetry must never abort a run.
        worker_id: Recorded in telemetry for attribution.
        model_version: Planner identifier, recorded so results can be compared
            across model versions.
        dataset_version: Data the planner trained on, recorded for provenance.

    Returns:
        The scored episode. A simulator failure produces a ``status="failed"``
        score rather than an exception, so one bad episode does not abort a
        whole benchmark run.
    """
    infractions: list[Infraction] = []
    started = time.perf_counter()
    event_date = datetime.now(UTC).strftime("%Y-%m-%d")
    state = simulator.reset(spec)
    frames = 0
    status = "completed"
    reason = ""

    try:
        while frames < spec.max_steps:
            control = planner.plan(state)

            perception_started = time.perf_counter()
            result = simulator.step(control.throttle, control.steer, control.brake)
            step_ms = (time.perf_counter() - perception_started) * 1000.0

            frames += 1
            infractions.extend(result.infractions)

            if telemetry_sink is not None:
                try:
                    telemetry_sink(
                        {
                            "episode_id": spec.episode_id,
                            "frame_index": result.state.frame_index,
                            "timestamp": datetime.now(UTC),
                            "simulation_time": result.state.simulation_time,
                            "worker_id": worker_id,
                            "town": spec.town,
                            "weather": spec.weather,
                            "x": result.state.x,
                            "y": result.state.y,
                            "yaw_degrees": result.state.yaw_degrees,
                            "speed_mps": result.state.speed_mps,
                            "throttle": control.throttle,
                            "brake": control.brake,
                            "steer": control.steer,
                            "command": int(result.state.command),
                            "detections": result.state.detections,
                            "nearest_object_m": (
                                result.state.nearest_object_m
                                if result.state.nearest_object_m != float("inf")
                                else -1.0
                            ),
                            "planner_latency_ms": control.latency_ms,
                            "perception_latency_ms": step_ms,
                            "infraction": result.infractions[0].value
                            if result.infractions
                            else "",
                            "event_date": event_date,
                        }
                    )
                except Exception as error:
                    # Telemetry is derived data. Losing a row is acceptable;
                    # losing the episode because of a logging failure is not.
                    logger.warning("telemetry sink failed on frame %d: %s", frames, error)

            state = result.state
            if result.done:
                reason = result.reason
                break
        else:
            reason = "max steps reached"

    except Exception as error:
        logger.error("episode %s failed after %d frames: %s", spec.episode_id, frames, error)
        status = "failed"
        reason = str(error)

    return score_episode(
        episode_id=spec.episode_id,
        distance_travelled_m=state.distance_travelled_m,
        route_length_m=spec.route_length_m,
        infractions=infractions,
        frames=frames,
        duration_seconds=time.perf_counter() - started,
        status=status,
        termination_reason=reason,
    )
