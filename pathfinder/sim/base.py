"""
Simulator abstraction: what an episode is, independent of who simulates it.

CARLA does not run on Apple Silicon and needs a GPU besides, so the
orchestration, telemetry, and scoring layers would be untestable on a laptop if
they talked to CARLA directly. They talk to this interface instead.

The interface is deliberately narrow — ``reset``, ``step``, ``close`` — because
that is the whole surface the driving loop actually needs, and a narrow surface
is what makes a second backend honest rather than a pile of stubs.

What the kinematic backend does and does not prove
--------------------------------------------------
It exercises: route following, control saturation, collision and lane-invasion
bookkeeping, telemetry shape and volume, episode termination, driving-score
arithmetic, and the entire distributed orchestration path.

It does **not** prove: perception quality (no rendered images), CARLA-specific
physics, or traffic-manager behaviour. Numbers from it are numbers about the
*pipeline*, not about driving ability, and the demo labels them that way.
Conflating the two would be the whole point of a benchmark thrown away.
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = [
    "Command",
    "EpisodeSpec",
    "FrameState",
    "Infraction",
    "SimulatorBackend",
    "StepResult",
]


class Command(enum.IntEnum):
    """High-level navigation command. Values index the CIL model's branches, so
    they must match ``pathfinder/planning/cil_model.py``."""

    FOLLOW_LANE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    GO_STRAIGHT = 3


class Infraction(str, enum.Enum):
    """Scoreable violations, named to match the CARLA leaderboard."""

    COLLISION_PEDESTRIAN = "collision_pedestrian"
    COLLISION_VEHICLE = "collision_vehicle"
    COLLISION_STATIC = "collision_static"
    RED_LIGHT = "red_light"
    STOP_SIGN = "stop_sign"
    LANE_INVASION = "lane_invasion"
    OFF_ROAD = "off_road"
    ROUTE_DEVIATION = "route_deviation"
    AGENT_BLOCKED = "agent_blocked"


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything needed to run one reproducible episode.

    ``seed`` is part of the spec rather than global state so that two workers
    running the same spec produce the same episode. Without that, a distributed
    benchmark is not reproducible and a regression cannot be bisected.
    """

    episode_id: str
    town: str = "Town05"
    weather: str = "ClearNoon"
    route_length_m: float = 500.0
    seed: int = 0
    max_steps: int = 2000
    #: Simulation seconds per step. 0.05 = 20 Hz, CARLA's usual fixed delta.
    delta_seconds: float = 0.05
    traffic_density: float = 0.3
    pedestrian_density: float = 0.15

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "town": self.town,
            "weather": self.weather,
            "route_length_m": self.route_length_m,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "delta_seconds": self.delta_seconds,
            "traffic_density": self.traffic_density,
            "pedestrian_density": self.pedestrian_density,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> EpisodeSpec:
        return cls(**{key: payload[key] for key in payload if key in cls.__annotations__})


@dataclass
class FrameState:
    """Ego state and observations for one simulation step."""

    frame_index: int
    simulation_time: float
    x: float
    y: float
    yaw_degrees: float
    speed_mps: float
    command: Command
    distance_travelled_m: float
    #: Range to the nearest obstacle in metres; inf when nothing is near.
    nearest_object_m: float = float("inf")
    detections: int = 0
    #: Signed cross-track error to the route centreline, metres. Positive is
    #: left of the path. A real stack gets this from localization plus the route;
    #: exposing it here is what lets a controller actually track a route rather
    #: than dead-reckon from the discrete command alone.
    lateral_error_m: float = 0.0
    #: Ego heading minus reference-path heading, radians, wrapped to [-pi, pi].
    heading_error_rad: float = 0.0
    #: Curvature of the path at the lookahead point, 1/m. Signed: positive left.
    lookahead_curvature: float = 0.0
    #: State of the next traffic light ahead: "red", "green", or "" when none is
    #: within sensing range. A real stack perceives these; without exposing them
    #: the policy cannot avoid a red-light infraction and the metric measures
    #: the simulator's omission rather than the policy.
    traffic_light_state: str = ""
    #: Distance to that light in metres; inf when none is in range.
    traffic_light_distance_m: float = float("inf")
    #: Present only when the backend renders images. The kinematic backend does
    #: not, which is why perception quality is explicitly out of its scope.
    image: object | None = None


@dataclass
class StepResult:
    """Outcome of one control step."""

    state: FrameState
    done: bool = False
    infractions: list[Infraction] = field(default_factory=list)
    reason: str = ""


class SimulatorBackend(ABC):
    """Minimal simulator surface used by the driving loop."""

    @abstractmethod
    def reset(self, spec: EpisodeSpec) -> FrameState:
        """Start an episode and return the initial state."""

    @abstractmethod
    def step(self, throttle: float, steer: float, brake: float) -> StepResult:
        """Apply a control command for one tick."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier, recorded in telemetry so results are never
        misattributed to a simulator that did not produce them."""

    def __enter__(self) -> SimulatorBackend:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
