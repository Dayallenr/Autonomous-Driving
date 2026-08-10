"""
CARLA simulator backend.

Used when the ``carla`` package is importable and a server is reachable. CARLA
does not ship for Apple Silicon and requires a GPU, so :func:`build_simulator`
falls back to the kinematic backend and says so loudly rather than failing at
import time — the rest of the stack is fully exercisable without it.

Synchronous mode
----------------
The world is put into synchronous mode with a fixed delta. In asynchronous mode
the server ticks on its own clock, so a slow perception step means the control
loop acts on stale state and the episode is not reproducible run to run.
Benchmarking in asynchronous mode measures the host's spare capacity as much as
the planner. The original settings are restored on ``close`` because they are
server-global: leaving a shared CARLA server in synchronous mode makes every
other client hang waiting for ticks that never come.
"""
from __future__ import annotations

import logging
import math

from pathfinder.sim.base import (
    Command,
    EpisodeSpec,
    FrameState,
    Infraction,
    SimulatorBackend,
    StepResult,
)
from pathfinder.sim.kinematic import KinematicSimulator

logger = logging.getLogger(__name__)

__all__ = ["CarlaSimulator", "build_simulator", "carla_available"]

_WEATHER_PRESETS = (
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
    "MidRainyNoon", "HardRainNoon", "SoftRainNoon",
    "ClearSunset", "CloudySunset", "WetSunset",
)


def carla_available() -> bool:
    """True when the CARLA Python API can be imported."""
    try:
        import carla  # noqa: F401

        return True
    except ImportError:
        return False


class CarlaSimulator(SimulatorBackend):
    """Real CARLA backend."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 2000,
        timeout_seconds: float = 20.0,
    ) -> None:
        import carla

        self._carla = carla
        self._client = carla.Client(host, port)
        self._client.set_timeout(timeout_seconds)
        self._world = None
        self._vehicle = None
        self._sensors: list = []
        self._original_settings = None

        self._spec: EpisodeSpec | None = None
        self._frame = 0
        self._time = 0.0
        self._distance = 0.0
        self._last_location = None
        self._pending: list[Infraction] = []
        self._stopped_seconds = 0.0

    @property
    def name(self) -> str:
        return "carla"

    def reset(self, spec: EpisodeSpec) -> FrameState:
        carla = self._carla
        self._spec = spec
        self._cleanup_actors()

        self._world = self._client.load_world(spec.town)
        self._original_settings = self._world.get_settings()

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = spec.delta_seconds
        self._world.apply_settings(settings)

        if spec.weather in _WEATHER_PRESETS:
            self._world.set_weather(getattr(carla.WeatherParameters, spec.weather))

        blueprints = self._world.get_blueprint_library()
        vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"town {spec.town} has no spawn points")
        spawn = spawn_points[spec.seed % len(spawn_points)]
        self._vehicle = self._world.spawn_actor(vehicle_bp, spawn)

        # Collision and lane-invasion sensors push events asynchronously; they
        # are accumulated into _pending and drained on the next step so an
        # infraction is attributed to the frame the loop observes it in.
        collision_bp = blueprints.find("sensor.other.collision")
        collision = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._vehicle
        )
        collision.listen(self._on_collision)
        self._sensors.append(collision)

        lane_bp = blueprints.find("sensor.other.lane_invasion")
        lane = self._world.spawn_actor(lane_bp, carla.Transform(), attach_to=self._vehicle)
        lane.listen(lambda _event: self._pending.append(Infraction.LANE_INVASION))
        self._sensors.append(lane)

        self._frame = 0
        self._time = 0.0
        self._distance = 0.0
        self._stopped_seconds = 0.0
        self._pending = []
        self._last_location = self._vehicle.get_location()

        self._world.tick()
        return self._state()

    def _on_collision(self, event) -> None:
        other = getattr(event, "other_actor", None)
        type_id = getattr(other, "type_id", "") or ""
        if type_id.startswith("walker"):
            self._pending.append(Infraction.COLLISION_PEDESTRIAN)
        elif type_id.startswith("vehicle"):
            self._pending.append(Infraction.COLLISION_VEHICLE)
        else:
            self._pending.append(Infraction.COLLISION_STATIC)

    def _state(self) -> FrameState:
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        nearest = float("inf")
        detections = 0
        ego_location = transform.location
        for actor in self._world.get_actors().filter("*vehicle*"):
            if actor.id == self._vehicle.id:
                continue
            distance = ego_location.distance(actor.get_location())
            if distance < 50.0:
                detections += 1
                nearest = min(nearest, distance)

        return FrameState(
            frame_index=self._frame,
            simulation_time=self._time,
            x=transform.location.x,
            y=transform.location.y,
            yaw_degrees=transform.rotation.yaw,
            speed_mps=speed,
            command=Command.FOLLOW_LANE,
            distance_travelled_m=self._distance,
            nearest_object_m=nearest,
            detections=detections,
        )

    def step(self, throttle: float, steer: float, brake: float) -> StepResult:
        if self._spec is None or self._vehicle is None:
            raise RuntimeError("step() called before reset()")
        carla = self._carla

        self._vehicle.apply_control(
            carla.VehicleControl(
                throttle=float(min(max(throttle, 0.0), 1.0)),
                steer=float(min(max(steer, -1.0), 1.0)),
                brake=float(min(max(brake, 0.0), 1.0)),
            )
        )
        self._world.tick()
        self._frame += 1
        self._time += self._spec.delta_seconds

        location = self._vehicle.get_location()
        if self._last_location is not None:
            self._distance += location.distance(self._last_location)
        self._last_location = location

        infractions, self._pending = self._pending, []
        state = self._state()

        done = False
        reason = ""
        if state.speed_mps < 0.1:
            self._stopped_seconds += self._spec.delta_seconds
            if self._stopped_seconds >= 20.0:
                infractions.append(Infraction.AGENT_BLOCKED)
                done, reason = True, "agent blocked"
        else:
            self._stopped_seconds = 0.0

        if self._distance >= self._spec.route_length_m:
            done, reason = True, "route completed"
        elif self._frame >= self._spec.max_steps:
            done, reason = True, "max steps reached"

        return StepResult(state=state, done=done, infractions=infractions, reason=reason)

    def _cleanup_actors(self) -> None:
        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:  # actor may already be gone after a world reload
                pass
        self._sensors.clear()
        if self._vehicle is not None:
            try:
                self._vehicle.destroy()
            except Exception:
                pass
            self._vehicle = None

    def close(self) -> None:
        self._cleanup_actors()
        # Restore async mode: these settings are server-global, and leaving a
        # shared CARLA in synchronous mode hangs every other client.
        if self._world is not None and self._original_settings is not None:
            try:
                self._world.apply_settings(self._original_settings)
            except Exception as error:
                logger.warning("failed to restore CARLA world settings: %s", error)
        self._world = None


def build_simulator(backend: str = "auto", **kwargs) -> SimulatorBackend:
    """Construct a simulator backend.

    Args:
        backend: ``"auto"`` prefers CARLA and falls back to kinematic,
            ``"carla"`` requires CARLA, ``"kinematic"`` forces the portable one.

    Raises:
        RuntimeError: If ``"carla"`` is requested but unavailable.
        ValueError: On an unknown backend name.
    """
    normalized = backend.strip().lower()
    if normalized == "kinematic":
        return KinematicSimulator()
    if normalized == "carla":
        if not carla_available():
            raise RuntimeError(
                "the carla package is not importable. Install the CARLA client "
                "matching your server, or use backend='kinematic'."
            )
        return CarlaSimulator(**kwargs)
    if normalized == "auto":
        if carla_available():
            try:
                return CarlaSimulator(**kwargs)
            except Exception as error:
                logger.warning(
                    "CARLA is importable but a simulator could not be created (%s); "
                    "falling back to the kinematic backend", error
                )
        else:
            logger.info(
                "CARLA unavailable (not installed, or unsupported on this platform); "
                "using the kinematic backend. Pipeline metrics are valid; "
                "driving-quality metrics are not comparable to CARLA results."
            )
        return KinematicSimulator()
    raise ValueError(f"unknown simulator backend {backend!r}; expected auto, carla, or kinematic")
