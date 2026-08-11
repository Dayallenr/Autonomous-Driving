"""
CARLA simulator backend.

Used when the ``carla`` package is importable and a server is reachable. CARLA
does not ship for Apple Silicon and requires a GPU, so :func:`build_simulator`
falls back to the kinematic backend and says so loudly rather than failing at
import time — the rest of the stack is fully exercisable without it.

What an episode consists of
---------------------------
A planned route between two spawn points, driven in a world containing traffic
and pedestrians, observed through a forward camera, scored by the CARLA
Leaderboard metric. Each of those is load-bearing:

* **A route**, not free driving. Route completion is half the driving score, and
  it is meaningless without a defined route to complete. Geometry lives in
  :mod:`pathfinder.sim.route` so it is testable without a simulator.
* **Navigation commands** derived from the route's ``RoadOption``s. The planner
  has four branches; feeding it a constant ``FOLLOW_LANE`` leaves three of them
  untrained and makes per-branch infraction analysis impossible.
* **Traffic**, because a benchmark in an empty town measures lane keeping.
  Collision, occlusion, and yielding only exist when something else is moving.
* **A camera**, because the planner is a vision policy. Without pixels it cannot
  run at all.

Synchronous mode
----------------
The world runs in synchronous mode with a fixed delta. Asynchronously the server
ticks on its own clock, so a slow perception step means the control loop acts on
stale state and the episode is not reproducible. Benchmarking that way measures
the host's spare capacity as much as the planner. Settings are server-global, so
the originals are restored on ``close`` — leaving a shared CARLA in synchronous
mode hangs every other client waiting for ticks that never come.

Determinism
-----------
Measured on CARLA 0.9.16: two runs of one seed produced byte-identical
trajectories (0.0 m divergence over 120 ticks). That holds only with the fixed
delta, a seeded traffic manager, and synchronous mode all set, which is why they
are configured together in :meth:`CarlaSimulator.reset` rather than left to the
caller.
"""
from __future__ import annotations

import logging
import math
import queue
import random

from pathfinder.sim.base import (
    EpisodeSpec,
    FrameState,
    Infraction,
    SimulatorBackend,
    StepResult,
)
from pathfinder.sim.carla_paths import ensure_agents_importable
from pathfinder.sim.kinematic import KinematicSimulator
from pathfinder.sim.render import RENDER_HEIGHT, RENDER_WIDTH
from pathfinder.sim.route import RoutePoint, RouteTracker, road_option_to_command

logger = logging.getLogger(__name__)

__all__ = ["CarlaSimulator", "build_simulator", "carla_available"]

_WEATHER_PRESETS = (
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
    "MidRainyNoon", "HardRainNoon", "SoftRainNoon",
    "ClearSunset", "CloudySunset", "WetSunset",
)

#: Waypoint spacing for the planned route, metres. Fine enough that cross-track
#: error is not dominated by discretisation, coarse enough that a 500 m route is
#: a few hundred points rather than tens of thousands.
ROUTE_RESOLUTION_M = 2.0

#: Seconds below :data:`STOPPED_SPEED_MPS` before the episode is abandoned. The
#: Leaderboard uses a comparable blocked-agent timeout; without one a planner
#: that stalls forever consumes a worker for the whole run.
BLOCKED_TIMEOUT_S = 30.0
STOPPED_SPEED_MPS = 0.1

#: Range within which a traffic light is reported to the planner.
TRAFFIC_LIGHT_RANGE_M = 30.0


def carla_available() -> bool:
    """True when the CARLA Python API can be imported."""
    try:
        import carla  # noqa: F401

        return True
    except ImportError:
        return False


class CarlaSimulator(SimulatorBackend):
    """Real CARLA backend: routed episodes with traffic, camera, and scoring."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 2000,
        timeout_seconds: float = 30.0,
        traffic_manager_port: int = 8000,
        render_camera: bool = True,
    ) -> None:
        import carla

        self._carla = carla
        self._client = carla.Client(host, port)
        self._client.set_timeout(timeout_seconds)
        self._traffic_manager_port = traffic_manager_port
        self._render_camera = render_camera

        self._world = None
        self._loaded_town: str | None = None
        self._original_settings = None

        self._vehicle = None
        self._sensors: list = []
        self._traffic: list = []
        self._walkers: list = []
        self._walker_controllers: list = []
        self._image_queue: queue.Queue = queue.Queue()

        self._spec: EpisodeSpec | None = None
        self._route: RouteTracker | None = None
        self._frame = 0
        self._time = 0.0
        self._distance = 0.0
        self._last_location = None
        self._pending: list[Infraction] = []
        self._stopped_seconds = 0.0
        self._latest_image = None
        self._was_at_red = False

    @property
    def name(self) -> str:
        return "carla"

    # ── world ────────────────────────────────────────────────────────────────

    def _ensure_world(self, town: str):
        """Load ``town`` only when it differs from the one already loaded.

        ``load_world`` costs 10-20 s. Calling it per episode would make a
        1000-episode benchmark spend more wall time loading maps than driving,
        so the world is reused whenever the town is unchanged.
        """
        if self._world is not None and self._loaded_town == town:
            return self._world

        self._world = self._client.load_world(town)
        self._loaded_town = town
        self._original_settings = self._world.get_settings()
        return self._world

    def _configure_determinism(self, spec: EpisodeSpec) -> None:
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = spec.delta_seconds
        # Substepping bounds the physics step regardless of delta. Without it a
        # large delta silently changes integration accuracy, and two runs at
        # different host loads can diverge.
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        settings.deterministic_ragdolls = True
        self._world.apply_settings(settings)

        manager = self._client.get_trafficmanager(self._traffic_manager_port)
        manager.set_synchronous_mode(True)
        # Without a seeded device the traffic manager makes different lane and
        # speed choices every run, and no amount of fixed delta makes the
        # episode reproducible.
        manager.set_random_device_seed(spec.seed)
        self._manager = manager

    # ── route ────────────────────────────────────────────────────────────────

    def _build_route(self, spec: EpisodeSpec, start, rng: random.Random) -> RouteTracker:
        """Plan a route from ``start`` of at least ``spec.route_length_m``.

        Raises:
            RuntimeError: If no destination yields a usable route, which means
                the town's topology cannot support the requested length.
        """
        ensure_agents_importable(raise_on_missing=True)
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        planner = GlobalRoutePlanner(self._world.get_map(), ROUTE_RESOLUTION_M)
        spawn_points = self._world.get_map().get_spawn_points()

        # Try destinations in a seeded order and keep the first route long
        # enough. Sampling rather than taking the farthest point keeps routes
        # varied across episodes while staying reproducible for a given seed.
        candidates = list(spawn_points)
        rng.shuffle(candidates)

        best: list[RoutePoint] = []
        for destination in candidates[:20]:
            if destination.location.distance(start.location) < spec.route_length_m * 0.5:
                continue
            try:
                plan = planner.trace_route(start.location, destination.location)
            except Exception as error:  # topology gaps raise rather than return empty
                logger.debug("route planning failed for a destination: %s", error)
                continue
            if len(plan) < 2:
                continue

            points = [
                RoutePoint(
                    x=waypoint.transform.location.x,
                    y=waypoint.transform.location.y,
                    yaw_rad=math.radians(waypoint.transform.rotation.yaw),
                    command=road_option_to_command(option),
                )
                for waypoint, option in plan
            ]
            if len(points) > len(best):
                best = points
            if len(points) * ROUTE_RESOLUTION_M >= spec.route_length_m:
                break

        if len(best) < 2:
            raise RuntimeError(
                f"could not plan a route of {spec.route_length_m:.0f} m from the chosen "
                f"spawn point in {spec.town}"
            )
        return RouteTracker(best)

    # ── actors ───────────────────────────────────────────────────────────────

    def _spawn_ego(self, spawn) -> None:
        carla = self._carla
        blueprints = self._world.get_blueprint_library()
        vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]

        for _ in range(10):
            self._vehicle = self._world.try_spawn_actor(vehicle_bp, spawn)
            if self._vehicle is not None:
                break
            self._world.tick()
        if self._vehicle is None:
            raise RuntimeError("could not spawn the ego vehicle after 10 attempts")

        collision = self._world.spawn_actor(
            blueprints.find("sensor.other.collision"), carla.Transform(), attach_to=self._vehicle
        )
        collision.listen(self._on_collision)
        self._sensors.append(collision)

        lane = self._world.spawn_actor(
            blueprints.find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=self._vehicle,
        )
        lane.listen(self._on_lane_invasion)
        self._sensors.append(lane)

        if self._render_camera:
            camera_bp = blueprints.find("sensor.camera.rgb")
            # Matched to the kinematic renderer so a policy trained against one
            # backend sees the same geometry in the other.
            camera_bp.set_attribute("image_size_x", str(RENDER_WIDTH))
            camera_bp.set_attribute("image_size_y", str(RENDER_HEIGHT))
            camera_bp.set_attribute("fov", "90")
            camera = self._world.spawn_actor(
                camera_bp,
                carla.Transform(carla.Location(x=1.5, z=2.4)),
                attach_to=self._vehicle,
            )
            camera.listen(self._image_queue.put)
            self._sensors.append(camera)

    def _spawn_traffic(self, spec: EpisodeSpec, rng: random.Random, occupied) -> None:
        """Populate the world with vehicles and pedestrians.

        A benchmark in an empty town measures lane keeping. Collisions,
        occlusion, and yielding behaviour only exist when something else moves.
        """
        carla = self._carla
        blueprints = self._world.get_blueprint_library()
        spawn_points = [p for p in self._world.get_map().get_spawn_points() if p is not occupied]
        rng.shuffle(spawn_points)

        vehicle_count = int(len(spawn_points) * max(0.0, min(spec.traffic_density, 1.0)))
        vehicle_bps = blueprints.filter("vehicle.*")
        for point in spawn_points[:vehicle_count]:
            blueprint = vehicle_bps[rng.randrange(len(vehicle_bps))]
            actor = self._world.try_spawn_actor(blueprint, point)
            if actor is None:
                continue
            actor.set_autopilot(True, self._traffic_manager_port)
            self._traffic.append(actor)

        walker_count = int(60 * max(0.0, min(spec.pedestrian_density, 1.0)))
        walker_bps = blueprints.filter("walker.pedestrian.*")
        controller_bp = blueprints.find("controller.ai.walker")
        for _ in range(walker_count):
            location = self._world.get_random_location_from_navigation()
            if location is None:
                continue
            blueprint = walker_bps[rng.randrange(len(walker_bps))]
            walker = self._world.try_spawn_actor(blueprint, carla.Transform(location))
            if walker is None:
                continue
            controller = self._world.try_spawn_actor(
                controller_bp, carla.Transform(), attach_to=walker
            )
            if controller is None:
                walker.destroy()
                continue
            self._walkers.append(walker)
            self._walker_controllers.append(controller)

        # Controllers must be started after a tick, once the walkers exist in
        # the simulation, or start() silently does nothing.
        self._world.tick()
        for controller in self._walker_controllers:
            controller.start()
            destination = self._world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)

        logger.info(
            "spawned %d vehicles and %d pedestrians", len(self._traffic), len(self._walkers)
        )

    # ── sensor callbacks ─────────────────────────────────────────────────────

    def _on_collision(self, event) -> None:
        type_id = getattr(getattr(event, "other_actor", None), "type_id", "") or ""
        if type_id.startswith("walker"):
            self._pending.append(Infraction.COLLISION_PEDESTRIAN)
        elif type_id.startswith("vehicle"):
            self._pending.append(Infraction.COLLISION_VEHICLE)
        else:
            self._pending.append(Infraction.COLLISION_STATIC)

    def _on_lane_invasion(self, event) -> None:
        """Record only solid-line crossings.

        CARLA fires this sensor on every lane marking, including the broken
        centre lines a car legitimately crosses when changing lanes. Counting
        those would make lane invasions dominate every episode's infraction
        totals and drown out the violations that matter.
        """
        carla = self._carla
        solid = {
            carla.LaneMarkingType.Solid,
            carla.LaneMarkingType.SolidSolid,
            carla.LaneMarkingType.Curb,
        }
        if any(marking.type in solid for marking in event.crossed_lane_markings):
            self._pending.append(Infraction.LANE_INVASION)

    # ── episode ──────────────────────────────────────────────────────────────

    def reset(self, spec: EpisodeSpec) -> FrameState:
        carla = self._carla
        self._spec = spec
        rng = random.Random(spec.seed)

        self._cleanup_actors()
        self._ensure_world(spec.town)
        self._configure_determinism(spec)

        if spec.weather in _WEATHER_PRESETS:
            self._world.set_weather(getattr(carla.WeatherParameters, spec.weather))

        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"town {spec.town} has no spawn points")
        start = spawn_points[spec.seed % len(spawn_points)]

        self._route = self._build_route(spec, start, rng)
        self._spawn_ego(start)
        self._spawn_traffic(spec, rng, start)

        self._frame = 0
        self._time = 0.0
        self._distance = 0.0
        self._stopped_seconds = 0.0
        self._pending = []
        self._latest_image = None
        self._was_at_red = False
        self._last_location = self._vehicle.get_location()

        self._world.tick()
        self._drain_image()
        return self._state()

    def _drain_image(self) -> None:
        """Take the newest camera frame for this tick.

        In synchronous mode exactly one image is produced per tick, but a queue
        can hold a backlog after a slow step. Draining to the newest keeps the
        observation aligned with the current state rather than lagging behind
        it, which would make the policy act on the past.
        """
        if not self._render_camera:
            return
        image = None
        while True:
            try:
                image = self._image_queue.get_nowait()
            except queue.Empty:
                break
        if image is None:
            return

        import numpy as np

        buffer = np.frombuffer(image.raw_data, dtype=np.uint8)
        # CARLA delivers BGRA; drop alpha and reverse to RGB.
        self._latest_image = buffer.reshape(image.height, image.width, 4)[:, :, :3][:, :, ::-1]

    def _traffic_light_ahead(self) -> tuple[str, float]:
        light = self._vehicle.get_traffic_light()
        if light is None:
            return "", float("inf")
        distance = self._vehicle.get_location().distance(light.get_transform().location)
        if distance > TRAFFIC_LIGHT_RANGE_M:
            return "", float("inf")
        return str(light.get_state()).split(".")[-1].lower(), distance

    def _state(self) -> FrameState:
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        yaw = math.radians(transform.rotation.yaw)

        tracking = self._route.update(transform.location.x, transform.location.y, yaw)

        nearest = float("inf")
        detections = 0
        ego_location = transform.location
        for actor in list(self._world.get_actors().filter("*vehicle*")) + list(
            self._world.get_actors().filter("*walker*")
        ):
            if actor.id == self._vehicle.id:
                continue
            distance = ego_location.distance(actor.get_location())
            if distance < 50.0:
                detections += 1
                nearest = min(nearest, distance)

        light_state, light_distance = self._traffic_light_ahead()

        return FrameState(
            frame_index=self._frame,
            simulation_time=self._time,
            x=transform.location.x,
            y=transform.location.y,
            yaw_degrees=transform.rotation.yaw,
            speed_mps=speed,
            command=tracking["command"],
            distance_travelled_m=self._distance,
            nearest_object_m=nearest,
            detections=detections,
            lateral_error_m=tracking["lateral_error_m"],
            heading_error_rad=tracking["heading_error_rad"],
            lookahead_curvature=tracking["lookahead_curvature"],
            traffic_light_state=light_state,
            traffic_light_distance_m=light_distance,
            image=self._latest_image,
        )

    def _check_red_light(self, state: FrameState) -> bool:
        """Detect crossing a stop line while the light is red.

        The transition matters, not the instantaneous state: a car legitimately
        waits at a red for many frames, and counting each of those as a
        violation would make the infraction count a measure of patience. The
        violation is leaving the light's influence, at speed, without it having
        turned green.
        """
        at_red = state.traffic_light_state == "red"
        ran = (
            self._was_at_red
            and not at_red
            and state.traffic_light_state != "green"
            and state.speed_mps > 1.0
        )
        self._was_at_red = at_red
        return ran

    def step(self, throttle: float, steer: float, brake: float) -> StepResult:
        if self._spec is None or self._vehicle is None or self._route is None:
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
        self._drain_image()

        location = self._vehicle.get_location()
        if self._last_location is not None:
            self._distance += location.distance(self._last_location)
        self._last_location = location

        infractions, self._pending = self._pending, []
        state = self._state()

        if self._check_red_light(state):
            infractions.append(Infraction.RED_LIGHT)

        done = False
        reason = ""

        if state.speed_mps < STOPPED_SPEED_MPS:
            self._stopped_seconds += self._spec.delta_seconds
            if self._stopped_seconds >= BLOCKED_TIMEOUT_S:
                infractions.append(Infraction.AGENT_BLOCKED)
                done, reason = True, "agent blocked"
        else:
            self._stopped_seconds = 0.0

        if self._route.completion >= 0.99:
            done, reason = True, "route completed"
        elif abs(state.lateral_error_m) > 30.0:
            infractions.append(Infraction.ROUTE_DEVIATION)
            done, reason = True, "route deviation"
        elif self._frame >= self._spec.max_steps:
            done, reason = True, "max steps reached"

        return StepResult(state=state, done=done, infractions=infractions, reason=reason)

    @property
    def route_completion(self) -> float:
        """Fraction of the planned route covered — the driving score's first half."""
        return self._route.completion if self._route else 0.0

    # ── teardown ─────────────────────────────────────────────────────────────

    def _cleanup_actors(self) -> None:
        for controller in self._walker_controllers:
            try:
                controller.stop()
                controller.destroy()
            except Exception:
                pass
        self._walker_controllers.clear()

        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:  # may already be gone after a world reload
                pass
        self._sensors.clear()

        # Batch-destroy the crowd: one RPC per actor makes teardown of a few
        # hundred actors take longer than the episode did.
        doomed = [actor for actor in self._traffic + self._walkers if actor is not None]
        if doomed and self._client is not None:
            try:
                self._client.apply_batch(
                    [self._carla.command.DestroyActor(actor) for actor in doomed]
                )
            except Exception as error:
                logger.debug("batch destroy failed: %s", error)
        self._traffic.clear()
        self._walkers.clear()

        if self._vehicle is not None:
            try:
                self._vehicle.destroy()
            except Exception:
                pass
            self._vehicle = None

        while not self._image_queue.empty():
            try:
                self._image_queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self._cleanup_actors()
        # Restore async mode: settings are server-global, and leaving a shared
        # CARLA synchronous hangs every other client.
        if self._world is not None and self._original_settings is not None:
            try:
                self._world.apply_settings(self._original_settings)
                self._client.get_trafficmanager(
                    self._traffic_manager_port
                ).set_synchronous_mode(False)
            except Exception as error:
                logger.warning("failed to restore CARLA world settings: %s", error)
        self._world = None
        self._loaded_town = None


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
