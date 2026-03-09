"""
AutonomousAgent: orchestrates all ADS modules in the CARLA simulation loop.

Per-tick execution order:
  1. world.tick()                — advance the simulation one step
  2. SensorHub.snapshot()       — collect latest sensor readings
  3. EKFLocalizer.predict/update — update ego pose estimate
  4. YOLODetector.detect()      — detect objects in the camera frame
  5. Predictor.update()         — track + predict agent futures
  6. GlobalRoutePlanner query   — get current high-level navigation command
  7. CILPlanner.plan()          — predict target waypoints
  8. VehicleController.run_step() — compute throttle/brake/steer
  9. vehicle.apply_control()    — send command to CARLA

An optional pygame HUD window visualises the camera feed and detections.
"""
from __future__ import annotations

import math
import sys
import time
import numpy as np
from typing import List, Optional, Tuple

from sensors.sensor_hub import SensorHub, SensorData
from perception.detector import YOLODetector
from perception.utils import Detection, draw_detections
from localization.ekf import EKFLocalizer, EgoPose
from prediction.predictor import Predictor, AgentPrediction
from planning.planner import CILPlanner
from control.pid_controller import VehicleController


class AutonomousAgent:
    """
    Full autonomous driving agent that runs inside a CARLA world.

    Args:
        world:   carla.World instance (synchronous mode must be enabled).
        vehicle: carla.Actor ego vehicle.
        cfg:     Parsed config dict from config/config.yaml.
        destination: carla.Location goal for the route planner.
        enable_hud: Show pygame camera + detection overlay window.
    """

    def __init__(self, world, vehicle, cfg: dict, destination=None, enable_hud: bool = True) -> None:
        import carla

        self._world   = world
        self._vehicle = vehicle
        self._cfg     = cfg
        self._map     = world.get_map()

        # ---- Sensors ----
        self._sensors = SensorHub(world, vehicle, cfg)

        # ---- Modules ----
        self._detector    = YOLODetector(
            model_path           = cfg["perception"]["model_path"],
            confidence_threshold = cfg["perception"]["confidence_threshold"],
            iou_threshold        = cfg["perception"]["iou_threshold"],
            input_size           = cfg["perception"]["input_size"],
            device               = cfg["perception"]["device"],
        )
        self._localizer   = EKFLocalizer(cfg)
        self._predictor   = Predictor(cfg)
        self._planner     = CILPlanner(cfg)
        self._controller  = VehicleController(cfg)

        # ---- Route planner ----
        wp_spacing = cfg.get("planning", {}).get("waypoint_spacing", 2.0)
        from carla import GlobalRoutePlanner
        self._route_planner = GlobalRoutePlanner(self._map, sampling_resolution=wp_spacing)
        self._destination   = destination or self._random_destination()
        self._route         = []

        # ---- State ----
        self._ego_pose: Optional[EgoPose] = None
        self._step = 0

        # ---- HUD ----
        self._hud_enabled = enable_hud
        self._hud_surface = None
        if enable_hud:
            self._init_hud(
                cfg["sensors"]["camera"]["width"],
                cfg["sensors"]["camera"]["height"],
            )

        print("[Agent] Initialised. Waiting for first sensor data...")

    # ------------------------------------------------------------------
    # Main loop step
    # ------------------------------------------------------------------

    def tick(self) -> bool:
        """
        Advance the agent by one simulation step.

        Returns:
            True to continue, False to stop (destination reached or error).
        """
        import carla

        self._world.tick()
        data: SensorData = self._sensors.snapshot()

        if not data.is_complete:
            # Not all sensors have fired yet — send zero control
            self._vehicle.apply_control(carla.VehicleControl())
            return True

        # 1. Localization (EKF)
        if data.imu is not None:
            self._localizer.predict(data.imu)
        if data.gnss is not None:
            self._localizer.update(data.gnss)
        self._ego_pose = self._localizer.get_pose(data.timestamp)

        # 2. Perception
        detections: List[Detection] = []
        if data.camera_frame is not None:
            detections = self._detector.detect(data.camera_frame)

        # 3. Prediction
        predictions: List[AgentPrediction] = self._predictor.update(detections)

        # 4. High-level command from route planner
        command = self._get_navigation_command()

        # 5. Planning
        waypoints: List[Tuple[float, float]] = []
        if data.camera_frame is not None:
            try:
                waypoints = self._planner.plan(data.camera_frame, command)
            except Exception as e:
                print(f"[Agent] Planner error: {e}")
                num_wp = self._cfg.get("planning", {}).get("waypoint_lookahead", 5)
                waypoints = self._planner.plan_fallback(num_wp)
        else:
            num_wp = self._cfg.get("planning", {}).get("waypoint_lookahead", 5)
            waypoints = self._planner.plan_fallback(num_wp)

        # 6. Speed adjustment based on detections
        target_speed = self._controller.compute_target_speed(detections)

        # 7. Control
        current_speed_kmh = self._get_current_speed_kmh()
        control = self._controller.run_step(waypoints, current_speed_kmh, target_speed)
        self._vehicle.apply_control(control)

        # 8. HUD update
        if self._hud_enabled and data.camera_frame is not None:
            vis = draw_detections(data.camera_frame, detections)
            self._render_hud(vis, current_speed_kmh, target_speed, command)

        # 9. Check destination
        if self._destination_reached():
            print("[Agent] Destination reached!")
            return False

        self._step += 1
        if self._step % 100 == 0:
            self._print_status(current_speed_kmh, command, len(detections), len(predictions))

        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_navigation_command(self) -> int:
        """Query GlobalRoutePlanner for the current high-level command."""
        import carla
        from carla import RoadOption

        try:
            vehicle_loc = self._vehicle.get_location()
            route = self._route_planner.trace_route(vehicle_loc, self._destination)
            if len(route) > 1:
                option = route[1][1]
                mapping = {
                    RoadOption.LANEFOLLOW:   0,
                    RoadOption.LEFT:         1,
                    RoadOption.RIGHT:        2,
                    RoadOption.STRAIGHT:     3,
                    RoadOption.CHANGELANELEFT:  1,
                    RoadOption.CHANGELANERIGHT: 2,
                }
                return mapping.get(option, 0)
        except Exception:
            pass
        return 0  # default: follow lane

    def _get_current_speed_kmh(self) -> float:
        v = self._vehicle.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6

    def _destination_reached(self) -> bool:
        loc = self._vehicle.get_location()
        dist = loc.distance(self._destination)
        return dist < 5.0

    def _random_destination(self):
        spawn_points = self._map.get_spawn_points()
        import random
        return random.choice(spawn_points).location

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _init_hud(self, width: int, height: int) -> None:
        try:
            import pygame
            pygame.init()
            self._hud_surface = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Autonomous Driving Agent")
            self._pygame = pygame
            print("[HUD] pygame window initialised.")
        except ImportError:
            print("[HUD] pygame not available — HUD disabled.")
            self._hud_enabled = False

    def _render_hud(
        self,
        bgr_frame: np.ndarray,
        speed_kmh: float,
        target_kmh: float,
        command: int,
    ) -> None:
        if not self._hud_enabled or self._hud_surface is None:
            return
        pygame = self._pygame
        import cv2

        # BGR → RGB for pygame
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self._hud_surface.blit(surface, (0, 0))

        # Overlay: speed and command
        cmd_names = {0: "LANE", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}
        font = pygame.font.SysFont("monospace", 18, bold=True)
        lines = [
            f"Speed: {speed_kmh:.1f} / {target_kmh:.1f} km/h",
            f"Command: {cmd_names.get(command, '?')}",
            f"Step: {self._step}",
        ]
        for i, line in enumerate(lines):
            text_surf = font.render(line, True, (255, 255, 100))
            self._hud_surface.blit(text_surf, (10, 10 + i * 22))

        pygame.display.flip()

        # Process window events to keep it responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("HUD window closed.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _print_status(
        self,
        speed_kmh: float,
        command: int,
        num_detections: int,
        num_tracks: int,
    ) -> None:
        cmd_names = {0: "LANE", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}
        pose = self._ego_pose
        pose_str = f"({pose.x:.1f}, {pose.y:.1f}, hdg={math.degrees(pose.heading):.1f}°)" if pose else "N/A"
        print(
            f"[Agent][Step {self._step:6d}] "
            f"speed={speed_kmh:5.1f}km/h  cmd={cmd_names.get(command,'?'):8s}  "
            f"detections={num_detections}  tracks={num_tracks}  pose={pose_str}"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        self._sensors.destroy()
        if self._hud_enabled:
            try:
                self._pygame.quit()
            except Exception:
                pass
        print("[Agent] Destroyed.")
