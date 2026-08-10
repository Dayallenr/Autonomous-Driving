"""
PID-based vehicle controller for CARLA 0.9.14.

Two independent PID loops:
  - Lateral:      heading error → steering angle
  - Longitudinal: speed error   → throttle / brake

The controller accepts a list of waypoints in the vehicle's local frame
(as produced by CILPlanner) and the current vehicle speed, then outputs
a carla.VehicleControl object.
"""
from __future__ import annotations

import math

import numpy as np


class PIDController:
    """Single-axis PID with integral wind-up clamping."""

    def __init__(self, kp: float, ki: float, kd: float, output_limits: tuple[float, float] = (-1.0, 1.0)) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._min, self._max = output_limits
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        self._integral += error * dt
        # Anti-windup: clamp integral contribution
        self._integral = np.clip(
            self._integral,
            self._min / (self.ki + 1e-9),
            self._max / (self.ki + 1e-9),
        )
        derivative = (error - self._prev_error) / dt
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        return float(np.clip(output, self._min, self._max))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0


class VehicleController:
    """
    Combines lateral and longitudinal PID to generate CARLA VehicleControl.

    Lateral control: steers toward the first (or nearest look-ahead) waypoint
    in the vehicle-local frame using a pure pursuit + PID approach.

    Longitudinal control: adjusts throttle/brake to match target speed,
    optionally reduced near detected objects.
    """

    def __init__(self, cfg: dict) -> None:
        ctrl_cfg = cfg.get("control", {})
        lat  = ctrl_cfg.get("lateral",      {})
        lon  = ctrl_cfg.get("longitudinal", {})

        self._lat_pid = PIDController(
            kp=lat.get("kp", 0.8),
            ki=lat.get("ki", 0.05),
            kd=lat.get("kd", 0.2),
            output_limits=(-lat.get("max_steer", 1.0), lat.get("max_steer", 1.0)),
        )
        self._lon_pid = PIDController(
            kp=lon.get("kp", 0.5),
            ki=lon.get("ki", 0.05),
            kd=lon.get("kd", 0.1),
            output_limits=(-1.0, lon.get("max_throttle", 0.75)),
        )

        self._max_throttle = lon.get("max_throttle", 0.75)
        self._max_brake    = lon.get("max_brake",    0.5)
        self._max_steer    = lat.get("max_steer",    1.0)

        ego_cfg = cfg.get("ego", {})
        self._default_target_speed = ego_cfg.get("target_speed", 30.0)  # km/h

        ctrl_cfg_extra = cfg.get("control", {})
        self._speed_near_pedestrian = ctrl_cfg_extra.get("speed_near_pedestrian", 10.0)
        self._speed_near_vehicle    = ctrl_cfg_extra.get("speed_near_vehicle",    20.0)
        self._proximity_threshold   = ctrl_cfg_extra.get("proximity_threshold",  15.0)

        self._dt = cfg.get("carla", {}).get("fixed_delta_seconds", 0.05)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_step(
        self,
        waypoints: list[tuple[float, float]],
        current_speed_kmh: float,
        target_speed_kmh: float | None = None,
    ):
        """
        Compute VehicleControl from waypoints and current speed.

        Args:
            waypoints:         List of (x, y) in vehicle-local metres.
                               x = forward, y = lateral (left positive).
            current_speed_kmh: Vehicle's current speed in km/h.
            target_speed_kmh:  Desired speed; uses config default if None.

        Returns:
            carla.VehicleControl
        """
        import carla

        if target_speed_kmh is None:
            target_speed_kmh = self._default_target_speed

        steer = self._compute_steer(waypoints)
        throttle, brake = self._compute_throttle_brake(current_speed_kmh, target_speed_kmh)

        control = carla.VehicleControl()
        control.steer    = float(np.clip(steer,    -self._max_steer,    self._max_steer))
        control.throttle = float(np.clip(throttle, 0.0,                 self._max_throttle))
        control.brake    = float(np.clip(brake,    0.0,                 self._max_brake))
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    def compute_target_speed(self, detections) -> float:
        """
        Reduce target speed based on nearby detected agents.

        Args:
            detections: List of Detection objects from the perception module.

        Returns:
            Adjusted target speed in km/h.
        """
        min_speed = self._default_target_speed

        for det in detections:
            # Rough proximity estimate from bounding box height
            # (larger box → closer object)
            bbox_height = float(det.bbox[3] - det.bbox[1])
            # Heuristic: treat objects with bbox_height > threshold as nearby
            if bbox_height > self._proximity_threshold * 3:
                if det.class_name in ("pedestrian", "person_sitting", "cyclist"):
                    min_speed = min(min_speed, self._speed_near_pedestrian)
                elif det.class_name in ("car", "van", "truck", "tram"):
                    min_speed = min(min_speed, self._speed_near_vehicle)

        return min_speed

    def reset(self) -> None:
        self._lat_pid.reset()
        self._lon_pid.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_steer(self, waypoints: list[tuple[float, float]]) -> float:
        if not waypoints:
            return 0.0

        # Use the first reachable waypoint for pure-pursuit steering
        target_x, target_y = waypoints[0]

        # Pure pursuit look-ahead angle: atan2(lateral, forward)
        if abs(target_x) < 1e-3 and abs(target_y) < 1e-3:
            heading_error = 0.0
        else:
            heading_error = math.atan2(target_y, max(target_x, 0.5))

        # Clamp heading error to [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        steer = self._lat_pid.step(heading_error, self._dt)
        return steer

    def _compute_throttle_brake(
        self, current_speed_kmh: float, target_speed_kmh: float
    ) -> tuple[float, float]:
        speed_error = (target_speed_kmh - current_speed_kmh) / 3.6  # convert to m/s error
        output = self._lon_pid.step(speed_error, self._dt)

        if output >= 0.0:
            throttle = output
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(output), self._max_brake)

        return throttle, brake
