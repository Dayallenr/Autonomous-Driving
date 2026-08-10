"""
Extended Kalman Filter (EKF) for ego vehicle localization.

State vector: x = [x, y, z, vx, vy, heading]  (6-DOF)

- Prediction step driven by IMU accelerations and yaw rate.
- Update step driven by GNSS lat/lon/alt measurements (converted to metres
  via an inline UTM-like equirectangular projection anchored at the first fix).

No external dependencies beyond numpy — no filterpy required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sensors.gnss_imu import GNSSMeasurement, IMUMeasurement


@dataclass
class EgoPose:
    """Estimated ego vehicle pose in local metric frame."""
    x: float = 0.0        # metres east from origin
    y: float = 0.0        # metres north from origin
    z: float = 0.0        # metres altitude
    vx: float = 0.0       # velocity east (m/s)
    vy: float = 0.0       # velocity north (m/s)
    heading: float = 0.0  # yaw angle, radians CCW from east
    timestamp: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


# ---------------------------------------------------------------------------
# Inline equirectangular GNSS → metres conversion
# ---------------------------------------------------------------------------
_EARTH_RADIUS_M = 6_371_000.0  # mean earth radius


def _latlon_to_metres(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """
    Convert (lat, lon) to local (x_east, y_north) metres relative to origin
    (lat0, lon0) using a simple equirectangular projection.
    """
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    y = dlat * _EARTH_RADIUS_M
    x = dlon * _EARTH_RADIUS_M * math.cos(math.radians(lat0))
    return x, y


# ---------------------------------------------------------------------------
# EKF implementation
# ---------------------------------------------------------------------------

class EKFLocalizer:
    """
    6-DOF Extended Kalman Filter fusing GNSS and IMU.

    State: [x, y, z, vx, vy, heading]
    """

    DIM_X = 6  # state dimension
    DIM_Z = 3  # GNSS measurement dimension [x_m, y_m, z_m]

    def __init__(self, cfg: dict) -> None:
        loc_cfg = cfg.get("localization", {})

        # State and covariance
        self._x = np.zeros(self.DIM_X)
        self._P = np.eye(self.DIM_X) * 10.0  # initial uncertainty

        # Process noise covariance Q
        q_pos = loc_cfg.get("process_noise_pos", 0.1)
        q_vel = loc_cfg.get("process_noise_vel", 0.5)
        q_hdg = loc_cfg.get("process_noise_heading", 0.05)
        self._Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_hdg]) ** 2

        # GNSS measurement noise covariance R
        gnss_std = loc_cfg.get("gnss_noise_std", 3.0)
        self._R = np.eye(self.DIM_Z) * (gnss_std ** 2)

        # GNSS measurement matrix H (maps state → [x, y, z])
        self._H = np.zeros((self.DIM_Z, self.DIM_X))
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0

        # Origin for GNSS projection — set on first fix
        self._origin_lat: float | None = None
        self._origin_lon: float | None = None
        self._origin_alt: float = 0.0
        self._initialized: bool = False

        self._last_imu_timestamp: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, imu: IMUMeasurement) -> None:
        """
        EKF prediction step using IMU data.
        dt is computed from consecutive IMU timestamps.
        """
        if self._last_imu_timestamp == 0.0:
            self._last_imu_timestamp = imu.timestamp
            return
        dt = imu.timestamp - self._last_imu_timestamp
        self._last_imu_timestamp = imu.timestamp
        if dt <= 0.0 or dt > 1.0:
            return

        x, y, z, vx, vy, heading = self._x

        # Yaw rate from gyroscope z-axis
        yaw_rate = imu.gyro_z

        # New heading
        new_heading = _normalize_angle(heading + yaw_rate * dt)

        # Body-frame acceleration → world frame
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        ax_world = cos_h * imu.accel_x - sin_h * imu.accel_y
        ay_world = sin_h * imu.accel_x + cos_h * imu.accel_y

        new_x = x + vx * dt
        new_y = y + vy * dt
        new_z = z  # vertical handled by GNSS
        new_vx = vx + ax_world * dt
        new_vy = vy + ay_world * dt

        self._x = np.array([new_x, new_y, new_z, new_vx, new_vy, new_heading])

        # Jacobian of f w.r.t. state
        F = np.eye(self.DIM_X)
        F[0, 3] = dt
        F[1, 4] = dt
        F[3, 5] = (-sin_h * imu.accel_x - cos_h * imu.accel_y) * dt
        F[4, 5] = ( cos_h * imu.accel_x - sin_h * imu.accel_y) * dt

        self._P = F @ self._P @ F.T + self._Q

    def update(self, gnss: GNSSMeasurement) -> None:
        """
        EKF update step using a GNSS measurement.
        The first call initialises the projection origin.
        """
        if self._origin_lat is None:
            self._origin_lat = gnss.latitude
            self._origin_lon = gnss.longitude
            self._origin_alt = gnss.altitude
            # Seed the state position directly
            self._x[0] = 0.0
            self._x[1] = 0.0
            self._x[2] = 0.0
            self._initialized = True
            return

        x_m, y_m = _latlon_to_metres(
            gnss.latitude, gnss.longitude,
            self._origin_lat, self._origin_lon,
        )
        z_m = gnss.altitude - self._origin_alt
        z_meas = np.array([x_m, y_m, z_m])

        # Innovation
        y_innov = z_meas - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)

        self._x = self._x + K @ y_innov
        self._x[5] = _normalize_angle(self._x[5])
        self._P = (np.eye(self.DIM_X) - K @ self._H) @ self._P

    def get_pose(self, timestamp: float = 0.0) -> EgoPose:
        x, y, z, vx, vy, heading = self._x
        return EgoPose(x=x, y=y, z=z, vx=vx, vy=vy, heading=heading, timestamp=timestamp)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
