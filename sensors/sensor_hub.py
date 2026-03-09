"""
SensorHub: unified interface that spawns and manages all sensors on the ego
vehicle and provides a single tick-based data snapshot to the agent loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .camera import RGBCamera
from .lidar import LiDAR
from .gnss_imu import GNSSSensor, IMUSensor, GNSSMeasurement, IMUMeasurement


@dataclass
class SensorData:
    """Snapshot of all sensor readings at one simulation tick."""
    camera_frame: Optional[np.ndarray] = None       # BGR (H, W, 3) uint8
    lidar_points: Optional[np.ndarray] = None       # (N, 4) float32 [x,y,z,intensity]
    gnss: Optional[GNSSMeasurement] = None
    imu: Optional[IMUMeasurement] = None
    timestamp: float = 0.0

    @property
    def is_complete(self) -> bool:
        """True when all four sensors have delivered at least one reading."""
        return (
            self.camera_frame is not None
            and self.lidar_points is not None
            and self.gnss is not None
            and self.imu is not None
        )


class SensorHub:
    """
    Spawns all sensors on the ego vehicle, operates them in CARLA synchronous
    mode, and exposes a single `snapshot()` method that returns a `SensorData`
    object populated from the most recent callbacks.
    """

    def __init__(self, world, vehicle, cfg: dict) -> None:
        sensor_cfg = cfg.get("sensors", {})

        cam_cfg = sensor_cfg.get("camera", {})
        self._camera = RGBCamera(
            world,
            vehicle,
            width=cam_cfg.get("width", 800),
            height=cam_cfg.get("height", 600),
            fov=cam_cfg.get("fov", 90.0),
            position=tuple(cam_cfg.get("position", [1.5, 0.0, 2.4])),
            rotation=tuple(cam_cfg.get("rotation", [0.0, 0.0, 0.0])),
        )

        lidar_cfg = sensor_cfg.get("lidar", {})
        self._lidar = LiDAR(
            world,
            vehicle,
            channels=lidar_cfg.get("channels", 64),
            range_m=lidar_cfg.get("range", 50.0),
            points_per_second=lidar_cfg.get("points_per_second", 100000),
            rotation_frequency=lidar_cfg.get("rotation_frequency", 20.0),
            upper_fov=lidar_cfg.get("upper_fov", 10.0),
            lower_fov=lidar_cfg.get("lower_fov", -30.0),
            position=tuple(lidar_cfg.get("position", [0.0, 0.0, 2.5])),
            rotation=tuple(lidar_cfg.get("rotation", [0.0, 0.0, 0.0])),
        )

        gnss_cfg = sensor_cfg.get("gnss", {})
        self._gnss = GNSSSensor(
            world,
            vehicle,
            position=tuple(gnss_cfg.get("position", [0.0, 0.0, 0.0])),
        )

        imu_cfg = sensor_cfg.get("imu", {})
        self._imu = IMUSensor(
            world,
            vehicle,
            position=tuple(imu_cfg.get("position", [0.0, 0.0, 0.0])),
        )

        print("[SensorHub] All sensors spawned successfully.")

    def snapshot(self) -> SensorData:
        """
        Collect the latest data from all sensors into a SensorData snapshot.
        Call this once per simulation tick after `world.tick()`.
        """
        gnss = self._gnss.get_measurement()
        imu = self._imu.get_measurement()
        return SensorData(
            camera_frame=self._camera.get_frame(),
            lidar_points=self._lidar.get_points(),
            gnss=gnss,
            imu=imu,
            timestamp=self._camera.get_timestamp(),
        )

    def destroy(self) -> None:
        """Clean up all sensor actors."""
        for sensor in [self._camera, self._lidar, self._gnss, self._imu]:
            try:
                sensor.destroy()
            except Exception:
                pass
        print("[SensorHub] All sensors destroyed.")
