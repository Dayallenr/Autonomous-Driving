"""
LiDAR sensor setup and point cloud buffering for CARLA 0.9.14.
"""
from __future__ import annotations

import threading
import weakref

import numpy as np


class LiDAR:
    """
    Attaches a semantic LiDAR to a CARLA vehicle and keeps the latest
    point cloud as a (N, 4) float32 array: [x, y, z, intensity].
    """

    def __init__(
        self,
        world,
        vehicle,
        channels: int = 64,
        range_m: float = 50.0,
        points_per_second: int = 100000,
        rotation_frequency: float = 20.0,
        upper_fov: float = 10.0,
        lower_fov: float = -30.0,
        position: tuple = (0.0, 0.0, 2.5),
        rotation: tuple = (0.0, 0.0, 0.0),
    ) -> None:
        import carla

        self._lock = threading.Lock()
        self._points: np.ndarray | None = None
        self._timestamp: float = 0.0

        bp_lib = world.get_blueprint_library()
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(channels))
        lidar_bp.set_attribute("range", str(range_m))
        lidar_bp.set_attribute("points_per_second", str(points_per_second))
        lidar_bp.set_attribute("rotation_frequency", str(rotation_frequency))
        lidar_bp.set_attribute("upper_fov", str(upper_fov))
        lidar_bp.set_attribute("lower_fov", str(lower_fov))

        spawn_point = carla.Transform(
            carla.Location(x=position[0], y=position[1], z=position[2]),
            carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2]),
        )

        self._sensor = world.spawn_actor(lidar_bp, spawn_point, attach_to=vehicle)
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda data: LiDAR._on_lidar(weak_self, data))

    @staticmethod
    def _on_lidar(weak_self, data) -> None:
        self = weak_self()
        if self is None:
            return
        # Each point is (x, y, z, intensity) as float32
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = points.reshape(-1, 4)
        with self._lock:
            self._points = points.copy()
            self._timestamp = data.timestamp

    def get_points(self) -> np.ndarray | None:
        """Return the latest point cloud as (N, 4) float32 [x, y, z, intensity]."""
        with self._lock:
            return self._points.copy() if self._points is not None else None

    def get_timestamp(self) -> float:
        with self._lock:
            return self._timestamp

    def destroy(self) -> None:
        if self._sensor is not None and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
