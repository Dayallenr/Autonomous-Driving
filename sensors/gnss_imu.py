"""
GNSS and IMU sensor setup for CARLA 0.9.14.
"""
from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass


@dataclass
class GNSSMeasurement:
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    timestamp: float = 0.0


@dataclass
class IMUMeasurement:
    # Linear acceleration (m/s²) in vehicle frame
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    # Angular velocity (rad/s) in vehicle frame
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    # CARLA's onboard compass heading (radians north)
    compass: float = 0.0
    timestamp: float = 0.0


class GNSSSensor:
    """Attaches a GNSS sensor to the ego vehicle."""

    def __init__(self, world, vehicle, position: tuple = (0.0, 0.0, 0.0)) -> None:
        import carla

        self._lock = threading.Lock()
        self._measurement = GNSSMeasurement()

        bp_lib = world.get_blueprint_library()
        gnss_bp = bp_lib.find("sensor.other.gnss")

        spawn_point = carla.Transform(carla.Location(*position))
        self._sensor = world.spawn_actor(gnss_bp, spawn_point, attach_to=vehicle)
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda data: GNSSSensor._on_gnss(weak_self, data))

    @staticmethod
    def _on_gnss(weak_self, data) -> None:
        self = weak_self()
        if self is None:
            return
        with self._lock:
            self._measurement = GNSSMeasurement(
                latitude=data.latitude,
                longitude=data.longitude,
                altitude=data.altitude,
                timestamp=data.timestamp,
            )

    def get_measurement(self) -> GNSSMeasurement:
        with self._lock:
            return GNSSMeasurement(
                self._measurement.latitude,
                self._measurement.longitude,
                self._measurement.altitude,
                self._measurement.timestamp,
            )

    def destroy(self) -> None:
        if self._sensor is not None and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()


class IMUSensor:
    """Attaches an IMU sensor to the ego vehicle."""

    def __init__(self, world, vehicle, position: tuple = (0.0, 0.0, 0.0)) -> None:
        import carla

        self._lock = threading.Lock()
        self._measurement = IMUMeasurement()

        bp_lib = world.get_blueprint_library()
        imu_bp = bp_lib.find("sensor.other.imu")

        spawn_point = carla.Transform(carla.Location(*position))
        self._sensor = world.spawn_actor(imu_bp, spawn_point, attach_to=vehicle)
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda data: IMUSensor._on_imu(weak_self, data))

    @staticmethod
    def _on_imu(weak_self, data) -> None:
        self = weak_self()
        if self is None:
            return
        with self._lock:
            self._measurement = IMUMeasurement(
                accel_x=data.accelerometer.x,
                accel_y=data.accelerometer.y,
                accel_z=data.accelerometer.z,
                gyro_x=data.gyroscope.x,
                gyro_y=data.gyroscope.y,
                gyro_z=data.gyroscope.z,
                compass=data.compass,
                timestamp=data.timestamp,
            )

    def get_measurement(self) -> IMUMeasurement:
        with self._lock:
            return IMUMeasurement(
                self._measurement.accel_x,
                self._measurement.accel_y,
                self._measurement.accel_z,
                self._measurement.gyro_x,
                self._measurement.gyro_y,
                self._measurement.gyro_z,
                self._measurement.compass,
                self._measurement.timestamp,
            )

    def destroy(self) -> None:
        if self._sensor is not None and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
