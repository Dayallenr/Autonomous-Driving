"""
RGB camera sensor setup and frame acquisition for CARLA 0.9.14.
"""
from __future__ import annotations

import numpy as np
import weakref
import threading
from typing import Callable, Optional


class RGBCamera:
    """
    Attaches an RGB camera to a CARLA vehicle and provides the latest frame
    as a numpy BGR array (matching OpenCV convention).
    """

    def __init__(
        self,
        world,
        vehicle,
        width: int = 800,
        height: int = 600,
        fov: float = 90.0,
        position: tuple = (1.5, 0.0, 2.4),
        rotation: tuple = (0.0, 0.0, 0.0),
    ) -> None:
        import carla

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0
        self._callback: Optional[Callable] = None

        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", str(fov))

        spawn_point = carla.Transform(
            carla.Location(x=position[0], y=position[1], z=position[2]),
            carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2]),
        )

        self._sensor = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda data: RGBCamera._on_image(weak_self, data))

        self.width = width
        self.height = height

    @staticmethod
    def _on_image(weak_self, image) -> None:
        self = weak_self()
        if self is None:
            return
        # CARLA delivers BGRA; convert to BGR for OpenCV
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3]
        with self._lock:
            self._frame = bgr.copy()
            self._frame_timestamp = image.timestamp
        if self._callback is not None:
            self._callback(bgr, image.timestamp)

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest BGR frame (or None if not yet available)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_timestamp(self) -> float:
        with self._lock:
            return self._frame_timestamp

    def set_callback(self, fn: Callable) -> None:
        self._callback = fn

    def destroy(self) -> None:
        if self._sensor is not None and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
