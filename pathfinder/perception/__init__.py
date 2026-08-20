"""Perception: turning what the camera sees into what the Policy consumes."""
from pathfinder.perception.geometry import (
    KINEMATIC_CAMERA,
    Box,
    CameraGeometry,
    range_from_box,
)

__all__ = [
    "KINEMATIC_CAMERA",
    "Box",
    "CameraGeometry",
    "range_from_box",
]
