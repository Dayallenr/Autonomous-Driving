"""
Monocular range estimation: a detected box in, a distance in metres out.

Why this is a separate module
-----------------------------
It is the inverse of the projection :mod:`pathfinder.sim.render` performs, and
it is the one place in the perception path where a sign slip or a misplaced
focal length still returns a plausible number of metres. Kept here, free of the
perception seam and of model weights, it can be round-tripped against the
renderer's own forward projection on any machine — which is how the arithmetic
gets checked against something other than a hand computation that could be wrong
in the same direction as the code.

The method
----------
A detected object's box bottom edge is assumed to be where the object meets the
road. The renderer projects a ground point at forward distance ``d`` to::

    v = cy + f · h / d

so inverting for the bottom row ``v_bottom`` gives::

    d = f · h / (v_bottom - cy)

where ``f`` is the focal length in pixels, ``h`` the camera height above the
road, and ``cy`` the principal point's row — the horizon, since the camera looks
along the ground plane. Only the bottom edge carries range. The box's width and
height describe the object's size, which without a known object class says
nothing about distance.

Assumptions, and where the estimate degrades
--------------------------------------------
- **Flat ground.** The road is the plane ``z = 0`` through the camera's nadir.
  On a crest or in a dip the recovered range is wrong in the direction of the
  slope, and an uphill road ahead reads as nearer than it is.
- **The box bottom touches the ground.** True for vehicles and pedestrians on
  the road; false for anything occluded at its base or airborne. An occluded
  base sits higher in frame and therefore reads as further away.
- **The box bottom is inside the frame.** See below — this is the assumption
  that fails first, and it fails unsafely.
- **A calibrated camera looking along the ground plane.** Zero pitch and roll,
  known focal length. Pitch error moves the horizon and biases every range in
  the scene at once.
- **Quantisation.** ``|dd/dv| = d² / (f·h)``, so a one-pixel error costs
  centimetres at 5 m and metres at 40 m. The estimate is a near-field
  instrument, which is what braking distance needs it to be.

The near field is where this breaks
-----------------------------------
An object closer than :attr:`CameraGeometry.min_measurable_range_m` has its
ground contact point below the bottom of the frame. The detector can only report
a box clipped to the image, so the bottom edge stops descending, the estimate
stops decreasing, and everything nearer than that reads as *further away than it
is* — the dangerous direction, and the one the vehicle has least time to
recover from. On the kinematic camera the floor is 3.5 m: an obstacle at 2.5 m
is reported at 3.6 m.

Nothing here can fix that. The information is not in the image, and inventing a
smaller number would be guessing. What this module can do is state the floor,
which it does, so that a Policy consuming these ranges can treat anything at or
near the floor as "at least this close, possibly much closer" and keep a braking
margin that covers it. Any driving score produced through this estimator carries
the limitation with it.

Every failure listed above is a reason a driving score under Detector-backed
perception may be worse than under privileged perception. That gap is the
measurement this module exists to make possible; it is not a defect to be tuned
away here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pathfinder.sim.render import (
    CAMERA_HEIGHT_M,
    FOCAL_LENGTH_PX,
    RENDER_HEIGHT,
    RENDER_WIDTH,
)

__all__ = [
    "KINEMATIC_CAMERA",
    "Box",
    "CameraGeometry",
    "range_from_box",
]

#: An ``(x1, y1, x2, y2)`` box in pixels, top-left to bottom-right. This is the
#: ``xyxy`` convention the Detector already emits, so no adapter is needed.
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraGeometry:
    """Pinhole camera parameters needed to invert a ground-plane projection.

    Args:
        focal_px: Focal length in pixels.
        camera_height_m: Height of the camera above the road surface.
        principal_y_px: Principal point row — the horizon, for a camera looking
            along the ground plane.
        image_width_px: Frame width, used to reject boxes outside the frame.
        image_height_px: Frame height, used likewise.

    Raises:
        ValueError: If any dimension is not positive. A zero focal length or a
            camera at road level makes the inversion meaningless rather than
            merely inaccurate, and silently returning infinity everywhere would
            read as "the road is clear".
    """

    focal_px: float
    camera_height_m: float
    principal_y_px: float
    image_width_px: int
    image_height_px: int

    def __post_init__(self) -> None:
        for name in ("focal_px", "camera_height_m", "image_width_px", "image_height_px"):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def min_measurable_range_m(self) -> float:
        """The nearest range this camera can distinguish.

        A ground point at this distance projects exactly to the bottom edge of
        the frame. Anything nearer projects below it, is clipped out of the
        detected box, and is therefore reported as being at roughly this
        distance rather than at its true, smaller one. Treat a range at the
        floor as an upper bound, never as a measurement.
        """
        return self.focal_px * self.camera_height_m / (self.image_height_px - self.principal_y_px)


#: The kinematic backend's synthetic camera, taken from the renderer's own
#: constants so the two cannot drift apart. CARLA's camera is configured
#: separately and gets its own geometry where that backend builds it.
KINEMATIC_CAMERA = CameraGeometry(
    focal_px=FOCAL_LENGTH_PX,
    camera_height_m=CAMERA_HEIGHT_M,
    principal_y_px=RENDER_HEIGHT / 2.0,
    image_width_px=RENDER_WIDTH,
    image_height_px=RENDER_HEIGHT,
)


def range_from_box(box: Box, camera: CameraGeometry) -> float:
    """Estimate the distance to a detected object from its bounding box.

    Args:
        box: ``(x1, y1, x2, y2)`` in pixels, top-left to bottom-right.
        camera: Geometry of the camera that produced the frame.

    Returns:
        Distance in metres to the box's ground contact point, or ``math.inf``
        when the box carries no usable range. Infinity is the honest answer for
        a degenerate box and it is also the value the speed governor already
        reads as "unobstructed", so a malformed detection cannot phantom-brake
        the vehicle. It is never negative and never raises: this runs once per
        detection per frame inside a benchmark that must not abort.

        A box whose bottom edge is clipped by the frame border yields
        ``camera.min_measurable_range_m``, which over-reads the true distance.
        See the module docstring — this is a limit of the method, not a bug in
        the arithmetic.
    """
    x1, y1, x2, y2 = box
    if not all(math.isfinite(coordinate) for coordinate in (x1, y1, x2, y2)):
        return math.inf
    # An inverted or empty box is a detector artefact, not an object with a
    # footprint. A zero-height box in particular has no distinguishable bottom.
    if x2 <= x1 or y2 <= y1:
        return math.inf
    # Wholly outside the frame: nothing was seen there. A box that merely
    # overhangs an edge is kept — a truncated obstacle is still an obstacle,
    # and reporting infinity for one at 2 m would be far worse than reporting
    # the floor.
    if x2 <= 0 or y2 <= 0 or x1 >= camera.image_width_px or y1 >= camera.image_height_px:
        return math.inf
    # At or above the horizon the ray never meets the ground plane, so there is
    # no finite intersection to report.
    rows_below_horizon = y2 - camera.principal_y_px
    if rows_below_horizon <= 0:
        return math.inf
    return camera.focal_px * camera.camera_height_m / rows_below_horizon
