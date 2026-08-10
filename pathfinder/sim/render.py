"""
Forward-view renderer for the kinematic backend.

Why render at all
-----------------
The CIL planner is a ResNet-18 over camera images. Without images the kinematic
backend can exercise orchestration and scoring but not the actual learned
planner, which would leave the largest ML claim in the project untestable
anywhere CARLA does not run.

So the kinematic backend renders a synthetic forward view: lane edges, the route
centreline, and obstacles, projected through a pinhole camera. The output is
``(H, W, 3) uint8`` — byte-identical in shape and dtype to CARLA's RGB camera —
so the CIL model, its transforms, and the DAgger loop are the same code on both
backends.

What this does and does not establish
-------------------------------------
It establishes that the architecture trains, that the four branches specialize
by command, and that DAgger reduces expert-student disagreement. Those are
properties of the *method*.

It does not establish real-world perception performance. These renders have no
texture, no lighting, no weather, and no sensor noise, so a model trained only on
them would transfer poorly to real images. Treating a number from this renderer
as a driving-quality result would be wrong, and the demo labels it accordingly.

Projection
----------
Standard pinhole with the camera at ``CAMERA_HEIGHT_M`` looking along the ego's
x-axis. A ground point at forward distance ``d`` and lateral offset ``l``
projects to::

    u = cx - f · (l / d)
    v = cy + f · (h / d)

Points with ``d <= NEAR_PLANE_M`` are dropped: as ``d`` approaches zero the
projection diverges, and a near-zero denominator produces coordinates that
overflow the canvas and draw garbage across the whole image.
"""
from __future__ import annotations

import numpy as np

__all__ = ["RENDER_HEIGHT", "RENDER_WIDTH", "render_forward_view"]

#: 200x88 is the input resolution used by the original CIL paper; keeping it
#: means the published architecture's stride/pooling arithmetic works unchanged.
RENDER_WIDTH = 200
RENDER_HEIGHT = 88

CAMERA_HEIGHT_M = 1.4
FOCAL_LENGTH_PX = 110.0
NEAR_PLANE_M = 2.0
FAR_PLANE_M = 60.0
LANE_HALF_WIDTH_M = 1.75

# Colours (BGR-agnostic; the model sees whatever ordering is consistent).
_SKY = (110, 90, 70)
_ROAD = (60, 60, 60)
_LANE_LINE = (220, 220, 220)
_CENTRE_LINE = (150, 160, 60)
_VEHICLE = (40, 40, 200)
_PEDESTRIAN = (40, 200, 220)
_STATIC = (120, 120, 120)
_RED_LIGHT = (30, 30, 230)
_GREEN_LIGHT = (30, 200, 30)


def _project(forward_m: float, lateral_m: float, height_m: float = 0.0) -> tuple[int, int] | None:
    """Project an ego-frame ground point to pixel coordinates, or None if behind."""
    if forward_m <= NEAR_PLANE_M:
        return None
    u = RENDER_WIDTH / 2.0 - FOCAL_LENGTH_PX * (lateral_m / forward_m)
    v = RENDER_HEIGHT / 2.0 + FOCAL_LENGTH_PX * ((CAMERA_HEIGHT_M - height_m) / forward_m)
    return int(round(u)), int(round(v))


def _path_lateral_at(distance: float, lateral_error: float, heading_error: float,
                     curvature: float) -> float:
    """Lateral offset of the route centreline at forward distance ``distance``.

    Second-order expansion of the path in the ego frame: constant term is the
    current cross-track error, linear term is heading divergence, quadratic term
    is curvature. Exact enough for a render, and it is what makes the image
    actually informative about the upcoming turn — the signal the CIL branches
    need to learn from.
    """
    return -lateral_error - heading_error * distance + 0.5 * curvature * distance**2


def _fill_polygon(canvas: np.ndarray, points: list[tuple[int, int]], colour: tuple) -> None:
    """Fill a convex polygon, clipped to the canvas."""
    if len(points) < 3:
        return
    try:
        import cv2

        cv2.fillPoly(canvas, [np.asarray(points, dtype=np.int32)], colour)
    except ImportError:
        # Bounding-box fallback keeps the renderer usable without OpenCV.
        xs = [max(0, min(RENDER_WIDTH - 1, x)) for x, _ in points]
        ys = [max(0, min(RENDER_HEIGHT - 1, y)) for _, y in points]
        canvas[min(ys) : max(ys) + 1, min(xs) : max(xs) + 1] = colour


def _draw_line(canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int],
               colour: tuple, thickness: int = 1) -> None:
    try:
        import cv2

        cv2.line(canvas, start, end, colour, thickness)
    except ImportError:
        # Nearest-neighbour rasterization; only used without OpenCV.
        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
        for step in range(steps + 1):
            x = int(start[0] + (end[0] - start[0]) * step / steps)
            y = int(start[1] + (end[1] - start[1]) * step / steps)
            if 0 <= x < RENDER_WIDTH and 0 <= y < RENDER_HEIGHT:
                canvas[y, x] = colour


def render_forward_view(
    *,
    lateral_error: float,
    heading_error: float,
    curvature: float,
    obstacles: list[tuple[float, float, str]],
    traffic_light: tuple[str, float] = ("", float("inf")),
) -> np.ndarray:
    """Render the forward camera view.

    Args:
        lateral_error: Signed cross-track error, metres (positive is left).
        heading_error: Ego heading minus path heading, radians.
        curvature: Path curvature ahead, 1/m.
        obstacles: ``(forward_m, lateral_m, kind)`` in the ego frame.
        traffic_light: ``(state, distance_m)`` for the next light.

    Returns:
        ``(RENDER_HEIGHT, RENDER_WIDTH, 3)`` uint8 image.
    """
    canvas = np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8)
    canvas[:, :] = _SKY

    # ── road surface ────────────────────────────────────────────────────────
    # Drawn as a strip of quads rather than one polygon so it follows curvature.
    samples = np.linspace(NEAR_PLANE_M + 0.1, FAR_PLANE_M, 40)
    previous: tuple[tuple[int, int], tuple[int, int]] | None = None
    for distance in samples:
        centre = _path_lateral_at(distance, lateral_error, heading_error, curvature)
        left = _project(distance, centre + LANE_HALF_WIDTH_M)
        right = _project(distance, centre - LANE_HALF_WIDTH_M)
        if left is None or right is None:
            previous = None
            continue
        if previous is not None:
            _fill_polygon(canvas, [previous[0], previous[1], right, left], _ROAD)
        previous = (left, right)

    # ── lane edges and centreline ───────────────────────────────────────────
    for offset, colour, dashed in (
        (LANE_HALF_WIDTH_M, _LANE_LINE, False),
        (-LANE_HALF_WIDTH_M, _LANE_LINE, False),
        (0.0, _CENTRE_LINE, True),
    ):
        last_point: tuple[int, int] | None = None
        for index, distance in enumerate(samples):
            centre = _path_lateral_at(distance, lateral_error, heading_error, curvature)
            point = _project(distance, centre + offset)
            if point is None:
                last_point = None
                continue
            # Dashes give the centreline a longitudinal frequency, which is a
            # far stronger cue for speed than a solid line.
            if last_point is not None and (not dashed or index % 4 < 2):
                _draw_line(canvas, last_point, point, colour, 1)
            last_point = point

    # ── obstacles ───────────────────────────────────────────────────────────
    # Far to near, so nearer objects overdraw farther ones.
    for forward, lateral, kind in sorted(obstacles, key=lambda item: -item[0]):
        if not NEAR_PLANE_M < forward < FAR_PLANE_M:
            continue
        colour = {"vehicle": _VEHICLE, "pedestrian": _PEDESTRIAN}.get(kind, _STATIC)
        width_m, height_m = (1.8, 1.5) if kind == "vehicle" else (0.6, 1.7)

        bottom_left = _project(forward, lateral + width_m / 2)
        bottom_right = _project(forward, lateral - width_m / 2)
        top_left = _project(forward, lateral + width_m / 2, height_m)
        top_right = _project(forward, lateral - width_m / 2, height_m)
        if None in (bottom_left, bottom_right, top_left, top_right):
            continue
        _fill_polygon(canvas, [top_left, top_right, bottom_right, bottom_left], colour)

    # ── traffic light ───────────────────────────────────────────────────────
    state, distance = traffic_light
    if state and NEAR_PLANE_M < distance < FAR_PLANE_M:
        centre = _path_lateral_at(distance, lateral_error, heading_error, curvature)
        # Mounted above the road, so it sits high in frame like a real signal.
        point = _project(distance, centre, height_m=5.0)
        if point is not None:
            radius = max(1, int(40.0 / distance))
            colour = _RED_LIGHT if state == "red" else _GREEN_LIGHT
            x, y = point
            y0, y1 = max(0, y - radius), min(RENDER_HEIGHT, y + radius + 1)
            x0, x1 = max(0, x - radius), min(RENDER_WIDTH, x + radius + 1)
            if y1 > y0 and x1 > x0:
                canvas[y0:y1, x0:x1] = colour

    return canvas
