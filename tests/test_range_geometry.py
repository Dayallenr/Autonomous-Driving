"""
Tests for monocular range estimation from a detected box.

This arithmetic is the kind that fails silently: a sign slip or an off-by-two in
a focal-length term still returns a plausible-looking number of metres, and the
only symptom is a Policy that brakes at the wrong time. The repository has
already shipped one off-by-2 curvature formula, so this module gets round-trip
tests against the renderer's own forward projection rather than hand-computed
expectations that could be wrong in the same direction as the code.

Two bands, tested differently
-----------------------------
Above ``min_measurable_range_m`` the estimate is a measurement and is checked
against the truth to within pixel quantisation. Below it the object's ground
contact has left the frame, the estimate saturates, and the tests pin that
saturation — including the fact that it over-reads. A test suite that only
sampled the band where the method works would be hiding its failure mode.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pathfinder.perception.geometry import (
    KINEMATIC_CAMERA,
    CameraGeometry,
    range_from_box,
)
from pathfinder.sim.render import (
    RENDER_HEIGHT,
    RENDER_WIDTH,
    project_ground_point,
    render_forward_view,
)

#: Distances whose ground contact projects inside the frame, so the estimate is
#: a measurement rather than a saturated floor. The lower end sits just above
#: ``KINEMATIC_CAMERA.min_measurable_range_m`` (3.5 m); the upper end is where
#: one pixel is already worth a metre and the estimate stops being useful.
MEASURABLE_DISTANCES_M = [4.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]

VEHICLE_COLOUR = np.array([40, 40, 200], dtype=np.uint8)


def box_from_projection(forward_m: float, lateral_m: float = 0.0,
                        width_m: float = 1.8, height_m: float = 1.5,
                        ) -> tuple[float, float, float, float]:
    """Project a box-shaped obstacle forward into image space, exactly as the
    renderer does, and return it as an xyxy box.

    Coordinates are left unclipped, so this is the box a camera with an
    unbounded sensor would see. Clipping is what
    :func:`rendered_box` exercises separately.
    """
    bottom_left = project_ground_point(forward_m, lateral_m + width_m / 2)
    bottom_right = project_ground_point(forward_m, lateral_m - width_m / 2)
    top_left = project_ground_point(forward_m, lateral_m + width_m / 2, height_m)
    assert bottom_left is not None and bottom_right is not None and top_left is not None
    return (
        float(bottom_left[0]),
        float(top_left[1]),
        float(bottom_right[0]),
        float(bottom_left[1]),
    )


def rendered_box(forward_m: float) -> tuple[float, float, float, float]:
    """The box a detector could actually draw: render a vehicle and measure its
    footprint in the image, so frame clipping is included."""
    image = render_forward_view(
        lateral_error=0.0,
        heading_error=0.0,
        curvature=0.0,
        obstacles=[(forward_m, 0.0, "vehicle")],
    )
    rows, columns = np.nonzero(np.all(image == VEHICLE_COLOUR, axis=-1))
    assert rows.size, f"the renderer drew no vehicle at {forward_m} m to measure"
    return (float(columns.min()), float(rows.min()), float(columns.max()), float(rows.max()))


def quantisation_tolerance_m(distance_m: float, camera: CameraGeometry = KINEMATIC_CAMERA) -> float:
    """Range error implied by rounding the box's bottom edge to a whole pixel.

    ``d = f·h / (v - cy)``, so ``|dd/dv| = d² / (f·h)`` and a half-pixel
    rounding error becomes ``d² / (2·f·h)`` metres. The estimate is therefore
    accurate to centimetres up close and to metres far away, which is the
    property that matters: braking distance is a near-field quantity.

    This holds only above ``camera.min_measurable_range_m``. Nearer than that,
    the error is set by frame clipping and is far larger.
    """
    return distance_m**2 / (2.0 * camera.focal_px * camera.camera_height_m) + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Round trip against the renderer's forward projection
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("distance_m", MEASURABLE_DISTANCES_M)
def test_round_trip_recovers_the_projected_distance(distance_m):
    box = box_from_projection(distance_m)
    assert range_from_box(box, KINEMATIC_CAMERA) == pytest.approx(
        distance_m, abs=quantisation_tolerance_m(distance_m)
    )


@pytest.mark.parametrize("distance_m", MEASURABLE_DISTANCES_M)
def test_round_trip_through_the_rendered_image(distance_m):
    """The full path: render an obstacle, find its footprint, recover its range."""
    assert range_from_box(rendered_box(distance_m), KINEMATIC_CAMERA) == pytest.approx(
        distance_m, abs=quantisation_tolerance_m(distance_m)
    )


@pytest.mark.parametrize("lateral_m", [-3.0, 0.0, 2.5])
def test_lateral_offset_does_not_change_the_range(lateral_m):
    box = box_from_projection(12.0, lateral_m=lateral_m)
    assert range_from_box(box, KINEMATIC_CAMERA) == pytest.approx(
        12.0, abs=quantisation_tolerance_m(12.0)
    )


@pytest.mark.parametrize("height_m", [0.5, 1.5, 3.0])
def test_object_height_does_not_change_the_range(height_m):
    """Only the bottom edge carries range; the top edge is object size."""
    box = box_from_projection(9.0, height_m=height_m)
    assert range_from_box(box, KINEMATIC_CAMERA) == pytest.approx(
        9.0, abs=quantisation_tolerance_m(9.0)
    )


def test_a_lower_box_is_nearer():
    near = range_from_box(box_from_projection(6.0), KINEMATIC_CAMERA)
    far = range_from_box(box_from_projection(24.0), KINEMATIC_CAMERA)
    assert near < far


# ─────────────────────────────────────────────────────────────────────────────
# The near field, where the method saturates
# ─────────────────────────────────────────────────────────────────────────────


def test_the_measurable_floor_is_the_bottom_of_the_frame():
    """A ground point at the floor projects exactly to the frame's bottom edge."""
    floor = KINEMATIC_CAMERA.min_measurable_range_m
    _, row = project_ground_point(floor, 0.0)
    assert row == pytest.approx(KINEMATIC_CAMERA.image_height_px, abs=1)
    assert MEASURABLE_DISTANCES_M[0] > floor


@pytest.mark.parametrize("distance_m", [2.5, 3.0, 3.5])
def test_a_clipped_obstacle_reads_as_the_floor_and_over_reads(distance_m):
    """Nearer than the floor, the estimate stops tracking the truth.

    The obstacle's ground contact is below the frame, so the detected box stops
    descending and every such obstacle reports roughly the same distance —
    larger than the truth. This is a limit of monocular geometry, not a bug to
    be fixed here, and it is pinned so it cannot regress silently into a claim
    that the near field works.
    """
    estimate = range_from_box(rendered_box(distance_m), KINEMATIC_CAMERA)
    floor = KINEMATIC_CAMERA.min_measurable_range_m
    assert estimate > distance_m, "clipping should over-read, not under-read"
    assert estimate == pytest.approx(floor, abs=0.1)


def test_a_clipped_obstacle_is_still_seen():
    """It over-reads, but it must never read as an empty road."""
    assert math.isfinite(range_from_box(rendered_box(2.5), KINEMATIC_CAMERA))


# ─────────────────────────────────────────────────────────────────────────────
# Degenerate inputs return infinity rather than raising
# ─────────────────────────────────────────────────────────────────────────────


def test_a_box_on_the_horizon_is_infinitely_far():
    horizon = KINEMATIC_CAMERA.principal_y_px
    assert range_from_box((10.0, horizon - 20.0, 40.0, horizon), KINEMATIC_CAMERA) == math.inf


def test_a_box_above_the_horizon_is_infinitely_far():
    horizon = KINEMATIC_CAMERA.principal_y_px
    assert range_from_box((10.0, 0.0, 40.0, horizon - 5.0), KINEMATIC_CAMERA) == math.inf


def test_a_zero_height_box_is_infinitely_far():
    row = KINEMATIC_CAMERA.principal_y_px + 20.0
    assert range_from_box((10.0, row, 40.0, row), KINEMATIC_CAMERA) == math.inf


def test_an_inverted_box_is_infinitely_far():
    cy = KINEMATIC_CAMERA.principal_y_px
    assert range_from_box((40.0, cy + 30.0, 10.0, cy + 10.0), KINEMATIC_CAMERA) == math.inf


@pytest.mark.parametrize(
    "box",
    [
        (-60.0, 60.0, -10.0, 80.0),
        (float(RENDER_WIDTH) + 10.0, 60.0, float(RENDER_WIDTH) + 50.0, 80.0),
        (10.0, float(RENDER_HEIGHT) + 5.0, 40.0, float(RENDER_HEIGHT) + 20.0),
    ],
)
def test_a_box_outside_the_frame_is_infinitely_far(box):
    assert range_from_box(box, KINEMATIC_CAMERA) == math.inf


def test_a_box_overhanging_the_frame_edge_still_has_a_range():
    """Truncation is normal at close range and must not read as 'nothing there'."""
    box = box_from_projection(6.0, width_m=20.0)
    assert box[0] < 0 < box[2], "this box should overhang the left edge"
    assert math.isfinite(range_from_box(box, KINEMATIC_CAMERA))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_coordinates_are_infinitely_far(value):
    assert range_from_box((10.0, 60.0, 40.0, value), KINEMATIC_CAMERA) == math.inf


def test_the_range_is_never_negative():
    for row in range(RENDER_HEIGHT + 20):
        estimate = range_from_box((10.0, float(row) - 1.0, 40.0, float(row)), KINEMATIC_CAMERA)
        assert estimate > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Camera geometry
# ─────────────────────────────────────────────────────────────────────────────


def test_the_kinematic_camera_matches_the_renderer():
    assert KINEMATIC_CAMERA.image_width_px == RENDER_WIDTH
    assert KINEMATIC_CAMERA.image_height_px == RENDER_HEIGHT


def test_a_taller_camera_sees_the_same_pixel_as_further_away():
    tall = CameraGeometry(
        focal_px=KINEMATIC_CAMERA.focal_px,
        camera_height_m=KINEMATIC_CAMERA.camera_height_m * 2,
        principal_y_px=KINEMATIC_CAMERA.principal_y_px,
        image_width_px=RENDER_WIDTH,
        image_height_px=RENDER_HEIGHT,
    )
    box = box_from_projection(10.0)
    assert range_from_box(box, tall) == pytest.approx(
        2 * range_from_box(box, KINEMATIC_CAMERA), rel=1e-9
    )
    assert tall.min_measurable_range_m == pytest.approx(
        2 * KINEMATIC_CAMERA.min_measurable_range_m, rel=1e-9
    )


@pytest.mark.parametrize(
    "field, value",
    [("focal_px", 0.0), ("focal_px", -110.0), ("camera_height_m", 0.0),
     ("image_width_px", 0), ("image_height_px", -1)],
)
def test_an_impossible_camera_is_rejected(field, value):
    settings = {
        "focal_px": 110.0,
        "camera_height_m": 1.4,
        "principal_y_px": 44.0,
        "image_width_px": 200,
        "image_height_px": 88,
    }
    settings[field] = value
    with pytest.raises(ValueError):
        CameraGeometry(**settings)
