"""
The privileged arm's range convention must match the Detector arm's.

Issue #10 recorded the bias these tests exist to prevent: the privileged arm
once measured 360-degree actor-origin-to-actor-origin 3D distance while the
Detector arm measures forward-only ground-plane range from the camera to the
obstacle's visible base. Under those conventions even a *perfect* detector
disagrees with the baseline by the camera offset plus half the obstacle's
length, and the privileged arm brakes for lateral and rear traffic the
Detector arm is structurally blind to — so part of the measured ablation gap
would be convention, not perception quality.

These tests pin the aligned convention as pure geometry, runnable without
CARLA: forward camera frustum only, ground-plane distance from the camera's
nadir to the nearest visible point of the obstacle's footprint. The
round-trip test at the bottom is the alignment claim itself — a perfect box
and the privileged measurement return the same metres for the same obstacle.
"""
from __future__ import annotations

import math

import pytest

from pathfinder.perception.geometry import range_from_box
from pathfinder.sim.carla_backend import (
    CAMERA_FOV_DEGREES,
    CARLA_CAMERA,
    footprint_corners,
    forward_range_to_footprint,
    to_camera_frame,
)

HALF_FOV_RAD = math.radians(CAMERA_FOV_DEGREES / 2.0)


def box_footprint(
    center_forward_m: float,
    center_lateral_m: float,
    half_length_m: float = 2.0,
    half_width_m: float = 1.0,
    yaw_rad: float = 0.0,
) -> list[tuple[float, float]]:
    return footprint_corners(
        center_forward_m, center_lateral_m, yaw_rad, half_length_m, half_width_m
    )


class TestForwardRange:
    def test_dead_ahead_measures_the_near_face_not_the_origin(self):
        # Centre at 20 m, half-length 2 m: the surface the camera (and a
        # detected box's base) actually sees is at 18 m.
        corners = box_footprint(20.0, 0.0, half_length_m=2.0)
        assert forward_range_to_footprint(corners, HALF_FOV_RAD) == pytest.approx(18.0)

    def test_rotated_footprint_measures_its_rotated_near_face(self):
        # Same vehicle turned 90 degrees: now its half-width faces the camera.
        corners = box_footprint(
            20.0, 0.0, half_length_m=2.0, half_width_m=1.0, yaw_rad=math.pi / 2.0
        )
        assert forward_range_to_footprint(corners, HALF_FOV_RAD) == pytest.approx(19.0)

    def test_traffic_behind_the_camera_is_invisible(self):
        corners = box_footprint(-10.0, 0.0)
        assert forward_range_to_footprint(corners, HALF_FOV_RAD) == math.inf

    def test_lateral_traffic_outside_the_frustum_is_invisible(self):
        # Alongside the ego: nearly pure-lateral, far outside a 90-degree FOV.
        corners = box_footprint(0.5, 10.0, half_length_m=2.0, half_width_m=1.0)
        assert forward_range_to_footprint(corners, HALF_FOV_RAD) == math.inf

    def test_footprint_straddling_the_frustum_edge_measures_its_visible_part(self):
        # Axis-aligned quad from (2,4) to (4,8): every point except the corner
        # (4,4) lies outside the wedge y <= x. The nearest point of the whole
        # footprint is (2,4) at ~4.47 m, but the nearest *visible* point is
        # (4,4) — the range must be measured to what the camera can see.
        corners = [(2.0, 4.0), (4.0, 4.0), (4.0, 8.0), (2.0, 8.0)]
        assert forward_range_to_footprint(corners, HALF_FOV_RAD) == pytest.approx(
            math.hypot(4.0, 4.0)
        )

    def test_camera_inside_a_footprint_reads_zero(self):
        corners = box_footprint(0.0, 0.0)
        assert forward_range_to_footprint(corners, HALF_FOV_RAD) == 0.0

    def test_range_is_never_negative(self):
        for forward in (-15.0, -1.0, 0.0, 3.0, 40.0):
            for lateral in (-20.0, 0.0, 20.0):
                value = forward_range_to_footprint(
                    box_footprint(forward, lateral), HALF_FOV_RAD
                )
                assert value >= 0.0


class TestToCameraFrame:
    def test_camera_frame_puts_forward_along_x(self):
        # Ego at (10, 5) heading +y (yaw 90 degrees), camera 1.5 m ahead of the
        # ego origin, so at (10, 6.5). A world point 20 m further along +y and
        # 2 m to the ego's left (world -x) must land at forward 20, lateral 2.
        (fx, fy), = to_camera_frame([(8.0, 26.5)], 10.0, 6.5, math.pi / 2.0)
        assert fx == pytest.approx(20.0)
        assert fy == pytest.approx(2.0)


class TestConventionAlignment:
    @pytest.mark.parametrize("distance_m", [8.0, 15.0, 30.0])
    def test_perfect_detector_and_privileged_measurement_agree(self, distance_m):
        # The claim issue #10 needs: for an obstacle dead ahead whose ground
        # contact is at `distance_m` from the camera nadir, the privileged
        # measurement and a perfect detection's inverted range are the same
        # number. Any residual ablation gap is then perception, not
        # convention.
        privileged = forward_range_to_footprint(
            box_footprint(distance_m + 2.0, 0.0, half_length_m=2.0), HALF_FOV_RAD
        )

        row = CARLA_CAMERA.principal_y_px + (
            CARLA_CAMERA.focal_px * CARLA_CAMERA.camera_height_m / distance_m
        )
        detected = range_from_box((90.0, row - 20.0, 110.0, row), CARLA_CAMERA)

        assert privileged == pytest.approx(distance_m)
        assert detected == pytest.approx(distance_m)
        assert privileged == pytest.approx(detected)
