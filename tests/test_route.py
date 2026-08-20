"""
Tests for route tracking geometry.

Route-following bugs are silent: the car still drives, the numbers still look
like numbers, and route completion is quietly wrong. These pin the properties
that make the CARLA Leaderboard driving score mean anything — monotonic
progress, correctly signed errors, and a command that leads the vehicle.
"""
from __future__ import annotations

import math

import pytest

from pathfinder.sim.base import Command
from pathfinder.sim.route import (
    ROUTE_DEVIATION_M,
    RoutePoint,
    RouteTracker,
    road_option_to_command,
)


def straight_route(length: int = 100, spacing: float = 1.0) -> list[RoutePoint]:
    """A route along +x, so lateral error is simply the y offset."""
    return [RoutePoint(x=i * spacing, y=0.0, yaw_rad=0.0) for i in range(length)]


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────


def test_route_length_is_the_sum_of_segments():
    tracker = RouteTracker(straight_route(11, spacing=2.0))
    assert tracker.total_length_m == pytest.approx(20.0)


def test_a_single_point_is_not_a_route():
    with pytest.raises(ValueError, match="at least 2 points"):
        RouteTracker([RoutePoint(0.0, 0.0, 0.0)])


# ─────────────────────────────────────────────────────────────────────────────
# Cross-track error
# ─────────────────────────────────────────────────────────────────────────────


def test_on_route_has_no_lateral_error():
    tracker = RouteTracker(straight_route())
    assert tracker.update(10.0, 0.0, 0.0)["lateral_error_m"] == pytest.approx(0.0, abs=1e-9)


def test_lateral_error_is_positive_to_the_left():
    """Sign is what tells a controller which way to steer. Driving along +x,
    'left' is +y."""
    tracker = RouteTracker(straight_route())
    assert tracker.update(10.0, 2.5, 0.0)["lateral_error_m"] == pytest.approx(2.5)
    assert tracker.update(10.0, -2.5, 0.0)["lateral_error_m"] == pytest.approx(-2.5)


def test_deviation_is_flagged_past_the_threshold():
    tracker = RouteTracker(straight_route())
    assert not tracker.update(10.0, ROUTE_DEVIATION_M - 1.0, 0.0)["deviated"]
    assert tracker.update(10.0, ROUTE_DEVIATION_M + 1.0, 0.0)["deviated"]


# ─────────────────────────────────────────────────────────────────────────────
# Heading error
# ─────────────────────────────────────────────────────────────────────────────


def test_heading_error_is_the_difference_from_lane_heading():
    tracker = RouteTracker(straight_route())
    assert tracker.update(10.0, 0.0, 0.3)["heading_error_rad"] == pytest.approx(0.3)


def test_heading_error_wraps_rather_than_reporting_a_near_full_turn():
    """A car at +179 deg against a lane at -179 deg is 2 deg off, not 358."""
    points = [RoutePoint(x=float(i), y=0.0, yaw_rad=math.radians(179)) for i in range(50)]
    error = RouteTracker(points).update(10.0, 0.0, math.radians(-179))["heading_error_rad"]
    assert abs(error) == pytest.approx(math.radians(2), abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Progress
# ─────────────────────────────────────────────────────────────────────────────


def test_completion_grows_as_the_ego_advances():
    tracker = RouteTracker(straight_route())
    first = tracker.update(0.0, 0.0, 0.0)["completion"]
    second = tracker.update(50.0, 0.0, 0.0)["completion"]
    assert first < second
    assert second == pytest.approx(50 / 99, abs=0.02)


def test_completion_reaches_one_when_driven_to_the_end():
    tracker = RouteTracker(straight_route())
    for x in range(0, 120, 2):
        result = tracker.update(float(x), 0.0, 0.0)
    assert result["completion"] == pytest.approx(1.0)


def test_a_jump_beyond_the_search_window_does_not_credit_progress():
    """Bounded forward search is a safety property, not a limitation.

    An ego that teleports — a respawn, a physics glitch, a corrupt replay —
    must not be handed route completion it never drove. Progress is only
    credited for distance actually covered, one window at a time.
    """
    tracker = RouteTracker(straight_route(500), search_window=10)
    completion = tracker.update(400.0, 0.0, 0.0)["completion"]
    assert completion < 0.05


def test_progress_never_goes_backwards_on_a_self_crossing_route():
    """The reason projection searches a forward window rather than the whole route.

    A figure-eight passes near its own earlier segment. A global nearest-point
    search snaps back to that earlier index, and route completion decreases
    while the car is driving forwards.
    """
    points = []
    for i in range(200):
        t = i * 2 * math.pi / 100
        points.append(
            RoutePoint(x=20 * math.sin(t), y=20 * math.sin(t) * math.cos(t), yaw_rad=0.0)
        )
    tracker = RouteTracker(points, search_window=20)

    completions = []
    for point in points:
        completions.append(tracker.update(point.x, point.y, 0.0)["completion"])

    assert all(
        later >= earlier - 1e-9
        for earlier, later in zip(completions, completions[1:], strict=False)
    ), "route completion decreased while advancing along the route"


def test_index_does_not_jump_past_the_search_window():
    """Bounding the window is what keeps projection O(window) rather than O(route)."""
    tracker = RouteTracker(straight_route(500), search_window=10)
    tracker.update(400.0, 0.0, 0.0)
    assert tracker.index <= 10


# ─────────────────────────────────────────────────────────────────────────────
# Commands and curvature
# ─────────────────────────────────────────────────────────────────────────────


def test_command_comes_from_the_lookahead_not_the_current_point():
    """A policy told 'turn left' once it is already in the junction cannot act."""
    points = straight_route(100)
    points = [
        RoutePoint(p.x, p.y, p.yaw_rad, Command.TURN_LEFT if i >= 20 else Command.FOLLOW_LANE)
        for i, p in enumerate(points)
    ]
    tracker = RouteTracker(points, lookahead_m=8.0)

    # At x=14 the turn begins at x=20 — 6 m ahead, inside the 8 m lookahead.
    assert tracker.update(14.0, 0.0, 0.0)["command"] == Command.TURN_LEFT
    # At x=5 it is 15 m away, still beyond it.
    assert RouteTracker(points, lookahead_m=8.0).update(5.0, 0.0, 0.0)["command"] == (
        Command.FOLLOW_LANE
    )


def test_a_straight_route_has_no_curvature():
    tracker = RouteTracker(straight_route())
    assert tracker.update(10.0, 0.0, 0.0)["lookahead_curvature"] == pytest.approx(0.0, abs=1e-9)


def test_curvature_is_signed_by_turn_direction():
    """Positive left, matching the lateral-error convention. Two encodings of
    'left' that disagree would make a controller steer into the corner."""
    radius = 30.0
    left = [
        RoutePoint(radius * math.sin(i / 50), radius * (1 - math.cos(i / 50)), 0.0)
        for i in range(100)
    ]
    right = [RoutePoint(p.x, -p.y, 0.0) for p in left]

    left_curvature = RouteTracker(left).update(left[10].x, left[10].y, 0.0)["lookahead_curvature"]
    right_curvature = RouteTracker(right).update(
        right[10].x, right[10].y, 0.0
    )["lookahead_curvature"]

    assert left_curvature > 0
    assert right_curvature < 0
    assert abs(left_curvature) == pytest.approx(1 / radius, rel=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# RoadOption mapping
# ─────────────────────────────────────────────────────────────────────────────


class FakeRoadOption:
    def __init__(self, name):
        self.name = name


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("LANEFOLLOW", Command.FOLLOW_LANE),
        ("LEFT", Command.TURN_LEFT),
        ("RIGHT", Command.TURN_RIGHT),
        ("STRAIGHT", Command.GO_STRAIGHT),
        ("VOID", Command.FOLLOW_LANE),
    ],
)
def test_road_options_map_to_planner_branches(option, expected):
    assert road_option_to_command(FakeRoadOption(option)) == expected


def test_lane_changes_are_not_turns():
    """A lane change has a different steering profile from a junction turn.
    Feeding it to the turn branches teaches them the wrong behaviour."""
    assert road_option_to_command(FakeRoadOption("CHANGELANELEFT")) == Command.FOLLOW_LANE
    assert road_option_to_command(FakeRoadOption("CHANGELANERIGHT")) == Command.FOLLOW_LANE


def test_an_unknown_road_option_degrades_rather_than_raising():
    """A new enum member in a future CARLA must not abort a 1000-episode run."""
    assert road_option_to_command(FakeRoadOption("SOMETHING_NEW")) == Command.FOLLOW_LANE
