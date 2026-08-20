"""
Route representation and progress tracking, independent of any simulator.

Why this is a separate module
-----------------------------
The geometry here — projecting the ego onto a polyline, measuring cross-track
and heading error, deciding when a route is complete or abandoned — is where
route-following bugs actually live, and it needs no simulator to be wrong. Kept
here it is unit-testable on any machine, including one that cannot run CARLA at
all. The CARLA layer then only has to convert waypoints into
:class:`RoutePoint` and read the answers back.

It also gives the kinematic and CARLA backends one definition of "route
completion" rather than two that drift, which matters because that number is
half of the CARLA Leaderboard driving score.

Progress is monotonic by construction
-------------------------------------
Projection searches forward from the last matched index rather than over the
whole route. A nearest-point search over all waypoints looks simpler and is
wrong on any route that crosses or doubles back on itself: the ego passes near
an earlier segment, the "nearest" point jumps backwards, and route completion
decreases while the car is driving forwards. Searching a forward window makes
progress monotonic and costs O(window) instead of O(route).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from pathfinder.sim.base import Command

__all__ = [
    "ROUTE_DEVIATION_M",
    "RoutePoint",
    "RouteTracker",
    "road_option_to_command",
]

#: Cross-track distance beyond which the ego is judged to have left the route.
#: The CARLA Leaderboard terminates a route on deviation rather than penalising
#: it, because a car 30 m off course is not driving the route being scored.
ROUTE_DEVIATION_M = 30.0

#: CARLA ``RoadOption`` names mapped onto the policy's four branches. Held as
#: strings so this module never imports CARLA.
_ROAD_OPTION_TO_COMMAND = {
    "LANEFOLLOW": Command.FOLLOW_LANE,
    "LEFT": Command.TURN_LEFT,
    "RIGHT": Command.TURN_RIGHT,
    "STRAIGHT": Command.GO_STRAIGHT,
    # A lane change is not a junction manoeuvre. Mapping it to TURN_* would feed
    # the policy's turn branches examples of lane changes, which is a different
    # behaviour with a different steering profile.
    "CHANGELANELEFT": Command.FOLLOW_LANE,
    "CHANGELANERIGHT": Command.FOLLOW_LANE,
    "VOID": Command.FOLLOW_LANE,
}


def road_option_to_command(road_option: object) -> Command:
    """Map a CARLA ``RoadOption`` (or its name) onto a policy branch.

    Unknown options fall back to ``FOLLOW_LANE`` rather than raising: a new
    enum member in a future CARLA should degrade to lane following, not abort a
    thousand-episode benchmark.
    """
    name = getattr(road_option, "name", None) or str(road_option)
    return _ROAD_OPTION_TO_COMMAND.get(name.upper(), Command.FOLLOW_LANE)


@dataclass(frozen=True)
class RoutePoint:
    """One waypoint on the planned route."""

    x: float
    y: float
    #: Lane heading at this point, radians.
    yaw_rad: float
    command: Command = Command.FOLLOW_LANE


@dataclass
class RouteTracker:
    """Tracks progress along a polyline route and reports control errors.

    Args:
        points: Ordered route waypoints, roughly evenly spaced.
        lookahead_m: Distance ahead used for curvature and the active command.
            The command must lead the vehicle — a policy told "turn left" only
            once it is already in the junction has no time to act.
        search_window: Waypoints ahead of the current index considered when
            re-projecting. Must exceed the distance covered in one tick.

    Raises:
        ValueError: If fewer than two points are given, since a single point
            defines no direction and no length.
    """

    points: list[RoutePoint]
    lookahead_m: float = 8.0
    search_window: int = 60

    index: int = field(default=0, init=False)
    _cumulative: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError(f"a route needs at least 2 points, got {len(self.points)}")

        self._cumulative = [0.0]
        for previous, current in zip(self.points, self.points[1:], strict=False):
            self._cumulative.append(
                self._cumulative[-1] + math.hypot(current.x - previous.x, current.y - previous.y)
            )

    @property
    def total_length_m(self) -> float:
        return self._cumulative[-1]

    @property
    def distance_along_m(self) -> float:
        """Arc length covered, taken from the current index."""
        return self._cumulative[self.index]

    @property
    def completion(self) -> float:
        """Fraction of the route travelled, clamped to [0, 1]."""
        if self.total_length_m <= 0:
            return 0.0
        return min(1.0, max(0.0, self.distance_along_m / self.total_length_m))

    def update(self, x: float, y: float, yaw_rad: float) -> dict:
        """Project the ego onto the route and report errors from that point.

        Returns a dict with ``lateral_error_m`` (positive left),
        ``heading_error_rad`` wrapped to [-pi, pi], ``lookahead_curvature``,
        ``command`` for the lookahead point, ``completion``, and ``deviated``.
        """
        self.index = self._advance(x, y)

        lateral = self._signed_lateral_error(x, y)
        reference = self.points[min(self.index, len(self.points) - 1)]
        heading_error = _wrap_angle(yaw_rad - reference.yaw_rad)

        lookahead_index = self._index_ahead(self.lookahead_m)
        return {
            "lateral_error_m": lateral,
            "heading_error_rad": heading_error,
            "lookahead_curvature": self._curvature_at(lookahead_index),
            "command": self.points[lookahead_index].command,
            "completion": self.completion,
            "deviated": abs(lateral) > ROUTE_DEVIATION_M,
        }

    def _advance(self, x: float, y: float) -> int:
        """Nearest index within a forward window, so progress cannot go backwards."""
        best_index = self.index
        best_distance = float("inf")
        last = min(self.index + self.search_window, len(self.points) - 1)

        for candidate in range(self.index, last + 1):
            point = self.points[candidate]
            distance = (point.x - x) ** 2 + (point.y - y) ** 2
            if distance < best_distance:
                best_distance = distance
                best_index = candidate
        return best_index

    def _signed_lateral_error(self, x: float, y: float) -> float:
        """Perpendicular offset from the segment at the current index.

        Positive is left of the direction of travel. The sign is what lets a
        controller know which way to steer; an unsigned distance would leave it
        guessing and oscillating.
        """
        first = self.points[self.index]
        second = self.points[min(self.index + 1, len(self.points) - 1)]

        segment_x, segment_y = second.x - first.x, second.y - first.y
        length = math.hypot(segment_x, segment_y)
        if length < 1e-9:
            # Degenerate segment: fall back to the stored lane heading.
            segment_x, segment_y = math.cos(first.yaw_rad), math.sin(first.yaw_rad)
            length = 1.0

        # 2-D cross product of the segment direction with the ego offset.
        return ((second.x - first.x) * (y - first.y) - (second.y - first.y) * (x - first.x)) / length

    def _index_ahead(self, distance_m: float) -> int:
        target = self._cumulative[self.index] + distance_m
        index = self.index
        while index + 1 < len(self.points) and self._cumulative[index + 1] < target:
            index += 1
        return index

    def _curvature_at(self, index: int) -> float:
        """Signed curvature from the heading change across neighbouring points.

        Curvature is dtheta/ds, and the subtlety is which ``ds``. The heading
        change between two consecutive chords accumulates over the distance
        between those chords' midpoints — the *mean* segment length, not the
        span from ``index - 1`` to ``index + 1``. Dividing by the full span
        reports exactly half the true curvature, which is the kind of error that
        never crashes anything and quietly halves every steering command that
        depends on it.
        """
        if index <= 0 or index >= len(self.points) - 1:
            return 0.0
        previous, current, following = (
            self.points[index - 1], self.points[index], self.points[index + 1]
        )
        heading_in = math.atan2(current.y - previous.y, current.x - previous.x)
        heading_out = math.atan2(following.y - current.y, following.x - current.x)

        incoming = math.hypot(current.x - previous.x, current.y - previous.y)
        outgoing = math.hypot(following.x - current.x, following.y - current.y)
        arc = (incoming + outgoing) / 2.0
        if arc < 1e-6:
            return 0.0
        return _wrap_angle(heading_out - heading_in) / arc


def _wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
