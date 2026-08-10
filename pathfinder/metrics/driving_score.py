
"""
CARLA Leaderboard driving score.

The metric is not arbitrary and reproducing it exactly is the point — a
"driving score" computed some other way is not comparable to any published
number, which makes it useless for the only thing a benchmark is for.

    driving_score = route_completion x infraction_penalty

``route_completion`` is the fraction of the route distance covered, in [0, 1].

``infraction_penalty`` starts at 1.0 and is **multiplied** by a coefficient for
every infraction:

    ==========================  =====
    infraction                  coeff
    ==========================  =====
    collision with pedestrian   0.50
    collision with vehicle      0.60
    collision with static       0.65
    red light violation         0.70
    stop sign violation         0.80
    ==========================  =====

Multiplicative rather than additive is deliberate in the original metric and
worth preserving: two collisions are much worse than twice one collision, and a
subtractive penalty would let a long route "earn back" the cost of hitting a
pedestrian. It also keeps the penalty in (0, 1] so the score can never go
negative.

Two infractions do **not** carry a coefficient because they terminate the
episode instead, which caps route completion at wherever the agent stopped:
off-road driving and route deviation. ``agent_blocked`` likewise ends the run.
Penalising them multiplicatively *and* truncating the route would double-count.

Reference: CARLA Autonomous Driving Leaderboard, evaluation metrics.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from pathfinder.sim.base import Infraction

__all__ = [
    "INFRACTION_PENALTIES",
    "EpisodeScore",
    "aggregate",
    "score_episode",
]

#: Multiplicative penalty per infraction occurrence.
INFRACTION_PENALTIES: dict[Infraction, float] = {
    Infraction.COLLISION_PEDESTRIAN: 0.50,
    Infraction.COLLISION_VEHICLE: 0.60,
    Infraction.COLLISION_STATIC: 0.65,
    Infraction.RED_LIGHT: 0.70,
    Infraction.STOP_SIGN: 0.80,
}

#: Infractions that end the episode. They are not penalised multiplicatively —
#: truncating route completion is already the penalty, and applying both would
#: count the same failure twice.
TERMINAL_INFRACTIONS = frozenset(
    {Infraction.OFF_ROAD, Infraction.ROUTE_DEVIATION, Infraction.AGENT_BLOCKED}
)

#: Lane invasion is tracked and reported but carries no coefficient in the
#: leaderboard metric. Reporting it separately keeps our numbers comparable to
#: published ones while still surfacing a real quality signal.
UNSCORED_INFRACTIONS = frozenset({Infraction.LANE_INVASION})


@dataclass
class EpisodeScore:
    """Scored outcome of one episode."""

    episode_id: str
    route_completion: float
    infraction_penalty: float
    driving_score: float
    infractions: dict[str, int] = field(default_factory=dict)
    distance_travelled_m: float = 0.0
    route_length_m: float = 0.0
    frames: int = 0
    duration_seconds: float = 0.0
    status: str = "completed"
    termination_reason: str = ""

    @property
    def collisions(self) -> int:
        return sum(
            count
            for name, count in self.infractions.items()
            if name.startswith("collision_")
        )

    @property
    def mean_fps(self) -> float:
        return self.frames / self.duration_seconds if self.duration_seconds > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "route_completion": round(self.route_completion, 4),
            "infraction_penalty": round(self.infraction_penalty, 4),
            "driving_score": round(self.driving_score, 2),
            "collisions": self.collisions,
            "lane_invasions": self.infractions.get(Infraction.LANE_INVASION.value, 0),
            "red_lights_run": self.infractions.get(Infraction.RED_LIGHT.value, 0),
            "infractions": dict(self.infractions),
            "distance_travelled_m": round(self.distance_travelled_m, 1),
            "route_length_m": round(self.route_length_m, 1),
            "frames": self.frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "mean_fps": round(self.mean_fps, 1),
            "status": self.status,
            "termination_reason": self.termination_reason,
        }


def score_episode(
    *,
    episode_id: str,
    distance_travelled_m: float,
    route_length_m: float,
    infractions: list[Infraction],
    frames: int = 0,
    duration_seconds: float = 0.0,
    status: str = "completed",
    termination_reason: str = "",
) -> EpisodeScore:
    """Compute the leaderboard driving score for one episode.

    Raises:
        ValueError: If ``route_length_m`` is not positive — completion would be
            undefined, and silently returning 0 or 1 would corrupt an aggregate.
    """
    if route_length_m <= 0:
        raise ValueError(f"route_length_m must be positive, got {route_length_m}")

    # Clamped at 1.0: overshooting the route end is not extra credit.
    route_completion = min(1.0, max(0.0, distance_travelled_m / route_length_m))

    counts = Counter(infraction.value for infraction in infractions)
    penalty = 1.0
    for infraction, coefficient in INFRACTION_PENALTIES.items():
        occurrences = counts.get(infraction.value, 0)
        if occurrences:
            penalty *= coefficient**occurrences

    return EpisodeScore(
        episode_id=episode_id,
        route_completion=route_completion,
        infraction_penalty=penalty,
        # x100 to match the leaderboard's 0-100 presentation.
        driving_score=route_completion * penalty * 100.0,
        infractions=dict(counts),
        distance_travelled_m=distance_travelled_m,
        route_length_m=route_length_m,
        frames=frames,
        duration_seconds=duration_seconds,
        status=status,
        termination_reason=termination_reason,
    )


@dataclass
class BenchmarkSummary:
    """Aggregate over a set of episodes."""

    episodes: int
    driving_score: float
    route_completion: float
    infraction_penalty: float
    collisions_per_km: float
    total_distance_km: float
    mean_fps: float
    failures: int
    infraction_totals: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "driving_score": round(self.driving_score, 2),
            "route_completion": round(self.route_completion, 4),
            "infraction_penalty": round(self.infraction_penalty, 4),
            "collisions_per_km": round(self.collisions_per_km, 3),
            "total_distance_km": round(self.total_distance_km, 2),
            "mean_fps": round(self.mean_fps, 1),
            "failures": self.failures,
            "infraction_totals": dict(self.infraction_totals),
        }


def aggregate(scores: list[EpisodeScore]) -> BenchmarkSummary:
    """Aggregate episode scores the way the leaderboard does.

    Driving score is the **arithmetic mean over episodes**, not a distance
    weighted average. That is the leaderboard's definition, and it matters: a
    weighted mean would let long easy routes mask failures on short hard ones,
    which is exactly the behaviour the metric is designed to punish.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if not scores:
        raise ValueError("cannot aggregate an empty score list")

    totals: Counter[str] = Counter()
    for score in scores:
        totals.update(score.infractions)

    total_distance_km = sum(score.distance_travelled_m for score in scores) / 1000.0
    total_collisions = sum(score.collisions for score in scores)
    total_frames = sum(score.frames for score in scores)
    total_seconds = sum(score.duration_seconds for score in scores)

    return BenchmarkSummary(
        episodes=len(scores),
        driving_score=sum(s.driving_score for s in scores) / len(scores),
        route_completion=sum(s.route_completion for s in scores) / len(scores),
        infraction_penalty=sum(s.infraction_penalty for s in scores) / len(scores),
        collisions_per_km=total_collisions / total_distance_km if total_distance_km else 0.0,
        total_distance_km=total_distance_km,
        mean_fps=total_frames / total_seconds if total_seconds else 0.0,
        failures=sum(1 for s in scores if s.status != "completed"),
        infraction_totals=dict(totals),
    )
