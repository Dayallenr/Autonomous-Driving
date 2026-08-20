"""
Sustained contact must count as one collision, not one per tick.

CARLA's collision sensor re-fires every tick for as long as two actors are
touching. Counted raw, a car resting against another vehicle registers hundreds
of "collisions" for a single event, and because the driving score penalises
collisions *multiplicatively* (0.60 per vehicle collision) the episode score
underflows to exactly 0.0:

    0.60 ** 475 == 0.0

That is not a hypothetical. The first CARLA perception ablation
(results/ablation/carla_report.json, 2026-08-20) recorded 475 vehicle
collisions in one 1,500-step episode and 266 static collisions in another, so
9 of 10 candidate episodes and 1 baseline episode scored exactly zero and
"collisions per km" reported sensor callbacks rather than collisions. Both arms
were affected, so even the *difference* between them was not meaningful.

The fix mirrors the CARLA Leaderboard's ``CollisionTest``, which remembers the
last collision per actor for ``MAX_ID_TIME`` seconds. These tests pin that rule
as pure logic, runnable without a CARLA server.
"""
from __future__ import annotations

from pathfinder.sim.carla_backend import COLLISION_COOLDOWN_S, CarlaSimulator


def tracker() -> CarlaSimulator:
    """A simulator instance with only the collision registry initialised.

    ``__new__`` avoids the constructor's server connection; ``_register_collision``
    is deliberately pure enough to need nothing else.
    """
    simulator = CarlaSimulator.__new__(CarlaSimulator)
    simulator._collision_seen = {}
    return simulator


def test_first_contact_with_an_actor_counts():
    assert tracker()._register_collision("vehicle-1", 0.0)


def test_sustained_contact_counts_once():
    """The regression: 0.05 s ticks against one actor for 3 s is one collision."""
    simulator = tracker()
    counted = sum(
        1 for tick in range(60) if simulator._register_collision("vehicle-1", tick * 0.05)
    )
    assert counted == 1


def test_contact_after_the_cooldown_counts_again():
    """A genuine second impact must still register, or the metric under-reports."""
    simulator = tracker()
    assert simulator._register_collision("vehicle-1", 0.0)
    assert not simulator._register_collision("vehicle-1", COLLISION_COOLDOWN_S - 0.01)
    assert simulator._register_collision("vehicle-1", COLLISION_COOLDOWN_S)


def test_different_actors_are_tracked_independently():
    """Hitting two cars at once is two collisions; one cooldown must not mask
    the other."""
    simulator = tracker()
    assert simulator._register_collision("vehicle-1", 0.0)
    assert simulator._register_collision("vehicle-2", 0.0)
    assert not simulator._register_collision("vehicle-1", 0.1)
    assert not simulator._register_collision("vehicle-2", 0.1)


def test_a_pileup_does_not_underflow_the_multiplicative_score():
    """The property that actually matters downstream.

    Ten seconds of continuous contact with one vehicle must leave the episode
    scoreable rather than collapsing 0.60**n to zero.
    """
    simulator = tracker()
    collisions = sum(
        1 for tick in range(200) if simulator._register_collision("vehicle-1", tick * 0.05)
    )
    assert collisions == 2  # one at t=0, one after the 5 s cooldown
    assert 0.60**collisions > 0.3
