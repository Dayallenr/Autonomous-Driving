"""
The perception seam: the Policy consumes what perception produced, not what
the simulator knows.

Everything here runs on the kinematic backend, on a machine without CARLA.
The central test is the characterisation test: a seeded Episode driven by the
modular Policy with privileged perception must be indistinguishable from the
pre-seam direct-reading path — same score, same infractions, same frame count.
That is what proves the refactor moved code without changing results.
"""
from __future__ import annotations

import math
import time

import pytest

from pathfinder.perception import PerceivedScene, PrivilegedPerception
from pathfinder.policies import ModularPolicy
from pathfinder.runner import ControlOutput, PurePursuitPolicy, run_episode
from pathfinder.sim import EpisodeSpec, KinematicSimulator
from pathfinder.sim.base import FrameState
from tests.support import make_frame as _frame


class _StubPerception:
    """Satisfies the Perception protocol structurally: defined here, without
    inheriting from or even importing the protocol."""

    def __init__(self, scene: PerceivedScene) -> None:
        self._scene = scene

    def perceive(self, state: FrameState) -> PerceivedScene:
        return self._scene


def test_privileged_perception_forwards_obstacle_fields_unchanged():
    scene = PrivilegedPerception().perceive(_frame(nearest_object_m=7.3, detections=2))
    assert scene == PerceivedScene(nearest_object_m=7.3, detections=2)


def test_an_empty_road_reads_as_nothing_seen():
    scene = PrivilegedPerception().perceive(
        _frame(nearest_object_m=math.inf, detections=0)
    )
    assert math.isinf(scene.nearest_object_m)
    assert scene.detections == 0


def test_perceived_scene_is_a_distinct_type_from_frame_state():
    """"What the simulator knows" and "what the system perceived" must not be
    conflatable by accident: a scene carries no pose, no localization, and no
    traffic-light state."""
    scene = PerceivedScene()
    assert not isinstance(scene, FrameState)
    for privileged_field in ("x", "y", "lateral_error_m", "traffic_light_state"):
        assert not hasattr(scene, privileged_field)


def test_modular_policy_drives_on_the_perceived_scene():
    """The seam is load-bearing: a perception reporting an obstacle on the
    bumper must brake the car even when the simulator says the road is clear."""
    clear_road = _frame(nearest_object_m=math.inf, detections=0)

    sees_wall = ModularPolicy(
        _StubPerception(PerceivedScene(nearest_object_m=1.0, detections=1)),
        PurePursuitPolicy(),
    )
    sees_nothing = ModularPolicy(
        _StubPerception(PerceivedScene()),
        PurePursuitPolicy(),
    )

    braking = sees_wall.plan(clear_road)
    cruising = sees_nothing.plan(clear_road)

    assert braking.throttle == 0.0
    assert braking.brake > cruising.brake
    assert cruising.throttle > 0.0


@pytest.mark.parametrize(
    ("spec_kwargs", "expect_infractions"),
    [
        ({"route_length_m": 200, "max_steps": 400, "seed": 0}, False),
        ({"route_length_m": 200, "max_steps": 400, "seed": 99}, False),
        # Dense enough to produce a red_light infraction, so the infraction
        # comparison below is not trivially empty-versus-empty.
        (
            {
                "route_length_m": 500,
                "max_steps": 1500,
                "seed": 4,
                "traffic_density": 0.6,
                "pedestrian_density": 0.3,
            },
            True,
        ),
    ],
)
def test_privileged_modular_policy_preserves_behaviour_exactly(
    spec_kwargs, expect_infractions
):
    """The characterisation test. Same seed, same spec: the composed Policy
    with privileged perception must reproduce the direct-reading path's result
    field for field, or the seam changed behaviour."""
    spec = EpisodeSpec(episode_id="characterisation", **spec_kwargs)

    with KinematicSimulator() as simulator:
        direct = run_episode(simulator, spec, PurePursuitPolicy())
    with KinematicSimulator() as simulator:
        modular = run_episode(
            simulator, spec, ModularPolicy(PrivilegedPerception(), PurePursuitPolicy())
        )

    assert modular.driving_score == direct.driving_score
    assert modular.route_completion == direct.route_completion
    assert modular.infractions == direct.infractions
    assert modular.frames == direct.frames
    assert modular.termination_reason == direct.termination_reason
    # The dense case exists to make the infraction comparison mean something.
    if expect_infractions:
        assert direct.infractions


def test_modular_policy_stamps_perception_provenance_and_latency():
    """Provenance comes from the component that actually ran, so a frame can
    never be attributed to a perception source that did not produce it. The
    perception cost is measured, and kept out of the controller's own latency."""

    class _SlowPerception:
        NAME = "slow_stub"

        def perceive(self, state: FrameState) -> PerceivedScene:
            time.sleep(0.005)
            return PerceivedScene()

    class _FixedLatencyController:
        def plan(self, state: FrameState) -> ControlOutput:
            return ControlOutput(throttle=0.1, steer=0.0, brake=0.0, latency_ms=1.25)

    control = ModularPolicy(_SlowPerception(), _FixedLatencyController()).plan(_frame())

    assert control.perception == "slow_stub"
    assert control.perception_latency_ms >= 5.0
    # Control latency is the controller's own figure, untouched by perception.
    assert control.latency_ms == 1.25


def test_perception_without_a_name_is_attributed_by_class():
    """Provenance is never empty: a Perception that declares no NAME is still
    identifiable in telemetry by what it is."""
    control = ModularPolicy(_StubPerception(PerceivedScene()), PurePursuitPolicy()).plan(
        _frame()
    )
    assert control.perception == "_StubPerception"


def test_privileged_perception_is_named_privileged():
    control = ModularPolicy(PrivilegedPerception(), PurePursuitPolicy()).plan(_frame())
    assert control.perception == "privileged"
    # The default must equal PrivilegedPerception's name: a Policy with no
    # perception seam reads the same ground truth PrivilegedPerception
    # forwards, and the two literals live in different modules.
    assert ControlOutput(throttle=0, steer=0, brake=0).perception == PrivilegedPerception.NAME


def test_every_telemetry_frame_carries_perception_provenance_and_latency():
    """The acceptance test for the telemetry half: an ablation is two numbers
    with no meaning unless every frame says which perception produced it and
    what that perception cost — distinct from what the controller cost."""

    class _StampedPolicy:
        """Reports a fixed, recognisable perception cost, so the test can tell
        a faithfully forwarded measurement apart from anything re-derived by
        the runner (the old code recorded simulator step time here)."""

        def plan(self, state: FrameState) -> ControlOutput:
            return ControlOutput(
                throttle=0.2, steer=0.0, brake=0.0,
                latency_ms=0.5, perception="stamped_stub", perception_latency_ms=7.75,
            )

    rows: list[dict] = []
    with KinematicSimulator() as simulator:
        run_episode(
            simulator,
            EpisodeSpec(episode_id="prov", route_length_m=60.0, max_steps=30),
            _StampedPolicy(),
            telemetry_sink=rows.append,
        )

    assert rows
    assert {row["perception"] for row in rows} == {"stamped_stub"}
    assert {row["perception_latency_ms"] for row in rows} == {7.75}
    assert {row["policy_latency_ms"] for row in rows} == {0.5}


def test_modular_episode_telemetry_is_attributed_to_its_perception():
    """End to end through the real seam: privileged perception, real runner,
    real sink rows."""
    rows: list[dict] = []
    with KinematicSimulator() as simulator:
        run_episode(
            simulator,
            EpisodeSpec(episode_id="prov-priv", route_length_m=60.0, max_steps=30),
            ModularPolicy(PrivilegedPerception(), PurePursuitPolicy()),
            telemetry_sink=rows.append,
        )

    assert rows
    assert {row["perception"] for row in rows} == {"privileged"}
    for row in rows:
        assert row["perception_latency_ms"] >= 0.0
