"""Shared test builders. Not a fixtures file: these are plain functions the
perception and comparison tests import, so each test file stays readable on
its own."""
from __future__ import annotations

from pathfinder.comparison import (
    ROLE_FLOOR,
    ROLE_REFERENCE,
    ROLE_STUDENT,
    ComparisonArm,
)
from pathfinder.policies import PURE_PURSUIT, CarlaBehaviorAgentPolicy, CILStudentPolicy
from pathfinder.runner import ControlOutput, PurePursuitPolicy
from pathfinder.sim.base import Command, FrameState


class StubReferencePolicy:
    """Drives blind but declares the behaviour agent's registry name, letting
    the reference column run on a machine with no CARLA — the same stance as
    ``test_policies``: the agent is CARLA's, so what is testable without a
    server is everything about how the suite treats it."""

    NAME = CarlaBehaviorAgentPolicy.NAME

    def plan(self, state) -> ControlOutput:
        return ControlOutput(throttle=0.3, steer=0.0, brake=0.0, latency_ms=0.1)


def make_comparison_arms(student, *, reference_policy=None, reference_skip=""):
    """The three columns as the CLI would build them, with the reference
    stubbed. ``reference_skip`` makes the reference a skipped arm instead."""
    name = CarlaBehaviorAgentPolicy.NAME
    if reference_skip:
        reference = ComparisonArm(
            role=ROLE_REFERENCE, policy_name=name, model_version=name,
            policy=None, skip_reason=reference_skip,
        )
    else:
        reference = ComparisonArm(
            role=ROLE_REFERENCE, policy_name=name, model_version=name,
            policy=reference_policy or StubReferencePolicy(),
        )
    return [
        ComparisonArm(
            role=ROLE_FLOOR,
            policy_name=PURE_PURSUIT,
            model_version=PURE_PURSUIT,
            policy=PurePursuitPolicy(),
        ),
        ComparisonArm(
            role=ROLE_STUDENT,
            policy_name=CILStudentPolicy.NAME,
            model_version=student.model_version,
            policy=student,
            weights=str(student.weights_path),
        ),
        reference,
    ]


def make_frame(**overrides) -> FrameState:
    """A plausible mid-Episode frame; override only what the test is about."""
    base = dict(
        frame_index=5,
        simulation_time=0.25,
        x=1.0,
        y=2.0,
        yaw_degrees=3.0,
        speed_mps=6.0,
        command=Command.FOLLOW_LANE,
        distance_travelled_m=12.0,
        nearest_object_m=7.3,
        detections=2,
    )
    base.update(overrides)
    return FrameState(**base)
