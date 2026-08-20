"""Shared test builders. Not a fixtures file: these are plain functions the
perception tests import, so each test file stays readable on its own."""
from __future__ import annotations

from pathfinder.sim.base import Command, FrameState


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
