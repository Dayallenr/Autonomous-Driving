"""
Privileged perception: the simulator's ground truth, forwarded unchanged.

This is the perfect-perception upper bound of the ablation, and the proof that
the seam itself costs nothing: an Episode driven through it must be
indistinguishable from one that read the simulator directly (pinned by the
characterisation test in ``tests/test_perception_seam.py``).
"""
from __future__ import annotations

from pathfinder.perception.base import PerceivedScene
from pathfinder.sim.base import FrameState

__all__ = ["PrivilegedPerception"]


class PrivilegedPerception:
    """Forwards the frame's ground-truth obstacle fields, unchanged."""

    #: Provenance identifier, carried into telemetry on every frame this
    #: implementation produces. Matches ``ControlOutput``'s default: a Policy
    #: with no perception seam reads the same ground truth this forwards.
    NAME = "privileged"

    def perceive(self, state: FrameState) -> PerceivedScene:
        return PerceivedScene(
            nearest_object_m=state.nearest_object_m,
            detections=state.detections,
        )
