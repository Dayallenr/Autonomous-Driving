"""
The perception seam: what the system perceived, as its own type.

The driving loop used to hand the Policy the simulator's ground truth
directly, so every driving score described a Policy with perfect perception.
This module is the seam that breaks that: a :class:`Perception` maps a frame
to a :class:`PerceivedScene`, and a Policy composed over it consumes the
scene instead of the privileged fields. Swapping what sits behind the seam —
ground truth or a Detector — is then a configuration change, which is what
makes a ground-truth-versus-Detector ablation a measurement rather than two
unrelated pipelines.

``PerceivedScene`` is deliberately a distinct type from
:class:`~pathfinder.sim.base.FrameState`. One is what the simulator knows,
the other is what the system perceived, and keeping them apart in the type
system is what stops the two from being conflated by accident.

What stays privileged, and why
------------------------------
Only the obstacle fields pass through this seam. Everything else the Policy
reads remains simulator ground truth, deliberately:

* **Localization** — cross-track error, heading error, lookahead curvature.
  These are localization-against-route quantities, not perception, and real
  modular stacks do localize against a map.
* **Traffic-light state and distance** — the trained Detector's eight KITTI
  classes do not include a traffic light, so it cannot see one. Perceiving
  lights needs a second model that does not exist yet.

Every write-up of a score produced through this seam must therefore say
"perceived obstacles, privileged localization and traffic lights", never
"full perception".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from pathfinder.sim.base import FrameState

__all__ = ["PerceivedScene", "Perception", "perception_name"]


def perception_name(perception: Perception) -> str:
    """The provenance identifier a perception stamps onto frames.

    One derivation, shared by ``ModularPolicy`` (which writes it into
    telemetry) and the ablation runner (which pre-checks it before spending
    hours of simulation) — two copies of this expression could drift, and the
    drift would surface as a mid-run RuntimeError instead of a startup error.
    A Perception that declares no ``NAME`` is still identifiable by class.
    """
    return getattr(perception, "NAME", type(perception).__name__)


@dataclass(frozen=True)
class PerceivedScene:
    """What perception reported for one frame.

    The defaults encode the empty scene: nothing seen reads as an infinite
    range and zero detections, which the speed governor already treats as an
    unobstructed road — never as a crash or a silently wrong range.
    """

    #: Range to the nearest perceived obstacle in metres; inf when nothing was
    #: seen. Same semantics as the privileged field it stands in for, which is
    #: what allows a ground-truth implementation to be behaviour-preserving.
    nearest_object_m: float = math.inf
    #: How many objects perception reported this frame. Carried so that a
    #: perception failure can be told apart from a control failure.
    detections: int = 0


class Perception(Protocol):
    """What the driving loop needs from perception. Structural, matching the
    Policy protocol's approach, so a test stub satisfies it without importing
    or inheriting anything."""

    def perceive(self, state: FrameState) -> PerceivedScene: ...
