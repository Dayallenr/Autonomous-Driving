"""
Policy selection, and CARLA's own behaviour agent as a labelled baseline.

Why a reference Policy
----------------------
A disappointing driving score has two very different causes — a weak Policy or
weak perception — and without a reference they are indistinguishable. CARLA
ships a competent behaviour agent that reads the simulator's ground truth
directly. Scoring an Episode with it gives an upper bound produced by the same
route, traffic, weather, and scoring code as everything else, so the difference
between it and a project Policy is attributable.

It is **not project work**, and the code refuses to let that get lost. It is
registered as ``carla_builtin_behavior_agent``: the name travels into
``model_version`` and from there into every result that records one, so nothing
downstream has to remember to add a caveat.

Why an adapter rather than a new seam
-------------------------------------
CARLA's ``BehaviorAgent`` drives an actor and returns a ``carla.VehicleControl``;
:class:`~pathfinder.runner.Policy` takes a :class:`FrameState` and returns a
:class:`ControlOutput`. The two do not line up, but the mismatch is entirely
about *where the state comes from* — the agent reads the world itself instead of
being told about it. So the adapter is constructed with the simulator and takes
the ego actor and route destination from it. The runner, the ``Policy``
protocol, and :class:`SimulatorBackend` are all untouched.

CARLA imports are deferred until the first :meth:`plan` call, so importing this
module — and therefore selecting any other Policy — works on a machine with no
CARLA installed.

The interface used to be called ``Planner``; ADR-0002 records why it is not.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from pathfinder.perception.base import Perception, perception_name
from pathfinder.runner import ControlOutput, Policy, PurePursuitPolicy
from pathfinder.sim.base import FrameState
from pathfinder.sim.carla_paths import ensure_agents_importable

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_POLICY",
    "POLICY_NAMES",
    "PURE_PURSUIT",
    "CILStudentPolicy",
    "CarlaBehaviorAgentPolicy",
    "ModularPolicy",
    "build_policy",
    "result_label",
]

#: Registry name of the project's geometric controller. Taken from the class
#: itself so telemetry stamped from an instance and results labelled from the
#: registry can never use two names for one controller.
PURE_PURSUIT = PurePursuitPolicy.NAME

#: What every entry point selects when nothing is asked for.
DEFAULT_POLICY = PURE_PURSUIT

#: Sentinel for "this backend has no such attribute at all", which is a
#: different mistake from "it has one and it is empty".
_MISSING = object()


def _behavior_agent_class():
    """Import CARLA's ``BehaviorAgent``.

    Separated out so it can be replaced in tests: everything about this adapter
    except the agent itself is testable on a machine that cannot run CARLA, and
    that is most of what can go wrong.
    """
    ensure_agents_importable(raise_on_missing=True)
    from agents.navigation.behavior_agent import BehaviorAgent

    return BehaviorAgent


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class CarlaBehaviorAgentPolicy:
    """CARLA's built-in behaviour agent, adapted to the ``Policy`` protocol.

    Args:
        simulator: The CARLA backend the Episode runs in. Read for the ego actor
            and the planned route's destination.
        behavior: One of CARLA's presets — ``cautious``, ``normal``, or
            ``aggressive``.

    Raises:
        ValueError: If ``simulator`` cannot supply an ego actor and a route
            destination. Checked here rather than on the first :meth:`plan` so
            that pairing this with the kinematic backend fails at startup: the
            episode runner turns a mid-Episode exception into a scored failure,
            so a lazy check would spend a whole benchmark suite producing
            zero-score Episodes instead of one error.

    The agent is built lazily on the first :meth:`plan` of an Episode, because at
    construction time the ego does not exist yet: the runner builds a Policy
    before calling ``reset``. It is rebuilt whenever the ego changes, which is
    once per Episode — ``reset`` destroys the previous vehicle, and an agent
    still holding that actor would be driving something that no longer exists.
    """

    #: Registry name. Says whose Policy this is; see the module docstring.
    NAME = "carla_builtin_behavior_agent"

    def __init__(self, simulator, *, behavior: str = "normal") -> None:
        if (
            getattr(simulator, "vehicle", _MISSING) is _MISSING
            or getattr(simulator, "route_destination", _MISSING) is _MISSING
        ):
            raise ValueError(
                f"{self.NAME} drives a CARLA actor directly, so it needs the CARLA "
                f"backend; {type(simulator).__name__} exposes no ego vehicle or route "
                "destination"
            )
        self._simulator = simulator
        self._behavior = behavior
        self._agent = None
        self._bound_vehicle_id: int | None = None

    @property
    def behavior(self) -> str:
        """The CARLA behaviour preset, exposed so result artifacts can record
        which one produced their numbers."""
        return self._behavior

    def plan(self, state: FrameState) -> ControlOutput:
        """Ask the agent for a control. ``state`` is ignored — the agent reads
        the world directly, which is exactly what makes it privileged."""
        started = time.perf_counter()
        agent = self._agent_for_current_episode()

        if agent.done():
            # The agent's plan ends at the route destination; the Episode ends
            # on the simulator's terms. Holding the vehicle still across that
            # gap is this adapter's decision rather than a guess about what
            # CARLA's local planner does with an exhausted waypoint queue.
            throttle, steer, brake = 0.0, 0.0, 1.0
        else:
            control = agent.run_step()
            throttle = _clamp(control.throttle, 0.0, 1.0)
            steer = _clamp(control.steer, -1.0, 1.0)
            brake = _clamp(control.brake, 0.0, 1.0)

        return ControlOutput(
            throttle=throttle,
            steer=steer,
            brake=brake,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _agent_for_current_episode(self):
        vehicle = self._simulator.vehicle
        destination = self._simulator.route_destination
        if vehicle is None:
            raise RuntimeError(
                f"{self.NAME} has no ego vehicle to drive — call reset() on the "
                "simulator before planning"
            )
        if destination is None:
            raise RuntimeError(
                f"{self.NAME} has no route destination to drive to — the Episode's "
                "route was never planned"
            )

        if self._agent is None or self._bound_vehicle_id != vehicle.id:
            agent_class = _behavior_agent_class()
            self._agent = agent_class(vehicle, behavior=self._behavior)
            self._agent.set_destination(destination)
            self._bound_vehicle_id = vehicle.id
            logger.info(
                "%s bound to ego actor %s with behaviour %r",
                self.NAME, vehicle.id, self._behavior,
            )
        return self._agent


class ModularPolicy:
    """ADR-0001's modular architecture, realised: perception feeding control.

    Composes a :class:`~pathfinder.perception.base.Perception` with an inner
    controller Policy, and satisfies the ``Policy`` protocol itself — so the
    episode runner drives it exactly as it drives anything else, and the seam
    count stays at one.

    :meth:`plan` perceives the frame, then hands the controller a copy of the
    frame whose obstacle fields are what perception reported rather than what
    the simulator knows. Only those two fields cross the seam; localization
    and traffic-light state pass through privileged, for the reasons recorded
    in ``pathfinder/perception/base.py``.
    """

    def __init__(self, perception: Perception, controller: Policy) -> None:
        self._perception = perception
        self._controller = controller
        # Provenance is read off the component itself rather than passed in, so
        # a frame can never be attributed to a perception that did not run.
        self._perception_name = perception_name(perception)

    def plan(self, state: FrameState) -> ControlOutput:
        perception_started = time.perf_counter()
        scene = self._perception.perceive(state)
        perception_ms = (time.perf_counter() - perception_started) * 1000.0

        perceived = replace(
            state,
            nearest_object_m=scene.nearest_object_m,
            detections=scene.detections,
        )
        control = self._controller.plan(perceived)
        return replace(
            control,
            perception=self._perception_name,
            perception_latency_ms=perception_ms,
        )


class CILStudentPolicy:
    """The trained CIL student behind the ``Policy`` protocol.

    Args:
        weights: Checkpoint to load — either the DAgger CLI's per-iteration
            format (``{"model": state_dict, ...}``) or a bare state dict.
            There is deliberately no default path: a student that silently
            picks up whatever weights lie around produces scores attributable
            to nothing.
        device: Where inference runs. Pinned at construction, mirroring
            :class:`~pathfinder.perception.detector.YoloDetector` — a device
            that changes mid-benchmark changes the numbers.

    The loaded model is put in eval mode and planning runs under inference
    mode (both via :class:`~pathfinder.dagger.CILPolicy`, which this wraps).
    ``model_version`` is derived from the checkpoint's bytes, so every result
    stamped with it is attributable to exactly those weights — two different
    checkpoints can never share a label the way two "cil_student" runs would.

    Raises:
        FileNotFoundError: If the checkpoint is missing. Eager and loud, like
            the Detector: the runner turns mid-Episode exceptions into scored
            failures, so a lazy check would spend a whole benchmark producing
            zero-score Episodes instead of one error.
        ValueError: If the checkpoint does not fit the control-output CIL
            architecture (a waypoints-mode checkpoint, or something else
            entirely).

    torch is imported at construction, not at module import, for the same
    reason CARLA is: the registry — and the orchestration layer that imports
    it — must work in an environment without torch installed.
    """

    #: Registry name. The stem of every ``model_version`` this policy stamps.
    NAME = "cil_student"

    def __init__(self, weights: str | Path, *, device: str = "cpu") -> None:
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(
                f"{self.NAME} weights not found at {path}. Train them with "
                "python -m pathfinder.dagger, or point at an existing checkpoint."
            )
        import torch

        from pathfinder.dagger import CILPolicy
        from pathfinder.planning.cil_model import CILModel

        payload = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = (
            payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        )
        model = CILModel(pretrained=False, output_mode="control")
        try:
            model.load_state_dict(state_dict)
        except (RuntimeError, TypeError, AttributeError) as error:
            raise ValueError(
                f"{path} does not hold weights for the control-output CIL model "
                f"({type(model).__name__} with output_mode='control'); a "
                "waypoints-mode checkpoint cannot drive — the runner consumes "
                f"controls. Original error: {error}"
            ) from error

        self._inner = CILPolicy(model, device=device)
        self.weights_path = path
        #: Carried into every result via the existing model-version label.
        self.model_version = (
            f"{self.NAME}@{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
        )
        logger.info("%s loaded %s onto %s", self.model_version, path, device)

    @property
    def model(self):
        return self._inner.model

    @property
    def device(self) -> str:
        return self._inner.device

    def plan(self, state: FrameState) -> ControlOutput:
        return self._inner.plan(state)


def _build_pure_pursuit(simulator, **kwargs) -> Policy:
    """The geometric controller reads only the ``FrameState`` it is handed, so
    the simulator is of no interest to it."""
    return PurePursuitPolicy(**kwargs)


def _build_cil_student(simulator, *, weights=None, **kwargs) -> Policy:
    """The student reads only the ``FrameState`` (pixels plus command), so the
    simulator is of no interest to it — but weights are non-negotiable."""
    if weights is None:
        raise ValueError(
            f"{CILStudentPolicy.NAME} refuses to run without trained weights; "
            "pass build_policy(name, weights=...) pointing at a DAgger checkpoint"
        )
    return CILStudentPolicy(weights, **kwargs)


def _build_carla_behavior_agent(simulator, **kwargs) -> Policy:
    if simulator is None:
        raise ValueError(
            f"{CarlaBehaviorAgentPolicy.NAME} drives the simulator's own ego actor, "
            "so it needs the simulator it will run in: build_policy(name, simulator=...)"
        )
    return CarlaBehaviorAgentPolicy(simulator, **kwargs)


#: Selectable Policies. ``pure_pursuit`` is the project's geometric controller;
#: ``cil_student`` is the trained CIL model (weights required, no default);
#: ``carla_builtin_behavior_agent`` is CARLA's own behaviour agent, kept as a
#: labelled reference rather than as project work. One table rather than a name
#: list beside a matching if-cascade, so a new Policy is one edit.
_BUILDERS: dict[str, Callable[..., Policy]] = {
    PURE_PURSUIT: _build_pure_pursuit,
    CILStudentPolicy.NAME: _build_cil_student,
    CarlaBehaviorAgentPolicy.NAME: _build_carla_behavior_agent,
}

POLICY_NAMES: tuple[str, ...] = tuple(_BUILDERS)


def result_label(policy_name: str, model_version: str | None) -> str:
    """What to record as the thing that drove an Episode.

    Defaults to the Policy's registered name, so a result is labelled with
    whatever drove it and no caller has to remember — which matters most for the
    Policies that are not project work. An explicit label still wins, because
    trained weights have a version their Policy's name does not carry.
    """
    return policy_name if model_version is None else model_version


def build_policy(name: str = DEFAULT_POLICY, *, simulator=None, **kwargs) -> Policy:
    """Construct a Policy by name, mirroring
    :func:`~pathfinder.sim.carla_backend.build_simulator`.

    Args:
        name: One of :data:`POLICY_NAMES`.
        simulator: Required by Policies that read the simulator directly.
        **kwargs: Passed to the Policy's constructor.

    Raises:
        ValueError: On an unknown name, or when a Policy that needs a simulator
            was not given one — or was given one that cannot serve it.
    """
    builder = _BUILDERS.get(name.strip().lower())
    if builder is None:
        raise ValueError(
            f"unknown policy {name!r}; expected one of {', '.join(POLICY_NAMES)}"
        )
    return builder(simulator, **kwargs)
