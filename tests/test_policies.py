"""
Tests for policy selection and the CARLA behaviour-agent baseline.

The baseline is CARLA's own Policy, not project work. These tests pin the two
properties that keep that honest: it is named so no report can mistake it for
something written here, and it is chosen through the same call as every other
policy. They also pin the property that keeps the suite runnable on a machine
without CARLA — nothing is imported from CARLA until a plan is actually asked
for.

The behaviour agent itself is CARLA's, so there is nothing here worth testing
about how it drives. What is worth testing is the adapter: that it binds to the
right vehicle, rebinds when an episode restarts, converts controls faithfully,
and fails with a message that names the fix.
"""
from __future__ import annotations

import pytest

from pathfinder.cloud.queue import LocalMessageQueue
from pathfinder.orchestration import BenchmarkCoordinator, EpisodeWorker
from pathfinder.policies import (
    POLICY_NAMES,
    PURE_PURSUIT,
    CarlaBehaviorAgentPolicy,
    CILStudentPolicy,
    build_policy,
)
from pathfinder.runner import ControlOutput, PurePursuitPolicy, run_episode
from pathfinder.sim.base import Command, EpisodeSpec, FrameState
from pathfinder.sim.carla_backend import CarlaSimulator
from pathfinder.sim.kinematic import KinematicSimulator

# ─────────────────────────────────────────────────────────────────────────────
# Doubles
# ─────────────────────────────────────────────────────────────────────────────


class FakeControl:
    """Stands in for ``carla.VehicleControl``."""

    def __init__(self, throttle: float, steer: float, brake: float) -> None:
        self.throttle = throttle
        self.steer = steer
        self.brake = brake


class FakeVehicle:
    def __init__(self, actor_id: int) -> None:
        self.id = actor_id


class FakeBehaviorAgent:
    """Stands in for ``agents.navigation.behavior_agent.BehaviorAgent``."""

    instances: list[FakeBehaviorAgent] = []

    def __init__(self, vehicle, behavior="normal") -> None:
        self.vehicle = vehicle
        self.behavior = behavior
        self.destination = None
        self.steps = 0
        self.finished = False
        self.control = FakeControl(0.4, -0.2, 0.0)
        FakeBehaviorAgent.instances.append(self)

    def set_destination(self, end_location) -> None:
        self.destination = end_location

    def done(self) -> bool:
        return self.finished

    def run_step(self):
        self.steps += 1
        return self.control


#: Distinguishes "not supplied" from a deliberate ``None``.
_UNSET = object()


class FakeCarlaSimulator(KinematicSimulator):
    """A simulator that drives like the kinematic backend but exposes the two
    handles the behaviour agent needs. Lets the adapter be driven through the
    real ``run_episode`` on a machine with no CARLA."""

    def __init__(self, *, vehicle=_UNSET, destination=_UNSET) -> None:
        super().__init__()
        self._fake_vehicle = FakeVehicle(1) if vehicle is _UNSET else vehicle
        self._destination = object() if destination is _UNSET else destination

    @property
    def vehicle(self):
        return self._fake_vehicle

    @property
    def route_destination(self):
        return self._destination


@pytest.fixture(autouse=True)
def _fake_agent_class(monkeypatch):
    FakeBehaviorAgent.instances = []
    monkeypatch.setattr(
        "pathfinder.policies._behavior_agent_class", lambda: FakeBehaviorAgent
    )
    return FakeBehaviorAgent


def a_state(**overrides) -> FrameState:
    defaults = dict(
        frame_index=0,
        simulation_time=0.0,
        x=0.0,
        y=0.0,
        yaw_degrees=0.0,
        speed_mps=5.0,
        command=Command.FOLLOW_LANE,
        distance_travelled_m=0.0,
    )
    defaults.update(overrides)
    return FrameState(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────


def test_the_default_planner_is_the_geometric_one():
    assert isinstance(build_policy(), PurePursuitPolicy)


def test_every_registered_name_is_buildable(cil_checkpoint):
    """A name in the registry that cannot be built is a broken advert. The
    student's weights are its documented required argument, not a workaround."""
    simulator = FakeCarlaSimulator()
    required = {CILStudentPolicy.NAME: {"weights": cil_checkpoint}}
    for name in POLICY_NAMES:
        policy = build_policy(name, simulator=simulator, **required.get(name, {}))
        assert hasattr(policy, "plan")


def test_an_unknown_policy_names_the_alternatives():
    with pytest.raises(ValueError, match="pure_pursuit"):
        build_policy("cil")


def test_policy_names_are_normalised():
    assert isinstance(build_policy("  Pure_Pursuit "), PurePursuitPolicy)


def test_the_baseline_name_says_it_is_carlas_own_driver():
    """The name is what stops a report from crediting this to the project."""
    name = CarlaBehaviorAgentPolicy.NAME
    assert name in POLICY_NAMES
    assert "carla" in name and "builtin" in name
    assert "pathfinder" not in name


def test_the_baseline_needs_a_simulator_to_drive():
    with pytest.raises(ValueError, match="simulator"):
        build_policy(CarlaBehaviorAgentPolicy.NAME)


def test_constructor_arguments_reach_the_planner():
    policy = build_policy(
        CarlaBehaviorAgentPolicy.NAME, simulator=FakeCarlaSimulator(), behavior="cautious"
    )
    policy.plan(a_state())
    assert FakeBehaviorAgent.instances[0].behavior == "cautious"


# ─────────────────────────────────────────────────────────────────────────────
# CARLA isolation
# ─────────────────────────────────────────────────────────────────────────────


def test_nothing_from_carla_is_touched_until_a_plan_is_asked_for(monkeypatch):
    """Construction must stay import-free, or the whole suite needs CARLA."""

    def explode():
        raise AssertionError("CARLA was reached for at construction time")

    monkeypatch.setattr("pathfinder.policies._behavior_agent_class", explode)
    CarlaBehaviorAgentPolicy(FakeCarlaSimulator())


def test_the_backend_exposes_the_handles_the_baseline_binds_to():
    """The adapter reaches for these two by name; a rename must fail here
    rather than on the one machine that can run a live server."""
    assert isinstance(CarlaSimulator.vehicle, property)
    assert isinstance(CarlaSimulator.route_destination, property)


# ─────────────────────────────────────────────────────────────────────────────
# Adapting the agent
# ─────────────────────────────────────────────────────────────────────────────


def test_the_agent_is_pointed_at_the_route_destination():
    simulator = FakeCarlaSimulator()
    CarlaBehaviorAgentPolicy(simulator).plan(a_state())
    assert FakeBehaviorAgent.instances[0].destination is simulator.route_destination


def test_the_agents_control_is_passed_through():
    policy = CarlaBehaviorAgentPolicy(FakeCarlaSimulator())
    control = policy.plan(a_state())
    assert isinstance(control, ControlOutput)
    assert (control.throttle, control.steer, control.brake) == (0.4, -0.2, 0.0)
    assert control.latency_ms >= 0.0


def test_out_of_range_controls_are_clamped():
    policy = CarlaBehaviorAgentPolicy(FakeCarlaSimulator())
    policy.plan(a_state())
    FakeBehaviorAgent.instances[0].control = FakeControl(1.7, -3.0, 2.0)
    control = policy.plan(a_state())
    assert (control.throttle, control.steer, control.brake) == (1.0, -1.0, 1.0)


def test_one_agent_is_built_for_the_whole_episode():
    """Rebuilding per frame would re-plan the global route every tick and make
    the baseline's cost meaningless."""
    policy = CarlaBehaviorAgentPolicy(FakeCarlaSimulator())
    for _ in range(5):
        policy.plan(a_state())
    assert len(FakeBehaviorAgent.instances) == 1
    assert FakeBehaviorAgent.instances[0].steps == 5


def test_a_new_episode_gets_a_new_agent():
    """``reset`` destroys the ego and spawns another; an agent still holding
    the old actor would drive a vehicle that no longer exists."""
    simulator = FakeCarlaSimulator()
    policy = CarlaBehaviorAgentPolicy(simulator)
    policy.plan(a_state())

    simulator._fake_vehicle = FakeVehicle(2)
    simulator._destination = object()
    policy.plan(a_state())

    assert len(FakeBehaviorAgent.instances) == 2
    assert FakeBehaviorAgent.instances[1].vehicle is simulator.vehicle
    assert FakeBehaviorAgent.instances[1].destination is simulator.route_destination


def test_a_finished_agent_brakes_instead_of_being_stepped():
    """CARLA's agent raises once its plan is exhausted. The episode ends on the
    simulator's terms, not the agent's, so the gap has to be covered."""
    policy = CarlaBehaviorAgentPolicy(FakeCarlaSimulator())
    policy.plan(a_state())
    FakeBehaviorAgent.instances[0].finished = True

    control = policy.plan(a_state())
    assert (control.throttle, control.steer, control.brake) == (0.0, 0.0, 1.0)
    assert FakeBehaviorAgent.instances[0].steps == 1


# ─────────────────────────────────────────────────────────────────────────────
# Failure modes
# ─────────────────────────────────────────────────────────────────────────────


def test_a_backend_with_no_ego_says_so():
    policy = CarlaBehaviorAgentPolicy(FakeCarlaSimulator(vehicle=None))
    with pytest.raises(RuntimeError, match="ego vehicle"):
        policy.plan(a_state())


def test_a_backend_with_no_route_says_so():
    policy = CarlaBehaviorAgentPolicy(FakeCarlaSimulator(destination=None))
    with pytest.raises(RuntimeError, match="destination"):
        policy.plan(a_state())


def test_a_backend_without_the_handles_is_refused_up_front():
    """Handing it the kinematic backend is a plausible mistake, and 'no
    attribute vehicle' is not an answer.

    It must fail at construction rather than on the first plan: ``run_episode``
    turns a mid-Episode exception into a scored failure, so a late check spends
    a whole suite producing zero-score Episodes instead of raising once."""
    with pytest.raises(ValueError, match="CARLA"):
        CarlaBehaviorAgentPolicy(KinematicSimulator())
    with pytest.raises(ValueError, match="CARLA"):
        build_policy(CarlaBehaviorAgentPolicy.NAME, simulator=KinematicSimulator())


def test_a_misconfigured_worker_stops_before_it_drains_the_queue():
    """The failure the up-front check exists to prevent, through the path an
    operator actually types."""
    queue = LocalMessageQueue()
    queue.send(EpisodeSpec(episode_id="ep-1", route_length_m=60.0, max_steps=50).to_dict())
    worker = EpisodeWorker(
        "worker-0", queue, policy_name=CarlaBehaviorAgentPolicy.NAME,
        simulator_backend="kinematic", idle_timeout_seconds=0.1,
    )

    with pytest.raises(ValueError, match="CARLA"):
        worker.run()
    assert queue.receive(max_messages=1, wait_seconds=0.0), "the episode must still be queued"


# ─────────────────────────────────────────────────────────────────────────────
# It plugs into the existing loop
# ─────────────────────────────────────────────────────────────────────────────


def test_the_baseline_drives_a_scored_episode_through_the_normal_runner():
    """No new seam: the unmodified runner consumes it like any other policy."""
    simulator = FakeCarlaSimulator()
    policy = build_policy(CarlaBehaviorAgentPolicy.NAME, simulator=simulator)
    spec = EpisodeSpec(episode_id="baseline-1", seed=3, max_steps=40)

    score = run_episode(simulator, spec, policy, model_version=CarlaBehaviorAgentPolicy.NAME)

    assert score.status == "completed"
    assert score.frames > 0
    assert FakeBehaviorAgent.instances[0].steps == score.frames


# ─────────────────────────────────────────────────────────────────────────────
# Selection through the orchestration layer
# ─────────────────────────────────────────────────────────────────────────────


def test_a_worker_selects_its_planner_by_name():
    queue = LocalMessageQueue()
    queue.send(EpisodeSpec(episode_id="ep-1", route_length_m=60.0, max_steps=200).to_dict())
    worker = EpisodeWorker(
        "worker-0", queue, policy_name=PURE_PURSUIT,
        simulator_backend="kinematic", idle_timeout_seconds=0.1,
    )

    assert len(worker.run()) == 1


def test_a_result_is_labelled_with_whatever_drove_it():
    """The point of naming the baseline is lost if the label has to be set by
    hand at every call site."""
    queue = LocalMessageQueue()
    assert EpisodeWorker("w", queue).model_version == PURE_PURSUIT
    assert (
        EpisodeWorker("w", queue, policy_name=CarlaBehaviorAgentPolicy.NAME).model_version
        == CarlaBehaviorAgentPolicy.NAME
    )
    assert (
        BenchmarkCoordinator(queue, policy_name=CarlaBehaviorAgentPolicy.NAME).model_version
        == CarlaBehaviorAgentPolicy.NAME
    )
    # An explicit label still wins — trained weights have a version the
    # policy's name does not carry.
    assert EpisodeWorker("w", queue, model_version="cil-v3").model_version == "cil-v3"


def test_every_telemetry_frame_carries_what_drove_it():
    """Provenance on the frame, not just on the submitted result: a Parquet
    partition is read long after the run, by someone who was not there."""
    rows: list[dict] = []
    run_episode(
        KinematicSimulator(),
        EpisodeSpec(episode_id="ep-1", route_length_m=60.0, max_steps=30),
        build_policy(),
        telemetry_sink=rows.append,
        model_version=CarlaBehaviorAgentPolicy.NAME,
        dataset_version="kitti-seqsplit-v1",
    )

    assert rows
    assert {row["model_version"] for row in rows} == {CarlaBehaviorAgentPolicy.NAME}
    assert {row["dataset_version"] for row in rows} == {"kitti-seqsplit-v1"}
