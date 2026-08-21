"""DAgger loop: backend/expert injection seam, aggregation contract, β schedule.

Everything here runs the real loop with tiny models and a scripted simulator —
no CARLA, no GPU, no network. The stubs implement the public protocols
(``SimulatorBackend``, ``Policy``); nothing reaches into ``dagger`` internals.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pathfinder.dagger import DAggerConfig, IterationReport, run_dagger
from pathfinder.runner import ControlOutput, PurePursuitPolicy
from pathfinder.sim.base import EpisodeSpec, FrameState, SimulatorBackend, StepResult
from pathfinder.sim.kinematic import KinematicSimulator
from tests.support import make_frame

# (steer, throttle, brake). Every component is exactly representable in float32
# (the loop routes actions through float32 arrays), survives CILPolicy's control
# clamps unchanged, and each student component sits exactly 0.5 from the
# expert's — so an L1 training loss on expert labels is exactly 0.5, and a loss
# on (mislabelled) student actions would be exactly 0.0.
EXPERT_ACTION = (0.25, 0.5, 0.0)
STUDENT_ACTION = (0.75, 1.0, 0.5)


class TinyBranchModel(nn.Module):
    """Smallest trainable thing with the CIL model's interface: 4 × (B, 3)."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.ModuleList(nn.Linear(3, 3) for _ in range(4))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        z = self.pool(x).flatten(1)
        return [head(z) for head in self.heads]


class ConstantStudentModel(nn.Module):
    """Always predicts ``STUDENT_ACTION``, so what the student does is exact.

    The dummy parameter keeps the output on the autograd graph (the training
    step calls ``backward``) without letting training change the prediction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.zero = nn.Parameter(torch.zeros(1))
        action = torch.tensor(STUDENT_ACTION, dtype=torch.float32)
        self.register_buffer("action", action)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.action.expand(x.shape[0], 3) + self.zero * 0.0
        return [out for _ in range(4)]


class ScriptedSimulator(SimulatorBackend):
    """Emits a fixed number of rendered frames and records every control."""

    def __init__(self, steps_per_episode: int = 5) -> None:
        self.steps_per_episode = steps_per_episode
        self.received_controls: list[tuple[float, float, float]] = []
        self.close_calls = 0
        self._frame = 0
        self._episode_step = 0

    def _state(self) -> FrameState:
        rng = np.random.default_rng(self._frame)
        state = make_frame(
            frame_index=self._frame,
            simulation_time=self._frame * 0.05,
            x=float(self._frame),
            distance_travelled_m=float(self._frame),
            image=rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8),
        )
        self._frame += 1
        return state

    def reset(self, spec: EpisodeSpec) -> FrameState:
        self._episode_step = 0
        return self._state()

    def step(self, throttle: float, steer: float, brake: float) -> StepResult:
        self.received_controls.append((throttle, steer, brake))
        self._episode_step += 1
        done = self._episode_step >= self.steps_per_episode
        return StepResult(state=self._state(), done=done)

    def close(self) -> None:
        self.close_calls += 1

    @property
    def name(self) -> str:
        return "scripted"

    def acted_frames(self, episodes: int) -> list[int]:
        """Frame indices an action was taken at: each episode emits
        ``steps_per_episode + 1`` frames and the terminal one is never acted on."""
        per_episode = self.steps_per_episode + 1
        return [
            episode * per_episode + step
            for episode in range(episodes)
            for step in range(self.steps_per_episode)
        ]


class RecordingExpert:
    """Constant-action expert that records every state it is asked to label."""

    def __init__(self) -> None:
        self.labelled_frames: list[int] = []

    def plan(self, state: FrameState) -> ControlOutput:
        self.labelled_frames.append(state.frame_index)
        steer, throttle, brake = EXPERT_ACTION
        return ControlOutput(throttle=throttle, steer=steer, brake=brake)


def tiny_config(**overrides) -> DAggerConfig:
    base = dict(
        iterations=2,
        episodes_per_iteration=1,
        route_length_m=10.0,
        max_steps=5,
        epochs_per_iteration=1,
        batch_size=4,
        # β at iteration 1 is beta_decay**1 ≈ 0: student control, expert labels.
        beta_decay=1e-9,
        seed=11,
    )
    base.update(overrides)
    return DAggerConfig(**base)


def run_scripted(model: nn.Module, **config_overrides):
    """Run the loop on a stub backend; return (simulator, expert, report)."""
    config = tiny_config(**config_overrides)
    simulator = ScriptedSimulator(steps_per_episode=config.max_steps)
    expert = RecordingExpert()
    _, report = run_dagger(
        config, model=model, simulator=simulator, expert=expert, progress=False
    )
    return simulator, expert, report


def test_injected_expert_labels_every_acted_state():
    simulator, expert, _ = run_scripted(TinyBranchModel())

    assert expert.labelled_frames == simulator.acted_frames(episodes=2)


def test_expert_labels_states_reached_under_student_control():
    simulator, expert, _ = run_scripted(ConstantStudentModel())

    steer, throttle, brake = EXPERT_ACTION
    iter0 = simulator.received_controls[:5]
    iter1 = simulator.received_controls[5:]

    # Iteration 0 is pure expert (β=1): the simulator drives on expert actions.
    assert all(control == (throttle, steer, brake) for control in iter0)
    # Iteration 1 (β≈0) drives on the student's actions — every step of it —
    # yet the expert still labelled every state that rollout visited.
    steer, throttle, brake = STUDENT_ACTION
    assert all(control == (throttle, steer, brake) for control in iter1)
    student_frames = simulator.acted_frames(episodes=2)[5:]
    assert [f for f in expert.labelled_frames if f in student_frames] == student_frames


def test_aggregate_dataset_holds_the_injected_experts_labels():
    # The constant student predicts exactly 0.5 (L1) from the expert's action on
    # every component, and training cannot move it. So the train loss is exactly
    # 0.5 if and only if the aggregate dataset is labelled with the *expert's*
    # actions — had the loop stored the student's own actions instead, the loss
    # would be exactly 0.0. Iteration 1's dataset includes the states reached
    # under student control, so the assertion covers those too.
    _, _, report = run_scripted(ConstantStudentModel())

    assert [item.train_loss for item in report.iterations] == [0.5, 0.5]


def test_beta_schedule_pure_expert_then_decay():
    _, _, report = run_scripted(TinyBranchModel(), iterations=3, beta_decay=0.5)

    assert [item.beta for item in report.iterations] == [1.0, 0.5, 0.25]


def test_injected_simulator_is_not_closed():
    simulator, _, _ = run_scripted(TinyBranchModel())

    # The caller opened the backend, so the caller owns its lifetime — a CARLA
    # connection must survive the training run that borrowed it.
    assert simulator.close_calls == 0


def report_row_key(item):
    """One iteration's numbers, minus wall-clock, for cross-run comparison."""
    return (
        item.iteration,
        item.beta,
        item.dataset_size,
        item.train_loss,
        "nan" if np.isnan(item.disagreement) else item.disagreement,
        item.driving_score,
        item.route_completion,
    )


def test_on_iteration_sees_each_iterations_new_samples_and_trained_row():
    from pathfinder.dagger import IterationSamples

    seen: list[tuple[int, int]] = []

    def on_iteration(row, samples, model, optimizer):
        # The callback fires after training, so the row is final and the model
        # already reflects it — that is what a checkpoint must capture.
        assert isinstance(samples, IterationSamples)
        assert len(samples.images) == len(samples.commands) == len(samples.actions)
        assert optimizer.state_dict() is not None
        seen.append((row.iteration, len(samples.images)))

    config = tiny_config()
    simulator = ScriptedSimulator(steps_per_episode=config.max_steps)
    _, report = run_dagger(
        config, model=TinyBranchModel(), simulator=simulator,
        expert=RecordingExpert(), progress=False, on_iteration=on_iteration,
    )

    # One call per iteration, and the per-iteration sample counts sum to the
    # final aggregate dataset size — nothing collected escapes the callback.
    assert [iteration for iteration, _ in seen] == [0, 1]
    assert sum(count for _, count in seen) == report.iterations[-1].dataset_size


def test_resume_reproduces_an_uninterrupted_run_exactly():
    """Killing a run after iteration k and resuming from its checkpoint must
    reproduce the uninterrupted run's remaining iterations bit-identically —
    the property that makes 'a crash costs one iteration' true rather than
    'a crash costs one iteration and changes the numbers'."""
    import copy

    from pathfinder.dagger import DAggerResume, IterationSamples

    def uninterrupted():
        torch.manual_seed(0)
        with KinematicSimulator(render=True) as simulator:
            _, report = run_dagger(
                tiny_config(iterations=3, max_steps=10, route_length_m=30.0, beta_decay=0.5),
                model=TinyBranchModel(),
                simulator=simulator,
                progress=False,
            )
        return [report_row_key(item) for item in report.iterations]

    def killed_then_resumed():
        torch.manual_seed(0)
        checkpoints = []

        def on_iteration(row, samples, model, optimizer):
            checkpoints.append(
                (row, samples, copy.deepcopy(optimizer.state_dict()))
            )

        # First sitting: two iterations, then the "crash".
        with KinematicSimulator(render=True) as simulator:
            model, _ = run_dagger(
                tiny_config(iterations=2, max_steps=10, route_length_m=30.0, beta_decay=0.5),
                model=TinyBranchModel(),
                simulator=simulator,
                progress=False,
                on_iteration=on_iteration,
            )

        rows = [row for row, _, _ in checkpoints]
        samples = IterationSamples(images=[], commands=[], actions=[])
        for _, batch, _ in checkpoints:
            samples.images.extend(batch.images)
            samples.commands.extend(batch.commands)
            samples.actions.extend(batch.actions)
        resume = DAggerResume(
            rows=rows, samples=samples, optimizer_state=checkpoints[-1][2]
        )

        # Second sitting: same target config, fresh simulator, resumed state.
        with KinematicSimulator(render=True) as simulator:
            _, report = run_dagger(
                tiny_config(iterations=3, max_steps=10, route_length_m=30.0, beta_decay=0.5),
                model=model,
                simulator=simulator,
                progress=False,
                resume_from=resume,
            )
        return [report_row_key(item) for item in report.iterations]

    assert killed_then_resumed() == uninterrupted()


def test_resume_rejects_already_complete_state():
    import pytest

    from pathfinder.dagger import DAggerResume, IterationSamples

    rows = [
        IterationReport(
            iteration=index, beta=1.0, dataset_size=1, train_loss=0.0,
            disagreement=0.0, driving_score=0.0, route_completion=0.0, seconds=0.0,
        )
        for index in range(2)
    ]
    resume = DAggerResume(
        rows=rows,
        samples=IterationSamples(images=[], commands=[], actions=[]),
        optimizer_state=None,
    )
    with pytest.raises(ValueError, match="already"):
        run_dagger(tiny_config(iterations=2), model=TinyBranchModel(),
                   simulator=ScriptedSimulator(), expert=RecordingExpert(),
                   progress=False, resume_from=resume)


def test_defaults_reproduce_kinematic_pure_pursuit_exactly():
    def run(**kwargs):
        torch.manual_seed(0)
        _, report = run_dagger(
            tiny_config(max_steps=10, route_length_m=30.0, beta_decay=0.5),
            model=TinyBranchModel(),
            progress=False,
            **kwargs,
        )
        # Unrounded per-iteration numbers, minus wall-clock; NaN (iteration 0
        # has no student, so no disagreement) normalised so it compares equal.
        return [
            (
                item.iteration,
                item.beta,
                item.dataset_size,
                item.train_loss,
                "nan" if np.isnan(item.disagreement) else item.disagreement,
                item.driving_score,
                item.route_completion,
            )
            for item in report.iterations
        ]

    defaulted = run()
    with KinematicSimulator(render=True) as simulator:
        explicit = run(simulator=simulator, expert=PurePursuitPolicy())

    assert defaulted == explicit
