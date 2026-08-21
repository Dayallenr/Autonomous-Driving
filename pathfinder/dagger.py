"""
DAgger: Dataset Aggregation for the conditional imitation-learning policy.

    python -m pathfinder.dagger --smoke
    python -m pathfinder.dagger --backend carla --device cuda --iterations 10

One command runs the loop and lands its artifacts together: per-iteration
weight checkpoints, a report JSON, and a generated write-up
(``pathfinder/dagger_writeup.py``) that labels what the numbers may mean.
Every iteration is checkpointed as it completes, and re-running the same
command in the same ``--output-dir`` resumes a killed run from its last
completed iteration — a crash costs one iteration, not the sitting.

The problem DAgger solves
-------------------------
Plain behaviour cloning trains on the *expert's* state distribution but is
evaluated on the *student's*. The moment the student makes a small error it
reaches a state the expert never visited, has no idea what to do, errs further,
and compounds. Ross et al. (2011) show the error grows quadratically in the
horizon for behaviour cloning versus linearly for DAgger.

The fix is to train on the distribution the student actually induces:

1. Roll out the **student** (it visits its own mistakes).
2. Ask the **expert** what it would have done at every visited state.
3. Add those (state, expert action) pairs to the aggregate dataset.
4. Retrain on everything collected so far.
5. Repeat.

The default expert is :class:`PurePursuitPolicy`, which is *privileged*: it
reads exact cross-track error, heading error, and curvature from the simulator.
The student sees only rendered pixels. That asymmetry is the point — the student
learns to infer from images what the expert is handed directly, which is exactly
the privileged-expert setup used by LBC and Roach. Both the expert and the
simulator backend are injectable through :func:`run_dagger`, so the same loop
can be pointed at CARLA with a different Policy in the expert seat — though no
CARLA-backed DAgger run has happened yet.

Beta scheduling
---------------
Iteration 0 rolls out the expert (β=1) because an untrained student produces a
useless state distribution. Thereafter β decays so control moves to the student.
Mixing rather than switching hard avoids the first student rollout immediately
leaving the road and collecting a dataset entirely of crash states.

What gets measured
------------------
**Expert-student action disagreement** (mean L1 over steer/throttle/brake) on
states drawn from the *student's own* rollouts. This is the quantity DAgger is
designed to reduce, and measuring it on the student's distribution rather than
the expert's is what makes the number meaningful.

Compute required — read this before quoting a number
----------------------------------------------------
A default CPU run of this module is a **smoke test**, not a result. On a few
thousand frames with a randomly-initialised ResNet-18 and a couple of epochs,
the student is badly underfit and disagreement typically *rises* across
iterations rather than falling. That is not a broken loop; it is the expected
ordering of two effects:

* as β decays the student drives more and reaches worse states, where the expert
  takes more extreme corrective actions — this pushes disagreement **up**;
* as the network fits the aggregate dataset, it predicts those corrections
  better — this pushes disagreement **down**.

The second effect only dominates once the model has enough data and epochs to
actually fit. Reaching that point needs an ImageNet-initialised backbone, tens
of thousands of frames, and tens of epochs — which is a GPU job. Use
``notebooks/train_cil_dagger.ipynb`` for a run whose error-reduction number
means something, and treat CPU output here as a check that the loop executes.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pathfinder.dagger_writeup import render_writeup
from pathfinder.runner import ControlOutput, Policy, PurePursuitPolicy
from pathfinder.sim.base import EpisodeSpec, FrameState, SimulatorBackend
from pathfinder.sim.kinematic import KinematicSimulator

logger = logging.getLogger(__name__)

__all__ = [
    "CILPolicy",
    "DAggerConfig",
    "DAggerReport",
    "DAggerResume",
    "IterationReport",
    "IterationSamples",
    "run_dagger",
]

# ImageNet statistics: the ResNet-18 backbone is pretrained with these, and
# feeding it differently-normalized input wastes most of the transfer.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: np.ndarray) -> torch.Tensor:
    """Convert a rendered HWC uint8 frame to a normalized CHW float tensor."""
    array = image.astype(np.float32) / 255.0
    array = (array - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1))


class CILPolicy:
    """Wraps a CIL model so it satisfies the :class:`Policy` protocol.

    Selects the branch matching the current command, which is what makes this
    *conditional* imitation learning: at an intersection the same image has
    several correct actions, and a single-head network trained on all of them
    regresses to their average — driving straight into the median. Branching on
    the command removes that ambiguity.
    """

    def __init__(self, model: nn.Module, *, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def plan(self, state: FrameState) -> ControlOutput:
        started = time.perf_counter()
        if state.image is None:
            raise ValueError(
                "CILPolicy requires rendered frames; construct the simulator "
                "with KinematicSimulator(render=True)"
            )
        tensor = preprocess(state.image).unsqueeze(0).to(self.device)
        branches = self.model(tensor)
        output = branches[int(state.command)][0]

        steer = float(torch.clamp(output[0], -1.0, 1.0))
        throttle = float(torch.clamp(output[1], 0.0, 1.0))
        brake = float(torch.clamp(output[2], 0.0, 1.0))
        return ControlOutput(
            throttle=throttle,
            steer=steer,
            brake=brake,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


@dataclass
class DAggerConfig:
    """Hyperparameters for a DAgger run."""

    iterations: int = 4
    episodes_per_iteration: int = 3
    route_length_m: float = 250.0
    max_steps: int = 700
    epochs_per_iteration: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-3
    #: β decay per iteration. β is the probability of using the expert's action.
    beta_decay: float = 0.5
    device: str = "cpu"
    seed: int = 7
    max_dataset_size: int = 40_000
    #: Towns and weathers cycled across the run's episodes, mirroring
    #: ``build_episode_specs``'s suite (weather strides by 3 so combinations
    #: vary). The kinematic backend ignores both; on CARLA they keep a training
    #: run from seeing a single town in a single weather for its whole life.
    towns: tuple[str, ...] = ("Town01", "Town03", "Town05")
    weathers: tuple[str, ...] = ("ClearNoon", "WetNoon", "HardRainNoon", "ClearSunset")


@dataclass
class IterationReport:
    """Metrics for one DAgger iteration."""

    iteration: int
    beta: float
    dataset_size: int
    train_loss: float
    #: Mean L1 expert-student control disagreement on the student's own states.
    disagreement: float
    driving_score: float
    route_completion: float
    seconds: float

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "beta": round(self.beta, 3),
            "dataset_size": self.dataset_size,
            "train_loss": round(self.train_loss, 5),
            # Iteration 0 has no student, so no disagreement; None rather than
            # NaN because NaN is not JSON and poisons artifact comparison.
            "disagreement": None if np.isnan(self.disagreement) else round(self.disagreement, 5),
            "driving_score": round(self.driving_score, 2),
            "route_completion": round(self.route_completion, 4),
            "seconds": round(self.seconds, 1),
        }


@dataclass
class DAggerReport:
    """Full DAgger run."""

    iterations: list[IterationReport] = field(default_factory=list)
    error_reduction: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "iterations": [item.to_dict() for item in self.iterations],
            "error_reduction_pct": round(self.error_reduction * 100, 1),
            "total_seconds": round(self.total_seconds, 1),
        }


@dataclass
class IterationSamples:
    """The (state, expert action) pairs one iteration added to the dataset.

    Handed to ``on_iteration`` so a checkpoint can persist exactly the new
    samples — re-persisting the whole aggregate every iteration would make
    checkpoint cost grow with the run instead of with the iteration.
    """

    images: list[np.ndarray]
    commands: list[int]
    actions: list[np.ndarray]


@dataclass
class DAggerResume:
    """State reconstructed from checkpoints to continue a killed run.

    ``rows`` are the completed iterations' reports; the run continues at
    iteration ``len(rows)``. ``samples`` is the aggregate dataset those
    iterations collected (it is re-trimmed to ``max_dataset_size`` on entry).
    The model's weights are the caller's job: load them into the ``model``
    passed to :func:`run_dagger` before calling.
    """

    rows: list[IterationReport]
    samples: IterationSamples
    optimizer_state: dict | None = None


def _error_reduction(rows: list[IterationReport]) -> float | None:
    """First measured disagreement to last. Iteration 0 has no student, so its
    NaN is skipped; below two measurements the reduction is undefined (None) —
    reporting 0.0 there would dress 'not measurable' up as 'no change'."""
    measured = [row.disagreement for row in rows if not np.isnan(row.disagreement)]
    if len(measured) >= 2 and measured[0] > 0:
        return (measured[0] - measured[-1]) / measured[0]
    return None


def _expert_control(policy: Policy, state: FrameState) -> np.ndarray:
    control = policy.plan(state)
    return np.array([control.steer, control.throttle, control.brake], dtype=np.float32)


def _collect(
    simulator: SimulatorBackend,
    expert: Policy,
    student: CILPolicy | None,
    spec: EpisodeSpec,
    beta: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[float], float]:
    """Roll out one episode under a β-mixed policy, labelling with the expert.

    Returns ``(images, commands, expert_actions, disagreements, completion)``.
    Every visited state is labelled by the expert regardless of who acted — that
    is the aggregation step, and it is why the dataset covers the student's
    mistakes rather than only the expert's trajectory.
    """
    images: list[np.ndarray] = []
    commands: list[int] = []
    actions: list[np.ndarray] = []
    disagreements: list[float] = []

    state = simulator.reset(spec)
    for _ in range(spec.max_steps):
        expert_action = _expert_control(expert, state)

        if state.image is not None:
            images.append(state.image.copy())
            commands.append(int(state.command))
            actions.append(expert_action)

        if student is not None:
            student_control = student.plan(state)
            student_action = np.array(
                [student_control.steer, student_control.throttle, student_control.brake],
                dtype=np.float32,
            )
            disagreements.append(float(np.abs(student_action - expert_action).mean()))
            use_expert = rng.random() < beta
            chosen = expert_action if use_expert else student_action
        else:
            chosen = expert_action

        result = simulator.step(float(chosen[1]), float(chosen[0]), float(chosen[2]))
        state = result.state
        if result.done:
            break

    completion = min(1.0, state.distance_travelled_m / spec.route_length_m)
    return images, commands, actions, disagreements, completion


def run_dagger(
    config: DAggerConfig | None = None,
    *,
    model: nn.Module | None = None,
    simulator: SimulatorBackend | None = None,
    expert: Policy | None = None,
    progress: bool = True,
    on_iteration: Callable[[IterationReport, IterationSamples, nn.Module, torch.optim.Optimizer], None]
    | None = None,
    resume_from: DAggerResume | None = None,
) -> tuple[nn.Module, DAggerReport]:
    """Run the DAgger loop and return the trained student plus its report.

    Args:
        simulator: Backend to roll out in. Must produce rendered frames
            (``FrameState.image``) — the student is trained on pixels. Defaults
            to ``KinematicSimulator(render=True)``. An injected simulator is
            **not** closed here; the caller that opened it owns its lifetime
            (a CARLA connection outlives any one training run). The default,
            internally-constructed one is closed on exit as before.
        expert: Labeller for every visited state; anything satisfying
            :class:`Policy`. Defaults to :class:`PurePursuitPolicy`.
        on_iteration: Called after each iteration's training with the finished
            report row, the samples that iteration added, and the model and
            optimizer as trained — everything a checkpoint must persist so a
            crash costs one iteration, not the run.
        resume_from: State recovered from checkpoints; the run continues at
            iteration ``len(resume_from.rows)``. The caller must have loaded
            the matching weights into ``model``. All randomness is derived per
            iteration from ``config.seed``, so a resumed run reproduces the
            uninterrupted run's remaining iterations exactly.

    Raises:
        ValueError: If ``config.iterations`` is below 2 — error *reduction* is
            undefined without at least a first and last measurement — or if
            ``resume_from`` already holds every iteration.
    """
    config = config or DAggerConfig()
    if config.iterations < 2:
        raise ValueError(
            f"need at least 2 iterations to measure error reduction, got {config.iterations}"
        )
    start_iteration = len(resume_from.rows) if resume_from is not None else 0
    if resume_from is not None and start_iteration >= config.iterations:
        raise ValueError(
            f"resume state already holds {start_iteration} iterations; the run of "
            f"{config.iterations} is already complete"
        )

    from pathfinder.planning.cil_model import CILModel

    # Seeds the default model's weight init; each iteration re-seeds below.
    torch.manual_seed(config.seed)

    if model is None:
        # pretrained=False keeps the run offline and reproducible; a real run
        # should use ImageNet initialisation.
        model = CILModel(pretrained=False, output_mode="control")
    model = model.to(config.device)

    if expert is None:
        expert = PurePursuitPolicy()
    owns_simulator = simulator is None
    if simulator is None:
        simulator = KinematicSimulator(render=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.L1Loss()

    aggregate_images: list[np.ndarray] = []
    aggregate_commands: list[int] = []
    aggregate_actions: list[np.ndarray] = []

    report = DAggerReport()
    if resume_from is not None:
        report.iterations.extend(resume_from.rows)
        aggregate_images.extend(resume_from.samples.images)
        aggregate_commands.extend(resume_from.samples.commands)
        aggregate_actions.extend(resume_from.samples.actions)
        # Checkpoints store each iteration's samples untrimmed; re-applying the
        # keep-most-recent bound here reproduces exactly the aggregate the
        # uninterrupted run would have held at this point.
        if len(aggregate_images) > config.max_dataset_size:
            overflow = len(aggregate_images) - config.max_dataset_size
            del aggregate_images[:overflow]
            del aggregate_commands[:overflow]
            del aggregate_actions[:overflow]
        if resume_from.optimizer_state is not None:
            optimizer.load_state_dict(resume_from.optimizer_state)
    run_started = time.perf_counter()

    try:
        for iteration in range(start_iteration, config.iterations):
            iteration_started = time.perf_counter()
            # Every stream of randomness is derived from (seed, iteration), not
            # threaded through the loop, so iteration k draws the same numbers
            # whether the run reached it in one sitting or resumed from a
            # checkpoint — without this, "resume" would silently change results.
            torch.manual_seed(config.seed + iteration + 1)
            rng = np.random.default_rng(config.seed + iteration + 1)

            # Iteration 0 is pure expert: an untrained student's rollouts are
            # noise, and aggregating them first would poison the dataset.
            beta = 1.0 if iteration == 0 else config.beta_decay**iteration
            student = None if iteration == 0 else CILPolicy(model, device=config.device)

            new_samples = IterationSamples(images=[], commands=[], actions=[])
            episode_disagreements: list[float] = []
            completions: list[float] = []
            for episode in range(config.episodes_per_iteration):
                suite_index = iteration * config.episodes_per_iteration + episode
                spec = EpisodeSpec(
                    episode_id=f"dagger-{iteration}-{episode}",
                    town=config.towns[suite_index % len(config.towns)],
                    weather=config.weathers[(suite_index * 3) % len(config.weathers)],
                    route_length_m=config.route_length_m,
                    max_steps=config.max_steps,
                    seed=config.seed + iteration * 100 + episode,
                )
                images, commands, actions, disagreements, completion = _collect(
                    simulator, expert, student, spec, beta, rng
                )
                new_samples.images.extend(images)
                new_samples.commands.extend(commands)
                new_samples.actions.extend(actions)
                episode_disagreements.extend(disagreements)
                completions.append(completion)

            aggregate_images.extend(new_samples.images)
            aggregate_commands.extend(new_samples.commands)
            aggregate_actions.extend(new_samples.actions)
            # Bound memory: keep the most recent samples, which are also the
            # ones from the current (most relevant) student distribution.
            if len(aggregate_images) > config.max_dataset_size:
                overflow = len(aggregate_images) - config.max_dataset_size
                del aggregate_images[:overflow]
                del aggregate_commands[:overflow]
                del aggregate_actions[:overflow]

            train_loss = _train(
                model, optimizer, criterion,
                aggregate_images, aggregate_commands, aggregate_actions,
                config,
            )

            disagreement = (
                float(np.mean(episode_disagreements)) if episode_disagreements else float("nan")
            )
            completion = float(np.mean(completions))
            row = IterationReport(
                iteration=iteration,
                beta=beta,
                dataset_size=len(aggregate_images),
                train_loss=train_loss,
                disagreement=disagreement,
                driving_score=completion * 100.0,
                route_completion=completion,
                seconds=time.perf_counter() - iteration_started,
            )
            report.iterations.append(row)
            if on_iteration is not None:
                on_iteration(row, new_samples, model, optimizer)
            if progress:
                logger.info(
                    "DAgger iter %d: beta=%.2f n=%d loss=%.4f disagreement=%s completion=%.3f",
                    iteration, beta, len(aggregate_images), train_loss,
                    "n/a" if np.isnan(disagreement) else f"{disagreement:.4f}",
                    completion,
                )
    finally:
        if owns_simulator:
            simulator.close()

    reduction = _error_reduction(report.iterations)
    # The dataclass field predates the None convention and stays a float for
    # existing callers (the notebook); the JSON artifact built by
    # build_run_report keeps the honest None.
    report.error_reduction = 0.0 if reduction is None else reduction
    report.total_seconds = time.perf_counter() - run_started
    return model, report


def _train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    images: list[np.ndarray],
    commands: list[int],
    actions: list[np.ndarray],
    config: DAggerConfig,
) -> float:
    """Train on the aggregate dataset, applying loss only to the active branch."""
    if not images:
        return float("nan")

    model.train()
    indices = np.arange(len(images))
    total_loss = 0.0
    batches = 0

    for epoch in range(config.epochs_per_iteration):
        # Seed varies per epoch. Re-seeding with the same value every epoch
        # produced the identical batch order each time, which removes the point
        # of shuffling and correlates gradient noise across epochs.
        np.random.default_rng(config.seed + epoch).shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            batch = indices[start : start + config.batch_size]
            if len(batch) == 0:
                continue

            tensor = torch.stack([preprocess(images[i]) for i in batch]).to(config.device)
            targets = torch.from_numpy(np.stack([actions[i] for i in batch])).to(config.device)
            branch_ids = torch.tensor([commands[i] for i in batch], dtype=torch.long)

            outputs = model(tensor)
            # Gather each sample's prediction from its own command branch. Loss
            # on all branches would train every branch on every command, which
            # is exactly the averaging that conditional branching exists to stop.
            stacked = torch.stack(outputs, dim=1)  # (B, 4, 3)
            selected = stacked[torch.arange(len(batch)), branch_ids]

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(selected, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

    model.eval()
    return total_loss / batches if batches else float("nan")


#: What a run must satisfy before its figures may be quoted as training
#: results rather than a loop check. The thresholds restate the compute notes
#: in the module docstring as numbers: below an ImageNet-initialised backbone,
#: tens of thousands of frames, and tens of total epochs the student is badly
#: underfit and its disagreement numbers mean nothing — and only the CARLA
#: backend produces driving that is driving at all.
REAL_RUN_BAR = {
    "backend": "carla",
    "imagenet_pretrained_backbone": True,
    "min_final_dataset_size": 20_000,
    "min_total_epochs": 20,
}

#: Stamped into every run report so the meaning of the per-iteration rollout
#: numbers travels with them.
ITERATION_METRICS_NOTE = (
    "Per-iteration driving_score and route_completion are measured under the "
    "β-mixed expert/student policy with no infraction scoring; they are loop "
    "diagnostics, not the student's driving score. The scored comparison "
    "suite is the only source of quotable driving scores."
)


def _real_run_bar(backend: str, pretrained: bool, config: DAggerConfig,
                  rows: list[IterationReport]) -> dict:
    final_size = rows[-1].dataset_size
    total_epochs = config.iterations * config.epochs_per_iteration
    criteria = {
        "backend": {
            "required": REAL_RUN_BAR["backend"],
            "actual": backend,
            "met": backend == REAL_RUN_BAR["backend"],
        },
        "imagenet_pretrained_backbone": {
            "required": REAL_RUN_BAR["imagenet_pretrained_backbone"],
            "actual": bool(pretrained),
            "met": bool(pretrained),
        },
        "final_dataset_size": {
            "required": REAL_RUN_BAR["min_final_dataset_size"],
            "actual": final_size,
            "met": final_size >= REAL_RUN_BAR["min_final_dataset_size"],
        },
        "total_epochs": {
            "required": REAL_RUN_BAR["min_total_epochs"],
            "actual": total_epochs,
            "met": total_epochs >= REAL_RUN_BAR["min_total_epochs"],
        },
    }
    return {"criteria": criteria, "met": all(item["met"] for item in criteria.values())}


def _run_scope(bar: dict) -> tuple[str, str]:
    """What the report's numbers are allowed to mean, mirroring the ablation's
    scope labelling: the safe failure mode is underclaiming."""
    if bar["met"]:
        return (
            "training-run",
            "Generated by a configuration meeting the documented real-run bar; "
            "its figures describe this training run under the stated setup. "
            "Driving quality is still measured only by the scored comparison "
            "suite, not by this loop's rollout numbers.",
        )
    unmet = ", ".join(sorted(
        name for name, item in bar["criteria"].items() if not item["met"]
    ))
    return (
        "smoke",
        f"Generated by a configuration below the documented real-run bar "
        f"(unmet: {unmet}). Smoke output is a loop check: it verifies the "
        "DAgger mechanism executes end to end and nothing else. None of these "
        "numbers is a result, and none may ever be quoted as one.",
    )


def build_run_report(
    *,
    backend: str,
    expert_name: str,
    student: dict,
    config: DAggerConfig,
    rows: list[IterationReport],
    weights: str,
) -> dict:
    """Assemble a DAgger run's report artifact from its per-iteration rows.

    Built from rows rather than from a :class:`DAggerReport` so a run that
    crashed after its last iteration checkpoint — but before this report —
    can still be finished from checkpoints alone. ``total_seconds`` is
    therefore the sum of per-iteration wall-clock, which survives a resume;
    a whole-run stopwatch would not.

    Args:
        student: Identity of the trained model as the CLI built it; must carry
            ``imagenet_pretrained``, which the real-run bar reads.
        weights: Path of the final checkpoint, relative to the run directory.
    """
    bar = _real_run_bar(backend, student["imagenet_pretrained"], config, rows)
    scope, scope_note = _run_scope(bar)
    return {
        "kind": "dagger_run",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": backend,
        "scope": scope,
        "scope_note": scope_note,
        "expert": expert_name,
        "student": student,
        # JSON round-trip normalises the config's tuples to lists so the
        # stored artifact equals its own re-parse.
        "config": json.loads(json.dumps(asdict(config))),
        "real_run_bar": bar,
        "iteration_metrics_note": ITERATION_METRICS_NOTE,
        "iterations": [row.to_dict() for row in rows],
        "error_reduction_pct": (
            None if (reduction := _error_reduction(rows)) is None else round(reduction * 100, 1)
        ),
        "total_seconds": round(sum(row.seconds for row in rows), 1),
        "weights": weights,
    }


class _RunDirectory:
    """Layout and lifecycle of one DAgger run's artifacts.

    ::

        <root>/
          manifest.json               the configuration this directory is bound to
          checkpoints/iteration_NNN.pt  model + optimizer + report row, per iteration
          data/iteration_NNN.npz        that iteration's new samples (deleted on finish)
          report.json / report.md       written only when the run completes

    An iteration counts as complete only when both its files exist; the ``.pt``
    is written last (atomically, via rename) so its presence is the marker.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.checkpoints = self.root / "checkpoints"
        self.data = self.root / "data"
        self.manifest_path = self.root / "manifest.json"
        self.report_path = self.root / "report.json"

    def prepare(self, manifest: dict) -> None:
        """Create the layout, binding the directory to this configuration.

        Raises:
            SystemExit: If the directory already holds a run with a different
                manifest — resuming under changed settings would silently mix
                two experiments into one artifact.
        """
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        canonical = json.loads(json.dumps(manifest))
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing != canonical:
                differing = sorted(
                    key for key in {**existing, **canonical}
                    if existing.get(key) != canonical.get(key)
                )
                raise SystemExit(
                    f"{self.root} holds a run whose {', '.join(differing)} does not "
                    "match this command's. Resume with the original command, or "
                    "point --output-dir at a fresh directory."
                )
        else:
            self.manifest_path.write_text(
                json.dumps(canonical, indent=2) + "\n", encoding="utf-8"
            )

    def checkpoint_path(self, iteration: int) -> Path:
        return self.checkpoints / f"iteration_{iteration:03d}.pt"

    def _samples_path(self, iteration: int) -> Path:
        return self.data / f"iteration_{iteration:03d}.npz"

    def completed_iterations(self) -> int:
        count = 0
        while self.checkpoint_path(count).exists() and self._samples_path(count).exists():
            count += 1
        return count

    def save_iteration(
        self,
        row: IterationReport,
        samples: IterationSamples,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """The ``on_iteration`` hook: persist everything a resume needs."""
        images = (
            np.stack(samples.images) if samples.images else np.zeros((0,), dtype=np.uint8)
        )
        actions = (
            np.stack(samples.actions)
            if samples.actions
            else np.zeros((0, 3), dtype=np.float32)
        )
        np.savez_compressed(
            self._samples_path(row.iteration),
            images=images,
            commands=np.asarray(samples.commands, dtype=np.int64),
            actions=actions,
        )
        # Write-then-rename so a crash mid-save cannot leave a truncated file
        # that scans as a complete iteration.
        staging = self.checkpoint_path(row.iteration).with_suffix(".pt.tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "row": asdict(row),
            },
            staging,
        )
        staging.replace(self.checkpoint_path(row.iteration))

    def load_resume(self, model: nn.Module) -> DAggerResume:
        """Rebuild resume state from checkpoints, loading the latest weights
        into ``model``. Call only when :meth:`completed_iterations` is > 0."""
        count = self.completed_iterations()
        rows: list[IterationReport] = []
        samples = IterationSamples(images=[], commands=[], actions=[])
        payload: dict = {}
        for iteration in range(count):
            payload = torch.load(
                self.checkpoint_path(iteration), map_location="cpu", weights_only=True
            )
            rows.append(IterationReport(**payload["row"]))
            with np.load(self._samples_path(iteration)) as archive:
                samples.images.extend(np.asarray(frame) for frame in archive["images"])
                samples.commands.extend(int(c) for c in archive["commands"])
                samples.actions.extend(
                    np.asarray(action, dtype=np.float32) for action in archive["actions"]
                )
        model.load_state_dict(payload["model"])
        return DAggerResume(
            rows=rows, samples=samples, optimizer_state=payload["optimizer"]
        )

    def finish(self, report: dict) -> None:
        """Land the report and write-up, then drop the sample archives — once
        the report exists there is nothing to resume, and a lingering archive
        would suggest the run did not finish (the ablation's partial rule)."""
        # encoding="utf-8" is not optional: the write-up contains β and —, and
        # write_text without it uses cp1252 on Windows (see ablation.py).
        self.report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        writeup = self.report_path.with_suffix(".md")
        writeup.write_text(
            render_writeup(report, source=str(self.report_path)), encoding="utf-8"
        )
        for archive in sorted(self.data.glob("*.npz")):
            archive.unlink()


def _build_model(args: argparse.Namespace) -> nn.Module:
    """Build the student. A separate function so tests can substitute a tiny
    model at exactly the point the CLI constructs the real one."""
    from pathfinder.planning.cil_model import CILModel

    # Seeded before construction so two fresh runs of one command initialise
    # identical weights; run_dagger re-seeds per iteration afterwards.
    torch.manual_seed(args.seed)
    return CILModel(pretrained=args.pretrained, output_mode="control")


#: The CPU loop check (issue #20): small enough to finish in minutes on a
#: laptop, and offline — no pretrained-weight download. Explicit flags win
#: over these, so ``--smoke --iterations 3`` means three tiny iterations.
_SMOKE_OVERRIDES = {
    "iterations": 2,
    "episodes_per_iteration": 1,
    "route_length_m": 60.0,
    "max_steps": 40,
    "epochs_per_iteration": 1,
    "batch_size": 8,
    "pretrained": False,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DAgger loop and land its artifacts together: per-iteration "
            "weight checkpoints, a report JSON, and a generated write-up. "
            "Re-running the same command in the same --output-dir resumes a "
            "killed run from its last completed iteration."
        )
    )
    defaults = DAggerConfig()
    parser.add_argument("--backend", choices=["kinematic", "carla"], default="kinematic",
                        help="simulator backend; only carla can meet the real-run bar")
    parser.add_argument("--expert", default=None,
                        help="expert Policy registry name; defaults to pure_pursuit on "
                             "kinematic and carla_builtin_behavior_agent on carla")
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--episodes-per-iteration", type=int,
                        default=defaults.episodes_per_iteration)
    parser.add_argument("--route-length-m", type=float, default=defaults.route_length_m)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--epochs-per-iteration", type=int,
                        default=defaults.epochs_per_iteration)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--beta-decay", type=float, default=defaults.beta_decay)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-dataset-size", type=int, default=defaults.max_dataset_size)
    parser.add_argument("--device", default="cpu",
                        help="training and rollout device; pass cuda on the GPU box")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="ImageNet-initialise the backbone (downloads weights on first "
                             "use); required by the real-run bar")
    parser.add_argument("--smoke", action="store_true",
                        help="minutes-long CPU loop check; output is labelled smoke and "
                             "is never a result")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="run directory; defaults to results/dagger/<backend>[_smoke]")
    args = parser.parse_args(argv)
    if args.smoke:
        for name, value in _SMOKE_OVERRIDES.items():
            if getattr(args, name) == parser.get_default(name):
                setattr(args, name, value)

    config = DAggerConfig(
        iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        route_length_m=args.route_length_m,
        max_steps=args.max_steps,
        epochs_per_iteration=args.epochs_per_iteration,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_decay=args.beta_decay,
        device=args.device,
        seed=args.seed,
        max_dataset_size=args.max_dataset_size,
    )

    # Imported here rather than at module top for the same reason as in
    # ablation.py: the library-level loop has no business pulling these in.
    from pathfinder.policies import (
        DEFAULT_POLICY,
        POLICY_NAMES,
        CarlaBehaviorAgentPolicy,
        build_policy,
    )
    from pathfinder.sim.carla_backend import build_simulator

    expert_name = args.expert or (
        CarlaBehaviorAgentPolicy.NAME if args.backend == "carla" else DEFAULT_POLICY
    )
    if expert_name not in POLICY_NAMES:
        parser.error(f"unknown expert {expert_name!r}; expected one of {', '.join(POLICY_NAMES)}")

    output_dir = args.output_dir or (
        Path("results/dagger") / f"{args.backend}{'_smoke' if args.smoke else ''}"
    )
    run_dir = _RunDirectory(output_dir)
    if run_dir.report_path.exists():
        raise SystemExit(
            f"{run_dir.report_path} already exists — this run is complete. "
            "Point --output-dir at a fresh directory for a new run."
        )
    run_dir.prepare({
        "config": asdict(config),
        "backend": args.backend,
        "expert": expert_name,
        "pretrained": bool(args.pretrained),
    })

    model = _build_model(args)
    student = {
        "model": type(model).__name__,
        # "unknown" rather than a guessed default: the report records what the
        # model declares, never what a CLI assumed on its behalf.
        "output_mode": getattr(model, "output_mode", "unknown"),
        "imagenet_pretrained": bool(args.pretrained),
    }

    done = run_dir.completed_iterations()
    resume = None
    if done:
        resume = run_dir.load_resume(model)
        print(f"resuming: {done} of {config.iterations} iterations already checkpointed")

    if done < config.iterations:
        simulator_kwargs = {"render": True} if args.backend == "kinematic" else {}
        with build_simulator(args.backend, **simulator_kwargs) as simulator:
            expert = build_policy(expert_name, simulator=simulator)
            _, report = run_dagger(
                config,
                model=model,
                simulator=simulator,
                expert=expert,
                on_iteration=run_dir.save_iteration,
                resume_from=resume,
            )
        rows = report.iterations
    else:
        # Crashed between the final checkpoint and the report: nothing to
        # train, only the documents to land.
        rows = resume.rows

    # Derived from the run directory's own naming so a checkpoint-layout
    # change cannot silently break the recorded path.
    weights = (
        run_dir.checkpoint_path(config.iterations - 1).relative_to(run_dir.root).as_posix()
    )
    report_dict = build_run_report(
        backend=args.backend,
        expert_name=expert_name,
        student=student,
        config=config,
        rows=rows,
        weights=weights,
    )
    run_dir.finish(report_dict)

    print(f"backend: {report_dict['backend']} ({report_dict['scope']})")
    if report_dict["scope"] == "smoke":
        print("NOTE: smoke run — a loop check, never a result.")
    change = report_dict["error_reduction_pct"]
    print(
        f"iterations: {len(rows)}, final dataset {rows[-1].dataset_size}, "
        f"first-to-last disagreement change "
        f"{'not measurable' if change is None else f'{change}%'}"
    )
    print(f"weights: {output_dir / weights}")
    print(f"report written to {run_dir.report_path}")
    print(f"write-up written to {run_dir.report_path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
