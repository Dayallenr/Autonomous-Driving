"""
DAgger: Dataset Aggregation for the conditional imitation-learning planner.

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

The expert here is :class:`PurePursuitPlanner`, which is *privileged*: it reads
exact cross-track error, heading error, and curvature from the simulator. The
student sees only rendered pixels. That asymmetry is the point — the student
learns to infer from images what the expert is handed directly, which is exactly
the privileged-expert setup used by LBC and Roach.

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

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from pathfinder.runner import ControlOutput, PurePursuitPlanner
from pathfinder.sim.base import EpisodeSpec, FrameState
from pathfinder.sim.kinematic import KinematicSimulator

logger = logging.getLogger(__name__)

__all__ = ["CILPlanner", "DAggerConfig", "DAggerReport", "IterationReport", "run_dagger"]

# ImageNet statistics: the ResNet-18 backbone is pretrained with these, and
# feeding it differently-normalized input wastes most of the transfer.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: np.ndarray) -> torch.Tensor:
    """Convert a rendered HWC uint8 frame to a normalized CHW float tensor."""
    array = image.astype(np.float32) / 255.0
    array = (array - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1))


class CILPlanner:
    """Wraps a CIL model so it satisfies the :class:`Planner` protocol.

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
                "CILPlanner requires rendered frames; construct the simulator "
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
            "disagreement": round(self.disagreement, 5),
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


def _expert_control(planner: PurePursuitPlanner, state: FrameState) -> np.ndarray:
    control = planner.plan(state)
    return np.array([control.steer, control.throttle, control.brake], dtype=np.float32)


def _collect(
    simulator: KinematicSimulator,
    expert: PurePursuitPlanner,
    student: CILPlanner | None,
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
    progress: bool = True,
) -> tuple[nn.Module, DAggerReport]:
    """Run the DAgger loop and return the trained student plus its report.

    Raises:
        ValueError: If ``config.iterations`` is below 2 — error *reduction* is
            undefined without at least a first and last measurement.
    """
    config = config or DAggerConfig()
    if config.iterations < 2:
        raise ValueError(
            f"need at least 2 iterations to measure error reduction, got {config.iterations}"
        )

    from planning.cil_model import CILModel

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    if model is None:
        # pretrained=False keeps the run offline and reproducible; a real run
        # should use ImageNet initialisation.
        model = CILModel(pretrained=False, output_mode="control")
    model = model.to(config.device)

    expert = PurePursuitPlanner()
    simulator = KinematicSimulator(render=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.L1Loss()

    aggregate_images: list[np.ndarray] = []
    aggregate_commands: list[int] = []
    aggregate_actions: list[np.ndarray] = []

    report = DAggerReport()
    run_started = time.perf_counter()

    try:
        for iteration in range(config.iterations):
            iteration_started = time.perf_counter()
            # Iteration 0 is pure expert: an untrained student's rollouts are
            # noise, and aggregating them first would poison the dataset.
            beta = 1.0 if iteration == 0 else config.beta_decay**iteration
            student = None if iteration == 0 else CILPlanner(model, device=config.device)

            episode_disagreements: list[float] = []
            completions: list[float] = []
            for episode in range(config.episodes_per_iteration):
                spec = EpisodeSpec(
                    episode_id=f"dagger-{iteration}-{episode}",
                    route_length_m=config.route_length_m,
                    max_steps=config.max_steps,
                    seed=config.seed + iteration * 100 + episode,
                )
                images, commands, actions, disagreements, completion = _collect(
                    simulator, expert, student, spec, beta, rng
                )
                aggregate_images.extend(images)
                aggregate_commands.extend(commands)
                aggregate_actions.extend(actions)
                episode_disagreements.extend(disagreements)
                completions.append(completion)

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
            report.iterations.append(
                IterationReport(
                    iteration=iteration,
                    beta=beta,
                    dataset_size=len(aggregate_images),
                    train_loss=train_loss,
                    disagreement=disagreement,
                    driving_score=completion * 100.0,
                    route_completion=completion,
                    seconds=time.perf_counter() - iteration_started,
                )
            )
            if progress:
                logger.info(
                    "DAgger iter %d: beta=%.2f n=%d loss=%.4f disagreement=%s completion=%.3f",
                    iteration, beta, len(aggregate_images), train_loss,
                    "n/a" if np.isnan(disagreement) else f"{disagreement:.4f}",
                    completion,
                )
    finally:
        simulator.close()

    # Error reduction compares the first *measured* disagreement (iteration 1,
    # since iteration 0 has no student) to the last.
    measured = [
        item.disagreement for item in report.iterations if not np.isnan(item.disagreement)
    ]
    if len(measured) >= 2 and measured[0] > 0:
        report.error_reduction = (measured[0] - measured[-1]) / measured[0]
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
