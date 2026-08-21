"""
The three-column comparison: floor, student, reference ceiling.

    python -m pathfinder.comparison --weights results/dagger/carla/checkpoints/iteration_009.pt
    python -m pathfinder.comparison --backend carla --weights ... --device cuda

One command scores three Policies over the perception ablation's exact seeded
suite — same episode specs, same seeds, same scoring — and writes one report
artifact plus a generated write-up:

* **floor** — ``pure_pursuit``, the project's geometric controller, reading
  privileged state. The ablation measured it as the binding constraint on
  driving score, so it is a *weak* floor and the write-up says so plainly.
* **student** — ``cil_student``, the trained CIL model, driving from rendered
  pixels plus the privileged route command: a strictly harder observation than
  either baseline's.
* **reference ceiling** — ``carla_builtin_behavior_agent``, CARLA's own
  behaviour agent. **Not project work**; it appears as an upper bound produced
  by the same routes and scoring, and the artifact labels it so.

Attribution is enforced, not hoped for: an arm cannot claim a Policy whose
declared ``NAME`` differs from the arm's, nor a weights version its Policy
does not carry, and two arms cannot share a result label. The student's
``model_version`` is derived from the checkpoint bytes and stamped into every
telemetry frame and report row, so a score can never be read as evidence
about weights that did not produce it.

The reference ceiling needs the CARLA backend (it drives the simulator's own
ego actor). On any other backend the CLI records that column as skipped, with
the reason in the artifact — a two-column table silently posing as the
comparison would be its own kind of lie.

Scope labelling mirrors the ablation's: only the CARLA backend produces
driving-quality numbers; a kinematic run verifies the mechanism and is
labelled pipeline-only in the artifact itself.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pathfinder.comparison_writeup import render_writeup
from pathfinder.metrics.driving_score import EpisodeScore, aggregate
from pathfinder.policies import CarlaBehaviorAgentPolicy
from pathfinder.runner import Policy, run_episode
from pathfinder.sim.base import EpisodeSpec, SimulatorBackend

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SUITE",
    "ROLE_FLOOR",
    "ROLE_REFERENCE",
    "ROLE_STUDENT",
    "STUDENT_OBSERVATION_BOUNDARY",
    "ComparisonArm",
    "run_comparison",
]

ROLE_FLOOR = "floor"
ROLE_STUDENT = "student"
ROLE_REFERENCE = "reference_ceiling"

#: The columns, in the order every report and write-up presents them.
_ROLES = (ROLE_FLOOR, ROLE_STUDENT, ROLE_REFERENCE)

#: What the student is allowed to observe. Stamped into every report because
#: the comparison is only meaningful under it: the baselines read privileged
#: state, so the student is solving a strictly harder task, and a table that
#: omits this reads as three Policies playing the same game.
STUDENT_OBSERVATION_BOUNDARY = (
    "rendered pixels plus the privileged route command; the student is never "
    "shown route errors, obstacle ranges, or traffic-light state"
)

#: The perception ablation's exact seeded suite (``build_episode_specs``
#: arguments). Shared so the comparison's numbers and the ablation's are
#: comparable; ``tests/test_comparison.py`` pins this against the recorded
#: CARLA ablation artifact.
DEFAULT_SUITE = {
    "count": 10,
    "route_length_m": 400.0,
    "base_seed": 1000,
    "max_steps": 1500,
}


def _scope(backend_name: str) -> tuple[str, str]:
    """What the report's numbers are allowed to mean. Mirrors the ablation:
    only CARLA earns the driving-quality label, and an unknown backend
    defaults to pipeline-only because the safe failure mode is underclaiming."""
    if backend_name == "carla":
        return (
            "driving-quality",
            "Generated on the CARLA backend: each column's score measures that "
            "Policy's driving quality under its stated observation boundary.",
        )
    return (
        "pipeline-only",
        f"Generated on the {backend_name} backend, which neither simulates real "
        "physics nor renders real scenes. These numbers verify the comparison "
        "pipeline end to end; they are not driving quality and must never be "
        "quoted as such. The real measurement comes from the CARLA backend.",
    )


@dataclass(frozen=True)
class ComparisonArm:
    """One column of the comparison.

    ``policy=None`` marks an arm that cannot run on this backend; only the
    reference ceiling may be skipped, and only with a ``skip_reason`` — the
    reason travels into the artifact so a missing column is a recorded fact,
    never an omission a reader has to notice.
    """

    role: str
    #: Registry name of the Policy this column claims to score.
    policy_name: str
    #: The label recorded in every result and telemetry frame; the student's
    #: carries its weights version.
    model_version: str
    policy: Policy | None = None
    skip_reason: str = ""
    #: Stated weights location, student only; recorded in the artifact.
    weights: str = ""


def _validate(specs: list[EpisodeSpec], arms: list[ComparisonArm]) -> None:
    if not specs:
        raise ValueError("cannot run a comparison with no episodes")
    roles = [arm.role for arm in arms]
    if sorted(roles) != sorted(_ROLES):
        raise ValueError(
            f"the comparison needs each role exactly once ({', '.join(_ROLES)}); "
            f"got {roles}"
        )
    labels = [arm.model_version for arm in arms]
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"two arms share a result label: {labels}; every column's scores "
            "must be attributable to exactly one Policy"
        )
    for arm in arms:
        if arm.policy is None:
            if arm.role != ROLE_REFERENCE:
                raise ValueError(
                    f"the {arm.role} arm cannot be skipped; only the "
                    f"{ROLE_REFERENCE} arm may be, and only because it needs "
                    "the CARLA backend"
                )
            if not arm.skip_reason:
                raise ValueError(
                    "a skipped arm needs a recorded reason; an unexplained "
                    "missing column is an omission, not a fact"
                )
            continue
        declared = getattr(arm.policy, "NAME", arm.policy_name)
        if declared != arm.policy_name:
            raise ValueError(
                f"the {arm.role} arm claims policy {arm.policy_name!r} but its "
                f"Policy declares {declared!r}; a mislabelled column poisons "
                "the whole comparison"
            )
        held_version = getattr(arm.policy, "model_version", arm.model_version)
        if held_version != arm.model_version:
            raise ValueError(
                f"the {arm.role} arm claims weights version "
                f"{arm.model_version!r} but its Policy carries "
                f"{held_version!r}; results must be attributable to the "
                "weights that produced them"
            )


def _run_arm(
    simulator: SimulatorBackend,
    specs: list[EpisodeSpec],
    arm: ComparisonArm,
    on_episode: Callable[[str, EpisodeScore], None] | None,
) -> dict:
    scores: list[EpisodeScore] = []
    latencies: list[float] = []

    def sink(row: dict) -> None:
        latencies.append(row["policy_latency_ms"])

    for spec in specs:
        score = run_episode(
            simulator,
            spec,
            arm.policy,
            telemetry_sink=sink,
            model_version=arm.model_version,
        )
        scores.append(score)
        if on_episode is not None:
            on_episode(arm.model_version, score)

    seed_by_id = {spec.episode_id: spec.seed for spec in specs}
    return {
        **_arm_identity(arm),
        "summary": aggregate(scores).to_dict(),
        "mean_policy_latency_ms": round(
            statistics.fmean(latencies) if latencies else 0.0, 3
        ),
        "episodes": [
            {"seed": seed_by_id[score.episode_id], **score.to_dict()}
            for score in scores
        ],
    }


def _arm_identity(arm: ComparisonArm) -> dict:
    identity = {
        "role": arm.role,
        "policy": arm.policy_name,
        "model_version": arm.model_version,
        # Baked into the artifact, not left to the write-up: the behaviour
        # agent's column must carry its disclaimer even when only the JSON
        # survives.
        "not_project_work": arm.policy_name == CarlaBehaviorAgentPolicy.NAME,
    }
    if arm.weights:
        identity["weights"] = arm.weights
    return identity


def run_comparison(
    simulator: SimulatorBackend,
    specs: list[EpisodeSpec],
    arms: list[ComparisonArm],
    *,
    on_episode: Callable[[str, EpisodeScore], None] | None = None,
) -> dict:
    """Score every arm over the same specs and report the three columns.

    Args:
        simulator: Backend all arms drive in, sequentially. Reuse is safe
            because ``reset(spec)`` reseeds everything from the spec.
        specs: The suite. Every runnable arm drives exactly this list, in this
            order, so the columns differ only in Policy.
        arms: The three columns; see :class:`ComparisonArm`. Order is
            normalised to floor, student, reference in the report.
        on_episode: Called with ``(model_version, score)`` after each Episode,
            in run order, so a caller can persist partial results — a
            three-arm CARLA suite is hours of simulation.

    Returns:
        The report as a JSON-serialisable dict: per-arm scores and aggregates,
        each arm's identity and result label, the student's observation
        boundary, failed-episode flags, and a scope label saying whether the
        numbers are driving quality at all.

    Raises:
        ValueError: If the suite is empty; if the roles are not exactly floor,
            student, and reference ceiling; if an arm other than the reference
            is skipped, or skipped without a reason; if an arm's Policy
            declares a different name or weights version than the arm claims;
            or if two arms share a result label.
    """
    _validate(specs, arms)
    by_role = {arm.role: arm for arm in arms}
    ordered = [by_role[role] for role in _ROLES]

    logger.info(
        "comparison: %d episodes on %s, columns %s",
        len(specs),
        simulator.name,
        ", ".join(arm.model_version for arm in ordered),
    )

    arm_reports: list[dict] = []
    for arm in ordered:
        if arm.policy is None:
            logger.warning(
                "%s arm skipped: %s", arm.role, arm.skip_reason
            )
            arm_reports.append(
                {**_arm_identity(arm), "skipped": True, "skip_reason": arm.skip_reason}
            )
            continue
        arm_reports.append(_run_arm(simulator, specs, arm, on_episode))

    # An Episode that failed mid-run scored the failure, not the Policy.
    # Flagged rather than dropped — silently excluding episodes is its own
    # kind of lie about what the suite covered (the ablation's rule).
    failed_episodes = sorted(
        {
            row["episode_id"]
            for arm in arm_reports
            if not arm.get("skipped")
            for row in arm["episodes"]
            if row["status"] != "completed"
        }
    )
    if failed_episodes:
        logger.warning(
            "%d episode(s) failed mid-run in at least one arm: %s",
            len(failed_episodes),
            ", ".join(failed_episodes),
        )

    scope, scope_note = _scope(simulator.name)
    return {
        "kind": "policy_comparison",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": simulator.name,
        "scope": scope,
        "scope_note": scope_note,
        "student_observation_boundary": STUDENT_OBSERVATION_BOUNDARY,
        "episodes": [spec.to_dict() for spec in specs],
        "arms": arm_reports,
        "failed_episodes": failed_episodes,
    }


def _build_arms(args: argparse.Namespace, simulator: SimulatorBackend) -> list[ComparisonArm]:
    """Build the three columns from the registry. A separate function so tests
    can substitute stubs at exactly the point the CLI constructs the real
    Policies."""
    from pathfinder.policies import PURE_PURSUIT, CILStudentPolicy, build_policy

    student = build_policy(
        CILStudentPolicy.NAME, weights=args.weights, device=args.device
    )
    try:
        reference = ComparisonArm(
            role=ROLE_REFERENCE,
            policy_name=CarlaBehaviorAgentPolicy.NAME,
            model_version=CarlaBehaviorAgentPolicy.NAME,
            policy=build_policy(CarlaBehaviorAgentPolicy.NAME, simulator=simulator),
        )
    except ValueError as error:
        # The behaviour agent drives the simulator's own ego actor, which only
        # the CARLA backend has. The column stays in the report, marked with
        # exactly why it did not run.
        reference = ComparisonArm(
            role=ROLE_REFERENCE,
            policy_name=CarlaBehaviorAgentPolicy.NAME,
            model_version=CarlaBehaviorAgentPolicy.NAME,
            policy=None,
            skip_reason=str(error),
        )
    return [
        ComparisonArm(
            role=ROLE_FLOOR,
            policy_name=PURE_PURSUIT,
            model_version=PURE_PURSUIT,
            policy=build_policy(PURE_PURSUIT),
        ),
        ComparisonArm(
            role=ROLE_STUDENT,
            policy_name=CILStudentPolicy.NAME,
            model_version=student.model_version,
            policy=student,
            weights=str(args.weights),
        ),
        reference,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score floor, student, and reference ceiling over the perception "
            "ablation's exact seeded suite."
        )
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_SUITE["count"], help="episodes per arm"
    )
    parser.add_argument(
        "--backend",
        choices=["kinematic", "carla"],
        default="kinematic",
        help="simulator backend; only carla produces driving-quality numbers "
        "and only carla can run the reference ceiling",
    )
    parser.add_argument(
        "--route-length-m", type=float, default=DEFAULT_SUITE["route_length_m"]
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_SUITE["max_steps"])
    parser.add_argument("--base-seed", type=int, default=DEFAULT_SUITE["base_seed"])
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="the student's checkpoint; the comparison refuses to run without one",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="student inference device; cpu is deterministic everywhere, pass "
        "cuda on the GPU box",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report path; defaults to results/comparison/<backend>_report.json",
    )
    args = parser.parse_args(argv)

    # Imported here rather than at module top for the same reason as in
    # ablation.py: orchestration pulls in the cloud stack, which the
    # library-level run_comparison has no business needing.
    from pathfinder.orchestration import build_episode_specs
    from pathfinder.sim.carla_backend import build_simulator

    specs = build_episode_specs(
        count=args.episodes,
        route_length_m=args.route_length_m,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
    )
    # render=True on kinematic so the student has pixels to drive from; the
    # CARLA backend renders by default.
    simulator_kwargs = {"render": True} if args.backend == "kinematic" else {}

    output = args.output or Path("results/comparison") / f"{args.backend}_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Every finished Episode lands here immediately, so a crash late in a
    # three-arm CARLA suite costs one episode, not hours. Removed once the
    # full report exists — a lingering partial would mean the run did not
    # finish (the ablation's rule).
    partial = output.with_suffix(".partial.jsonl")
    partial.unlink(missing_ok=True)

    def checkpoint(model_version: str, score: EpisodeScore) -> None:
        with partial.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"model_version": model_version, **score.to_dict()}) + "\n"
            )

    with build_simulator(args.backend, **simulator_kwargs) as simulator:
        arms = _build_arms(args, simulator)
        report = run_comparison(simulator, specs, arms, on_episode=checkpoint)

    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)

    # The write-up lands with the report so a CARLA sitting can never end with
    # numbers but no document stating what they are allowed to mean.
    writeup = output.with_suffix(".md")
    # encoding="utf-8" is not optional: write_text defaults to the locale
    # encoding, which is cp1252 on Windows, and the rendered write-up contains
    # non-ASCII characters (see ablation.py).
    writeup.write_text(render_writeup(report, source=str(output)), encoding="utf-8")

    print(f"backend: {report['backend']} ({report['scope']})")
    for arm in report["arms"]:
        if arm.get("skipped"):
            print(f"{arm['role']:>17} ({arm['model_version']}): "
                  f"not run — {arm['skip_reason']}")
        else:
            print(f"{arm['role']:>17} ({arm['model_version']}): "
                  f"driving score {arm['summary']['driving_score']}")
    if report["scope"] == "pipeline-only":
        print("NOTE: pipeline-only run — these numbers are not driving quality.")
    print(f"report written to {output}")
    print(f"write-up written to {writeup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
