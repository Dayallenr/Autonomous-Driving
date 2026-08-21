"""
The behaviour-agent reference run: record the ceiling before any training.

    python -m pathfinder.reference_run --backend carla

One command scores CARLA's built-in behaviour agent — **not project work** —
over the perception ablation's exact seeded suite, and writes one report
artifact plus a generated write-up. It exists because issue #16 needs the
reference ceiling *before* the student is trained, and the three-column
comparison CLI rightly refuses to run without student weights: a
reference-only measurement is its own kind of artifact, not a comparison
with two columns missing.

The artifact carries three things a reader must never have to reconstruct:

* **Whose driving this is.** ``not_project_work: true`` and the disclaimer
  sentence are baked into the JSON, and only a Policy declaring the behaviour
  agent's registry name can produce the report at all.
* **What the numbers mean.** Only the CARLA backend earns the
  driving-quality label; anything else is pipeline-only, in the artifact.
* **The go / stop-and-reassess gate.** Issue #16 gates the whole training
  plan on the agent's live score being meaningfully above the recorded
  PurePursuit floor. The gate is computed from the ablation artifact itself —
  never a hardcoded 25.2 — and only when the two reports are actually
  comparable: same seeded suite, both driving-quality. Otherwise the gate is
  recorded as not computed, with the reason.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pathfinder import reporting
from pathfinder.comparison import DEFAULT_SUITE
from pathfinder.metrics.driving_score import EpisodeScore, aggregate
from pathfinder.policies import CarlaBehaviorAgentPolicy
from pathfinder.reference_writeup import render_writeup
from pathfinder.runner import Policy, PurePursuitPolicy, run_episode
from pathfinder.sim.base import EpisodeSpec, SimulatorBackend

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_SUITE", "GATE_MARGIN_POINTS", "floor_gate", "run_reference"]

#: The sentence that travels inside the report JSON. The write-up carries its
#: own copy; this one exists so the disclaimer survives even when only the
#: JSON does.
NOT_PROJECT_WORK_NOTE = (
    "CARLA's own behaviour agent — not project work. It reads the simulator's "
    "world state directly and its score is a reference upper bound produced by "
    "the same routes, traffic, and scoring as the project's Policies; nothing "
    "here may be presented as this project's driving."
)

#: Pre-registered before any live run, like the ablation's fine-tuning
#: trigger: "meaningfully above the floor" means at least this many
#: driving-score points. The suite is bit-reproducible, so any positive
#: margin is real — but the gate is about teacher quality, not statistical
#: significance: a DAgger student imitates its teacher and rarely surpasses
#: it, so a teacher within ~10 points of the floor (well under the ablation's
#: own 16.34-point perception effect) would leave the whole training exercise
#: chasing a gain smaller than the effects already measured.
GATE_MARGIN_POINTS = 10.0

#: The rule, stated in the artifact so the verdict is checkable against it.
#: Issue #16 says only "meaningfully above" — the numeric threshold is this
#: project's operationalisation of that phrase, and the sentence must say so
#: rather than attribute a number to the issue that the issue never set.
_GATE_DEFINITION = (
    "Issue #16 gates the training plan on the behaviour agent's live score "
    "being 'meaningfully above' the recorded privileged-PurePursuit floor, "
    "without defining 'meaningfully'. This project operationalises it, "
    "pre-registered before any live run, as a margin of at least "
    f"{GATE_MARGIN_POINTS} driving-score points over the identical seeded "
    "suite; anything less is 'stop-and-reassess' — the teacher would be too "
    "close to the floor for imitating it to be worth a training run."
)


def _scope(backend_name: str) -> tuple[str, str]:
    """What the report's numbers are allowed to mean; the reference run's
    sentences over the shared rule that only CARLA earns the driving-quality
    label."""
    return reporting.scope(
        backend_name,
        driving_quality_note=(
            "Generated on the CARLA backend: the score measures the behaviour "
            "agent's driving quality over the recorded suite. It is a "
            "reference ceiling, not project work."
        ),
        pipeline_limits=(
            "cannot run the real behaviour agent and neither simulates real "
            "physics nor renders real scenes"
        ),
        pipeline_label="reference-run pipeline",
    )


def floor_gate(
    reference_report: dict, floor_report: dict | None, *, floor_source: str
) -> dict:
    """Issue #16's go / stop-and-reassess gate, computed from artifacts.

    Args:
        reference_report: A ``run_reference`` report dict.
        floor_report: The perception ablation's report dict, whose privileged
            arm is the PurePursuit floor — or ``None`` when no artifact was
            found.
        floor_source: Where the floor report came from, recorded in the gate
            so the floor number stays traceable.

    Returns:
        ``{"computed": False, "reason": ...}`` when the two reports are not
        comparable — missing floor, wrong kind or controller, either side
        pipeline-only, or different suites. Otherwise the gate with both
        scores, the margin, the pre-registered rule, and the verdict.
    """

    def not_computed(reason: str) -> dict:
        return {"computed": False, "reason": reason}

    if floor_report is None:
        return not_computed(f"no floor report found at {floor_source or '<none>'}")
    if reference_report["scope"] != reporting.SCOPE_DRIVING_QUALITY:
        return not_computed(
            f"this run is {reference_report['scope']}, and only a "
            "driving-quality run can be gated against the CARLA floor"
        )
    if floor_report.get("kind") != "perception_ablation":
        return not_computed(
            f"{floor_source} is not a perception_ablation report "
            f"(kind={floor_report.get('kind')!r})"
        )
    if floor_report.get("scope") != reporting.SCOPE_DRIVING_QUALITY:
        return not_computed(
            f"the floor report at {floor_source} is "
            f"{floor_report.get('scope')!r}, not driving-quality"
        )
    if floor_report.get("controller") != PurePursuitPolicy.NAME:
        return not_computed(
            f"the floor report's controller is "
            f"{floor_report.get('controller')!r}, not {PurePursuitPolicy.NAME!r} "
            "— the gate is defined against the PurePursuit floor"
        )
    if reference_report["episodes"] != floor_report.get("episodes"):
        return not_computed(
            "this run's suite differs from the floor report's; the gate is "
            "only meaningful over the identical seeded suite"
        )

    floor_score = floor_report["baseline"]["summary"]["driving_score"]
    reference_score = reference_report["summary"]["driving_score"]
    margin = round(reference_score - floor_score, 2)
    gate = {
        "computed": True,
        "definition": _GATE_DEFINITION,
        "floor_source": floor_source,
        "floor_policy": (
            f"{floor_report['controller']} under "
            f"{floor_report['baseline']['perception']} perception"
        ),
        "floor_driving_score": floor_score,
        "reference_driving_score": reference_score,
        "margin": margin,
        "required_margin": GATE_MARGIN_POINTS,
        "verdict": "go" if margin >= GATE_MARGIN_POINTS else "stop-and-reassess",
    }
    failed = reference_report["failed_episodes"]
    if failed:
        gate["caveat"] = (
            f"episode(s) {', '.join(failed)} failed mid-run; their scores "
            "measure the failure, not the agent, so this aggregate understates "
            "the ceiling and the verdict needs a human read"
        )
    return gate


def run_reference(
    simulator: SimulatorBackend,
    specs: list[EpisodeSpec],
    policy: Policy,
    *,
    floor_report: dict | None = None,
    floor_source: str = "",
    on_episode: Callable[[str, EpisodeScore], None] | None = None,
) -> dict:
    """Score the behaviour agent over the suite and report the ceiling.

    Args:
        simulator: Backend the agent drives in. Only the CARLA backend earns
            the driving-quality scope label.
        specs: The suite, normally the perception ablation's exact seeded
            suite so the ceiling and the floor are comparable.
        policy: Must declare the behaviour agent's registry name — this
            artifact's entire identity is "CARLA's agent, not project work",
            so nothing else may produce it.
        floor_report: The perception ablation report to gate against; see
            :func:`floor_gate`. ``None`` records the gate as not computed.
        floor_source: Where ``floor_report`` came from, for traceability.
        on_episode: Called with ``(model_version, score)`` after each Episode,
            in run order, so the CLI can persist partial results.

    Returns:
        The report as a JSON-serialisable dict: the suite, per-episode scores,
        the aggregate, the not-project-work label, the scope label, and the
        floor gate.

    Raises:
        ValueError: If ``specs`` is empty, or if ``policy`` does not declare
            the behaviour agent's name.
    """
    if not specs:
        raise ValueError("cannot run a reference baseline with no episodes")
    declared = getattr(policy, "NAME", None)
    if declared != CarlaBehaviorAgentPolicy.NAME:
        raise ValueError(
            f"the reference baseline is {CarlaBehaviorAgentPolicy.NAME!r} but the "
            f"Policy declares {declared!r}; this artifact records CARLA's own "
            "agent and nothing else may produce it"
        )

    logger.info(
        "reference baseline: %d episodes on %s", len(specs), simulator.name
    )

    scores: list[EpisodeScore] = []
    latencies: list[float] = []

    def sink(row: dict) -> None:
        latencies.append(row["policy_latency_ms"])

    for spec in specs:
        score = run_episode(
            simulator,
            spec,
            policy,
            telemetry_sink=sink,
            model_version=CarlaBehaviorAgentPolicy.NAME,
        )
        scores.append(score)
        if on_episode is not None:
            on_episode(CarlaBehaviorAgentPolicy.NAME, score)

    # A failed Episode scored the failure, not the agent. Flagged rather than
    # dropped — silently excluding episodes is its own kind of lie about what
    # the suite covered (the ablation's rule).
    failed_episodes = sorted(
        score.episode_id for score in scores if score.status != "completed"
    )
    if failed_episodes:
        logger.warning(
            "%d episode(s) failed mid-run: %s",
            len(failed_episodes),
            ", ".join(failed_episodes),
        )

    scope, scope_note = _scope(simulator.name)
    seed_by_id = {spec.episode_id: spec.seed for spec in specs}
    report = {
        "kind": "reference_baseline",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": simulator.name,
        "scope": scope,
        "scope_note": scope_note,
        "policy": CarlaBehaviorAgentPolicy.NAME,
        "model_version": CarlaBehaviorAgentPolicy.NAME,
        "not_project_work": True,
        "not_project_work_note": NOT_PROJECT_WORK_NOTE,
        "episodes": [spec.to_dict() for spec in specs],
        "summary": aggregate(scores).to_dict(),
        "mean_policy_latency_ms": round(
            statistics.fmean(latencies) if latencies else 0.0, 3
        ),
        "results": [
            {"seed": seed_by_id[score.episode_id], **score.to_dict()}
            for score in scores
        ],
        "failed_episodes": failed_episodes,
    }
    behavior = getattr(policy, "behavior", "")
    if behavior:
        report["behavior"] = behavior
    report["floor_gate"] = floor_gate(report, floor_report, floor_source=floor_source)
    return report


def _build_reference_policy(args: argparse.Namespace, simulator: SimulatorBackend) -> Policy:
    """Build the real behaviour agent. A separate function so tests can
    substitute a stub at exactly the point the CLI constructs the real thing."""
    from pathfinder.policies import build_policy

    return build_policy(
        CarlaBehaviorAgentPolicy.NAME, simulator=simulator, behavior=args.behavior
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score CARLA's built-in behaviour agent (not project work) over "
            "the perception ablation's exact seeded suite."
        )
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_SUITE["count"], help="episodes to run"
    )
    parser.add_argument(
        "--backend",
        choices=["kinematic", "carla"],
        default="carla",
        help="simulator backend; the behaviour agent only exists on carla — "
        "kinematic is for pipeline tests with a stubbed Policy",
    )
    parser.add_argument(
        "--behavior",
        choices=["cautious", "normal", "aggressive"],
        default="normal",
        help="CARLA behaviour preset, recorded in the artifact",
    )
    parser.add_argument(
        "--route-length-m", type=float, default=DEFAULT_SUITE["route_length_m"]
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_SUITE["max_steps"])
    parser.add_argument("--base-seed", type=int, default=DEFAULT_SUITE["base_seed"])
    parser.add_argument(
        "--floor-report",
        type=Path,
        default=Path("results/ablation/carla_report.json"),
        help="the perception ablation artifact whose privileged arm is the "
        "PurePursuit floor the go/stop gate reads",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report path; defaults to results/reference/<backend>_report.json",
    )
    args = parser.parse_args(argv)

    # Imported here rather than at module top for the same reason as in
    # ablation.py: orchestration pulls in the cloud stack, which the
    # library-level run_reference has no business needing.
    from pathfinder.orchestration import build_episode_specs
    from pathfinder.sim.carla_backend import build_simulator

    specs = build_episode_specs(
        count=args.episodes,
        route_length_m=args.route_length_m,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
    )
    floor_report = (
        json.loads(args.floor_report.read_text(encoding="utf-8"))
        if args.floor_report.exists()
        else None
    )
    # Said before the run, not discovered after it: an overridden suite flag
    # silently forfeits the gate, and an unattended sitting must not spend
    # hours producing an artifact that cannot answer the ticket's question.
    if floor_report is not None and [
        spec.to_dict() for spec in specs
    ] != floor_report.get("episodes"):
        print(
            "WARNING: this suite differs from the floor report's, so the "
            "go / stop-and-reassess gate will not be computed. Run with the "
            "default suite flags to gate against the floor."
        )
    # render=True on kinematic keeps the pipeline path identical to the other
    # CLIs'; the CARLA backend renders by default.
    simulator_kwargs = {"render": True} if args.backend == "kinematic" else {}

    output = args.output or Path("results/reference") / f"{args.backend}_report.json"
    artifact = reporting.ReportArtifact(output, label_key="model_version")

    with build_simulator(args.backend, **simulator_kwargs) as simulator:
        try:
            policy = _build_reference_policy(args, simulator)
        except ValueError as error:
            # Eager and loud: without CARLA there is no behaviour agent to
            # run, and a reference artifact with nothing in it helps nobody.
            raise SystemExit(str(error)) from error
        report = run_reference(
            simulator,
            specs,
            policy,
            floor_report=floor_report,
            floor_source=str(args.floor_report),
            on_episode=artifact.checkpoint,
        )

    writeup = artifact.finish(report, render_writeup)

    print(f"backend: {report['backend']} ({report['scope']})")
    print(
        f"{report['model_version']} (not project work): "
        f"driving score {report['summary']['driving_score']}"
    )
    gate = report["floor_gate"]
    if gate["computed"]:
        print(
            f"floor gate: {gate['verdict']} — reference "
            f"{gate['reference_driving_score']} vs floor "
            f"{gate['floor_driving_score']} (margin {gate['margin']}, "
            f"required {gate['required_margin']})"
        )
    else:
        print(f"floor gate: not computed — {gate['reason']}")
    reporting.print_cli_tail(report, output, writeup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
