"""
The perception ablation: two runs that differ in exactly one respect.

    python -m pathfinder.ablation --episodes 10 --backend kinematic
    python -m pathfinder.ablation --episodes 50 --backend carla --device cuda

One command runs the same seeded Episodes twice — once with privileged
perception (the simulator's ground truth, the upper bound) and once with the
candidate (normally the trained Detector reading the camera) — and reports
both aggregate driving scores and their difference. Because seed, route,
traffic, weather, controller, and scoring are all shared, the gap between the
two scores is a direct measurement of what imperfect perception costs, in
driving-score units, rather than an inference from two unrelated pipelines.

What the report can and cannot claim
------------------------------------
The report labels its own scope. On the CARLA backend the numbers are driving
quality; on the kinematic backend they are **pipeline-only** — the backend
neither simulates real physics nor renders real scenes, so a kinematic run
verifies the mechanism and nothing else. The label is written into the
artifact because a mislabelled number in a results file outlives any warning
in a docstring.

Provenance is observed, not assumed: each Episode's perception identity is
read back from the telemetry frames that actually ran (stamped per frame by
``ModularPolicy`` from the component that produced them), so the report cannot
attribute a result to a perception that did not run.
"""
from __future__ import annotations

import argparse
import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pathfinder import reporting
from pathfinder.ablation_writeup import render_writeup
from pathfinder.metrics.driving_score import EpisodeScore, aggregate
from pathfinder.perception.base import Perception, perception_name
from pathfinder.perception.detector import TRAINED_WEIGHTS
from pathfinder.perception.geometry import KINEMATIC_CAMERA
from pathfinder.perception.privileged import PrivilegedPerception
from pathfinder.policies import ModularPolicy
from pathfinder.runner import Policy, PurePursuitPolicy, run_episode
from pathfinder.sim.base import EpisodeSpec, SimulatorBackend
from pathfinder.sim.carla_backend import CARLA_CAMERA

logger = logging.getLogger(__name__)

__all__ = ["PERCEPTION_BOUNDARY", "run_ablation"]

#: The sentence every write-up of an ablation score must carry. The Detector
#: has no traffic-light class and localization is a map problem, so claiming
#: more than this would be false; see ``pathfinder/perception/base.py``.
PERCEPTION_BOUNDARY = "perceived obstacles, privileged localization and traffic lights"


def _scope(backend_name: str) -> tuple[str, str]:
    """What the report's numbers are allowed to mean; the ablation's sentences
    over the shared rule that only CARLA earns the driving-quality label."""
    return reporting.scope(
        backend_name,
        driving_quality_note=(
            "Generated on the CARLA backend: scores measure driving quality "
            f"under the stated perception boundary ({PERCEPTION_BOUNDARY})."
        ),
        pipeline_limits="neither simulates real physics nor renders real scenes",
        pipeline_label="ablation pipeline",
    )


#: Which camera geometry each backend's frames were rendered with. Used to
#: fail fast when a Perception that derives ranges from pixels is paired with
#: a backend whose camera it was not configured for — the mismatch produces
#: silently mis-scaled ranges, not an error, so it must be caught up front.
_BACKEND_CAMERAS = {"kinematic": KINEMATIC_CAMERA, "carla": CARLA_CAMERA}


def _controller_label(controller: Policy) -> str:
    """What telemetry records as ``model_version``. Prefers the controller's
    registry name (``NAME``) so ablation rows group with rows from every other
    entry point instead of splitting one controller into two names."""
    return getattr(controller, "NAME", type(controller).__name__)


@dataclass(frozen=True)
class _ArmResult:
    """One perception's pass over the suite, with observed provenance."""

    perception: str
    controller: str
    scores: list[EpisodeScore]
    #: episode_id -> the perception name observed on that Episode's frames.
    provenance: dict[str, str]
    mean_perception_latency_ms: float


def _run_arm(
    simulator: SimulatorBackend,
    specs: list[EpisodeSpec],
    perception: Perception,
    controller_factory: Callable[[], Policy],
    on_episode: Callable[[str, EpisodeScore], None] | None = None,
) -> _ArmResult:
    scores: list[EpisodeScore] = []
    provenance: dict[str, str] = {}
    latencies: list[float] = []
    controller_name = ""

    for spec in specs:
        seen: set[str] = set()

        def sink(row: dict, seen: set[str] = seen) -> None:
            seen.add(row["perception"])
            latencies.append(row["perception_latency_ms"])

        controller = controller_factory()
        controller_name = controller_name or _controller_label(controller)
        score = run_episode(
            simulator,
            spec,
            ModularPolicy(perception, controller),
            telemetry_sink=sink,
            model_version=controller_name,
        )
        if len(seen) > 1:
            # Mixed provenance within one Episode is attribution corruption,
            # and a misattributable Episode poisons the whole comparison.
            raise RuntimeError(
                f"episode {spec.episode_id} has mixed perception provenance: {sorted(seen)}"
            )
        if not seen:
            if score.status != "failed":
                raise RuntimeError(
                    f"episode {spec.episode_id} completed without emitting a single "
                    "telemetry frame; its perception cannot be attributed"
                )
            # Failed before its first full frame: there is nothing to
            # attribute, and the failure flag already marks the Episode as
            # unusable for the difference — one crash must not abort the arm.
            provenance[spec.episode_id] = "unobserved"
        else:
            provenance[spec.episode_id] = seen.pop()
        scores.append(score)
        if on_episode is not None:
            on_episode(provenance[spec.episode_id], score)

    observed = {name for name in provenance.values() if name != "unobserved"}
    if len(observed) > 1:
        raise RuntimeError(f"one arm observed multiple perceptions: {sorted(observed)}")
    if not observed:
        raise RuntimeError(
            "no episode in the arm produced an attributable frame; "
            "the arm's results cannot be assigned to any perception"
        )

    return _ArmResult(
        perception=observed.pop(),
        controller=controller_name,
        scores=scores,
        provenance=provenance,
        mean_perception_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
    )


def _arm_dict(arm: _ArmResult, specs: list[EpisodeSpec]) -> dict:
    seed_by_id = {spec.episode_id: spec.seed for spec in specs}
    return {
        "perception": arm.perception,
        "summary": aggregate(arm.scores).to_dict(),
        "mean_perception_latency_ms": round(arm.mean_perception_latency_ms, 3),
        "episodes": [
            {
                "perception": arm.provenance[score.episode_id],
                "seed": seed_by_id[score.episode_id],
                **score.to_dict(),
            }
            for score in arm.scores
        ],
    }


def run_ablation(
    simulator: SimulatorBackend,
    specs: list[EpisodeSpec],
    *,
    candidate: Perception,
    baseline: Perception | None = None,
    controller_factory: Callable[[], Policy] = PurePursuitPolicy,
    on_episode: Callable[[str, EpisodeScore], None] | None = None,
) -> dict:
    """Run every spec under both perceptions and report the difference.

    Args:
        simulator: Backend both arms drive in, sequentially. Reuse is safe
            because ``reset(spec)`` reseeds everything from the spec.
        specs: The suite. Both arms run exactly this list, in this order —
            the two runs are identical in seed and specification by
            construction, and the report records the specs so a reader can
            check.
        candidate: The perception being measured, normally Detector-backed.
        baseline: The upper bound; privileged ground truth unless overridden.
        controller_factory: Builds the inner controller, fresh per Episode so
            a stateful controller cannot leak state across arms.
        on_episode: Called with ``(perception_name, score)`` after each
            Episode completes, in run order. Exists so a caller can persist
            partial results as they land — a two-arm CARLA suite is hours of
            simulation, and losing all of it to one late exception is not an
            acceptable failure mode.

    Returns:
        The report as a JSON-serialisable dict: both arms' per-episode scores
        and aggregates, their difference, observed provenance, and a scope
        label that says whether the numbers are driving quality at all.

    Raises:
        ValueError: If ``specs`` is empty; if both perceptions carry the same
            provenance name — the difference would measure nothing; or if a
            perception's camera geometry does not match the backend's — the
            resulting ranges would all be silently mis-scaled.
    """
    if not specs:
        raise ValueError("cannot run an ablation with no episodes")
    baseline = PrivilegedPerception() if baseline is None else baseline

    baseline_name, candidate_name = perception_name(baseline), perception_name(candidate)
    if baseline_name == candidate_name:
        raise ValueError(
            f"both arms report perception {baseline_name!r}; the arms must differ "
            "in perception or the ablation measures nothing"
        )

    expected_camera = _BACKEND_CAMERAS.get(simulator.name)
    for name, perception in ((baseline_name, baseline), (candidate_name, candidate)):
        camera = getattr(perception, "camera", None)
        if expected_camera is not None and camera is not None and camera != expected_camera:
            raise ValueError(
                f"perception {name!r} is configured for a different camera than the "
                f"{simulator.name} backend renders with; every range it derives "
                "would be silently mis-scaled. Pass the backend's camera geometry "
                "(KINEMATIC_CAMERA or CARLA_CAMERA) when building it."
            )

    logger.info(
        "ablation: %d episodes on %s, %s vs %s",
        len(specs), simulator.name, baseline_name, candidate_name,
    )
    baseline_arm = _run_arm(simulator, specs, baseline, controller_factory, on_episode)
    candidate_arm = _run_arm(simulator, specs, candidate, controller_factory, on_episode)

    baseline_summary = aggregate(baseline_arm.scores)
    candidate_summary = aggregate(candidate_arm.scores)
    scope, scope_note = _scope(simulator.name)

    # An Episode that failed mid-run in either arm scored the failure, not the
    # perception; its delta must never be read as perception cost. Flagged in
    # the report rather than dropped, because silently excluding episodes is
    # its own kind of lie about what the suite covered.
    failed_episodes = sorted(
        {
            score.episode_id
            for score in (*baseline_arm.scores, *candidate_arm.scores)
            if score.status != "completed"
        }
    )
    if failed_episodes:
        logger.warning(
            "%d episode(s) failed mid-run and their deltas are not perception "
            "cost: %s", len(failed_episodes), ", ".join(failed_episodes),
        )

    return {
        "kind": "perception_ablation",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": simulator.name,
        "scope": scope,
        "scope_note": scope_note,
        "perception_boundary": PERCEPTION_BOUNDARY,
        "controller": baseline_arm.controller,
        "episodes": [spec.to_dict() for spec in specs],
        "baseline": _arm_dict(baseline_arm, specs),
        "candidate": _arm_dict(candidate_arm, specs),
        "difference": {
            "definition": (
                "baseline minus candidate, in each metric's own units; positive "
                "means the candidate's perception cost driving performance. "
                "Deltas for episodes listed in failed_episodes measure a "
                "mid-run failure, not perception, and must be excluded from "
                "any perception-cost claim."
            ),
            "failed_episodes": failed_episodes,
            "driving_score": round(
                baseline_summary.driving_score - candidate_summary.driving_score, 2
            ),
            "route_completion": round(
                baseline_summary.route_completion - candidate_summary.route_completion, 4
            ),
            "infraction_penalty": round(
                baseline_summary.infraction_penalty - candidate_summary.infraction_penalty, 4
            ),
            "per_episode_driving_score": [
                {
                    "episode_id": b.episode_id,
                    "baseline": round(b.driving_score, 2),
                    "candidate": round(c.driving_score, 2),
                    "delta": round(b.driving_score - c.driving_score, 2),
                    # Statuses appear only on flagged rows, so a clean report
                    # stays readable and a dirty row cannot be missed.
                    **(
                        {"baseline_status": b.status, "candidate_status": c.status}
                        if b.episode_id in failed_episodes
                        else {}
                    ),
                }
                for b, c in zip(baseline_arm.scores, candidate_arm.scores, strict=True)
            ],
        },
    }


def _candidate_from_args(args: argparse.Namespace) -> Perception:
    """Build the Detector arm. A separate function so tests can substitute a
    stub at exactly the point the CLI would construct the real thing."""
    from pathfinder.perception.detector import DetectorPerception, YoloDetector

    camera = _BACKEND_CAMERAS[args.backend]
    return DetectorPerception(
        YoloDetector(args.weights, device=args.device, confidence=args.confidence),
        camera=camera,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ground-truth-versus-Detector perception ablation."
    )
    parser.add_argument("--episodes", type=int, default=10, help="episodes per arm")
    parser.add_argument(
        "--backend",
        choices=["kinematic", "carla"],
        default="kinematic",
        # 'auto' is deliberately not offered: a silent fallback could label a
        # kinematic run with a CARLA-shaped expectation in the operator's head.
        help="simulator backend; only carla produces driving-quality numbers",
    )
    parser.add_argument("--route-length-m", type=float, default=400.0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--weights", type=Path, default=TRAINED_WEIGHTS)
    parser.add_argument(
        "--device", default="cpu",
        help="Detector device; cpu is deterministic everywhere, pass cuda on the GPU box",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="report path; defaults to results/ablation/<backend>_report.json",
    )
    args = parser.parse_args(argv)

    # Imported here rather than at module top: orchestration pulls in the
    # cloud stack, which the library-level run_ablation has no business needing.
    from pathfinder.orchestration import build_episode_specs
    from pathfinder.sim.carla_backend import build_simulator

    specs = build_episode_specs(
        count=args.episodes,
        route_length_m=args.route_length_m,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
    )
    candidate = _candidate_from_args(args)
    # render=True on kinematic so the Detector arm has pixels to read; the
    # CARLA backend renders by default.
    simulator_kwargs = {"render": True} if args.backend == "kinematic" else {}

    output = args.output or Path("results/ablation") / f"{args.backend}_report.json"
    artifact = reporting.ReportArtifact(output, label_key="perception")

    with build_simulator(args.backend, **simulator_kwargs) as simulator:
        report = run_ablation(
            simulator, specs, candidate=candidate, on_episode=artifact.checkpoint
        )

    writeup = artifact.finish(report, render_writeup)

    print(f"backend: {report['backend']} ({report['scope']})")
    print(f"baseline  {report['baseline']['perception']:>12}: "
          f"driving score {report['baseline']['summary']['driving_score']}")
    print(f"candidate {report['candidate']['perception']:>12}: "
          f"driving score {report['candidate']['summary']['driving_score']}")
    print(f"difference (baseline - candidate): {report['difference']['driving_score']}")
    reporting.print_cli_tail(report, output, writeup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
