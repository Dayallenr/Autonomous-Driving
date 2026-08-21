"""The behaviour-agent reference run (issue #16), proven CARLA-free.

The run's job is to record the reference ceiling as its own artifact before
any training exists. What these tests pin is that the artifact cannot lie:
it labels itself not-project-work, only a Policy declaring the behaviour
agent's name can produce it, failed episodes are flagged rather than
dropped, the default suite is byte-identical to the perception ablation's,
and the go / stop-and-reassess gate is computed only when the two reports
are actually comparable — same suite, both driving-quality.

The Policy here is a stub declaring the behaviour agent's name, the same
stance as ``test_policies`` and ``test_comparison``: the agent is CARLA's,
so what is testable without a server is everything about how the run
treats it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pathfinder.reference_run as reference_run
from pathfinder.comparison import DEFAULT_SUITE
from pathfinder.orchestration import build_episode_specs
from pathfinder.policies import CarlaBehaviorAgentPolicy
from pathfinder.reference_run import GATE_MARGIN_POINTS, floor_gate, run_reference
from pathfinder.runner import ControlOutput, PurePursuitPolicy
from pathfinder.sim.kinematic import KinematicSimulator
from tests.support import StubReferencePolicy

REFERENCE = CarlaBehaviorAgentPolicy.NAME

ABLATION_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "results/ablation/carla_report.json"
)


class ExplodingPolicy:
    NAME = REFERENCE

    def plan(self, state) -> ControlOutput:
        raise RuntimeError("deliberate mid-episode failure")


def tiny_specs(count: int = 2):
    return build_episode_specs(
        count=count, route_length_m=60.0, base_seed=5, max_steps=20
    )


def run(**kwargs):
    with KinematicSimulator(render=True) as simulator:
        return run_reference(
            simulator,
            kwargs.pop("specs", tiny_specs()),
            kwargs.pop("policy", StubReferencePolicy()),
            **kwargs,
        )


def carla_floor_report():
    return json.loads(ABLATION_ARTIFACT.read_text(encoding="utf-8"))


def as_driving_quality(report: dict, driving_score: float) -> dict:
    """Relabel a kinematic run as CARLA-scoped with a chosen aggregate, so the
    gate arithmetic is testable without a server. Only the gate reads these
    fields; the relabelling never leaves the test."""
    report = dict(report)
    report["backend"] = "carla"
    report["scope"] = "driving-quality"
    report["summary"] = {**report["summary"], "driving_score": driving_score}
    return report


# ─────────────────────────────────────────────────────────────────────────────
# The report
# ─────────────────────────────────────────────────────────────────────────────


def test_the_suite_is_scored_and_recorded():
    specs = tiny_specs()
    report = run(specs=specs)

    assert report["kind"] == "reference_baseline"
    assert report["policy"] == REFERENCE
    assert report["model_version"] == REFERENCE
    assert report["episodes"] == [spec.to_dict() for spec in specs]
    assert [row["episode_id"] for row in report["results"]] == [
        spec.episode_id for spec in specs
    ]
    assert [row["seed"] for row in report["results"]] == [spec.seed for spec in specs]
    assert report["summary"]["episodes"] == len(specs)


def test_the_artifact_labels_itself_not_project_work():
    report = run()
    assert report["not_project_work"] is True
    assert "not project work" in report["not_project_work_note"]


def test_the_kinematic_report_labels_itself_pipeline_only():
    report = run()
    assert report["backend"] == "kinematic"
    assert report["scope"] == "pipeline-only"
    assert "never be quoted" in report["scope_note"]


def test_only_carla_earns_the_driving_quality_scope():
    assert reference_run._scope("carla")[0] == "driving-quality"
    assert reference_run._scope("kinematic")[0] == "pipeline-only"
    assert reference_run._scope("some_future_backend")[0] == "pipeline-only"


def test_the_behavior_preset_is_recorded_when_the_policy_exposes_one():
    class PresetStub(StubReferencePolicy):
        behavior = "cautious"

    assert run(policy=PresetStub())["behavior"] == "cautious"
    assert "behavior" not in run()


# ─────────────────────────────────────────────────────────────────────────────
# Attribution cannot be faked
# ─────────────────────────────────────────────────────────────────────────────


def test_a_policy_that_is_not_the_behaviour_agent_is_refused():
    with pytest.raises(ValueError, match="declares"):
        run(policy=PurePursuitPolicy())


def test_a_policy_that_declares_no_name_is_refused():
    class NamelessPolicy:
        def plan(self, state):
            raise AssertionError("never reached")

    with pytest.raises(ValueError, match="declares"):
        run(policy=NamelessPolicy())


def test_an_empty_suite_is_refused():
    with pytest.raises(ValueError, match="episode"):
        run(specs=[])


# ─────────────────────────────────────────────────────────────────────────────
# Failures and checkpointing
# ─────────────────────────────────────────────────────────────────────────────


def test_failed_episodes_are_flagged_not_dropped():
    report = run(policy=ExplodingPolicy())
    assert report["summary"]["failures"] == 2
    assert report["failed_episodes"] == [spec.episode_id for spec in tiny_specs()]
    assert {row["status"] for row in report["results"]} == {"failed"}


def test_each_finished_episode_is_checkpointed_in_run_order():
    calls: list[tuple[str, str]] = []
    run(on_episode=lambda version, score: calls.append((version, score.episode_id)))
    assert calls == [(REFERENCE, spec.episode_id) for spec in tiny_specs()]


# ─────────────────────────────────────────────────────────────────────────────
# The floor gate
# ─────────────────────────────────────────────────────────────────────────────


def test_the_gate_needs_a_floor_report():
    gate = floor_gate(run(), None, floor_source="results/ablation/carla_report.json")
    assert gate["computed"] is False
    assert "floor report" in gate["reason"]


def test_the_gate_refuses_a_pipeline_only_reference():
    gate = floor_gate(run(), carla_floor_report(), floor_source=str(ABLATION_ARTIFACT))
    assert gate["computed"] is False
    assert "pipeline-only" in gate["reason"]


def test_the_gate_refuses_a_floor_that_is_not_the_carla_ablation():
    reference = as_driving_quality(run(specs=tiny_specs()), 40.0)

    wrong_kind = {**carla_floor_report(), "kind": "policy_comparison"}
    assert floor_gate(reference, wrong_kind, floor_source="x")["computed"] is False

    pipeline_only = {**carla_floor_report(), "scope": "pipeline-only"}
    assert floor_gate(reference, pipeline_only, floor_source="x")["computed"] is False

    wrong_controller = {**carla_floor_report(), "controller": "cil_student"}
    assert floor_gate(reference, wrong_controller, floor_source="x")["computed"] is False


def test_the_gate_requires_the_exact_same_suite():
    reference = as_driving_quality(run(specs=tiny_specs()), 40.0)
    gate = floor_gate(reference, carla_floor_report(), floor_source="x")
    assert gate["computed"] is False
    assert "suite" in gate["reason"]


def test_the_gate_verdict_goes_both_ways():
    floor = carla_floor_report()
    suite = build_episode_specs(**DEFAULT_SUITE)
    base = run(specs=tiny_specs())
    base["episodes"] = [spec.to_dict() for spec in suite]
    floor_score = floor["baseline"]["summary"]["driving_score"]

    clearly_above = floor_gate(
        as_driving_quality(base, floor_score + GATE_MARGIN_POINTS),
        floor,
        floor_source=str(ABLATION_ARTIFACT),
    )
    assert clearly_above["computed"] is True
    assert clearly_above["floor_driving_score"] == floor_score
    assert clearly_above["margin"] == GATE_MARGIN_POINTS
    assert clearly_above["verdict"] == "go"
    assert clearly_above["floor_source"] == str(ABLATION_ARTIFACT)

    barely_above = floor_gate(
        as_driving_quality(base, floor_score + GATE_MARGIN_POINTS - 0.1),
        floor,
        floor_source=str(ABLATION_ARTIFACT),
    )
    assert barely_above["verdict"] == "stop-and-reassess"


def test_the_gate_flags_failed_episodes_as_a_caveat():
    floor = carla_floor_report()
    suite = build_episode_specs(**DEFAULT_SUITE)
    base = run(specs=tiny_specs(), policy=ExplodingPolicy())
    base["episodes"] = [spec.to_dict() for spec in suite]

    gate = floor_gate(as_driving_quality(base, 50.0), floor, floor_source="x")
    assert gate["computed"] is True
    assert "failed" in gate["caveat"]


def test_the_recorded_carla_floor_is_the_ablations_privileged_arm():
    """The 25.2 the ticket gates against is the privileged PurePursuit arm of
    the recorded CARLA ablation; pin that the gate reads exactly that field."""
    floor = carla_floor_report()
    assert floor["baseline"]["perception"] == "privileged"
    assert floor["controller"] == "pure_pursuit"
    assert floor["baseline"]["summary"]["driving_score"] == 25.2


# ─────────────────────────────────────────────────────────────────────────────
# The CLI
# ─────────────────────────────────────────────────────────────────────────────


def test_the_default_suite_is_the_perception_ablations_exact_suite():
    assert reference_run.DEFAULT_SUITE is DEFAULT_SUITE


def test_the_cli_runs_end_to_end_on_kinematic_with_a_stub(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        reference_run,
        "_build_reference_policy",
        lambda args, simulator: StubReferencePolicy(),
    )
    output = tmp_path / "report.json"
    assert (
        reference_run.main(
            [
                "--backend", "kinematic",
                "--episodes", "2",
                "--route-length-m", "60",
                "--max-steps", "20",
                "--output", str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["kind"] == "reference_baseline"
    assert report["scope"] == "pipeline-only"
    assert report["not_project_work"] is True
    # The gate is present but honestly uncomputed: a pipeline-only run cannot
    # be compared against the CARLA floor.
    assert report["floor_gate"]["computed"] is False
    # Both artifacts land together, and the partial file is gone.
    assert output.with_suffix(".md").exists()
    assert not output.with_suffix(".partial.jsonl").exists()
    # The suite differs from the floor report's, and an unattended run must
    # hear about the forfeited gate before the episodes start, not after.
    assert "WARNING: this suite differs from the floor report's" in capsys.readouterr().out


def test_the_cli_refuses_the_kinematic_backend_without_a_stub(tmp_path):
    """Without CARLA there is no behaviour agent to run; the refusal must be
    eager and name the constraint, not a mid-suite crash."""
    with pytest.raises(SystemExit, match="CARLA"):
        reference_run.main(
            [
                "--backend", "kinematic",
                "--episodes", "1",
                "--output", str(tmp_path / "report.json"),
            ]
        )
