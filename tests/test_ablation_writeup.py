"""
The ablation write-up renderer: report JSON in, honest Markdown out.

Issue #10's acceptance criteria put hard requirements on the write-up — the
exact perception-boundary sentence, an infraction breakdown, traceability to
the run that produced it, and a recorded (not acted-on) fine-tuning trigger.
Generating the write-up from the report artifact makes those properties
checkable here instead of hoping a future author remembers them.

The fixtures run the real ablation on the kinematic backend so the rendered
dict has exactly the shape ``run_ablation`` produces; the CARLA-scope cases
relabel that report the same way ``_scope`` would, because no Mac can run
CARLA and the renderer only ever reads the dict.
"""
from __future__ import annotations

import json

import pytest

from pathfinder.ablation import PERCEPTION_BOUNDARY, run_ablation
from pathfinder.ablation_writeup import render_writeup
from pathfinder.perception import PerceivedScene
from pathfinder.sim import EpisodeSpec, KinematicSimulator
from pathfinder.sim.base import FrameState


class _BlindPerception:
    NAME = "stub_blind"

    def perceive(self, state: FrameState) -> PerceivedScene:
        return PerceivedScene()


def _specs(count: int = 2) -> list[EpisodeSpec]:
    return [
        EpisodeSpec(
            episode_id=f"abl-{index:02d}",
            town=("Town01", "Town03")[index % 2],
            weather=("ClearNoon", "WetNoon")[index % 2],
            route_length_m=80.0,
            max_steps=40,
            seed=100 + index,
        )
        for index in range(count)
    ]


@pytest.fixture(scope="module")
def kinematic_report() -> dict:
    with KinematicSimulator() as simulator:
        return run_ablation(simulator, _specs(), candidate=_BlindPerception())


def _as_carla(report: dict, **overrides) -> dict:
    """Relabel a kinematic report as the CARLA-scope report it is shaped like,
    using the ablation's own scope labelling so the two cannot drift apart."""
    from pathfinder.ablation import _scope

    relabelled = json.loads(json.dumps(report))  # deep copy, JSON-clean
    relabelled["backend"] = "carla"
    relabelled["scope"], relabelled["scope_note"] = _scope("carla")
    relabelled.update(overrides)
    return relabelled


def _set_scores(report: dict, baseline: float, candidate: float) -> dict:
    """Force the aggregate driving scores so the trigger rule is testable."""
    report["baseline"]["summary"]["driving_score"] = baseline
    report["candidate"]["summary"]["driving_score"] = candidate
    report["difference"]["driving_score"] = round(baseline - candidate, 2)
    for row in report["difference"]["per_episode_driving_score"]:
        row["baseline"], row["candidate"] = baseline, candidate
        row["delta"] = round(baseline - candidate, 2)
    return report


def test_writeup_states_the_perception_boundary_verbatim(kinematic_report):
    """Criterion: the write-up says 'perceived obstacles, privileged
    localization and traffic lights' rather than claiming full perception."""
    text = render_writeup(kinematic_report, source="kinematic_report.json")
    assert PERCEPTION_BOUNDARY in text


def test_writeup_carries_scores_completion_and_infractions(kinematic_report):
    """Criterion: both driving scores, route completion, an infraction
    breakdown, and the difference."""
    report = _as_carla(kinematic_report)
    report["baseline"]["summary"]["infraction_totals"] = {"collision_vehicle": 3}
    report["candidate"]["summary"]["infraction_totals"] = {"lane_invasion": 2}
    text = render_writeup(report, source="carla_report.json")

    for summary in (report["baseline"]["summary"], report["candidate"]["summary"]):
        assert f"{summary['driving_score']}" in text
        assert f"{summary['route_completion']}" in text
    assert f"{report['difference']['driving_score']}" in text
    assert "collision_vehicle" in text
    assert "lane_invasion" in text


def test_writeup_names_its_source_artifact(kinematic_report):
    """Criterion: a reader can trace the write-up back to the run. The
    write-up must name the report file and its generation timestamp."""
    text = render_writeup(kinematic_report, source="results/ablation/kinematic_report.json")
    assert "results/ablation/kinematic_report.json" in text
    assert kinematic_report["generated_at"] in text


def test_pipeline_only_reports_render_with_the_warning_and_no_verdict(kinematic_report):
    """A kinematic write-up must shout pipeline-only and must not evaluate
    the fine-tuning trigger — its numbers are not driving quality."""
    text = render_writeup(kinematic_report, source="kinematic_report.json")
    assert "pipeline-only" in text
    assert "not driving quality" in text
    assert "fine-tuning" not in text.lower()


def test_fine_tuning_trigger_recorded_when_perception_binds(kinematic_report):
    """Criterion: if perception is the binding constraint, that is recorded as
    triggering the deferred fine-tuning decision — recorded, not acted on.
    Binding: perception costs more score than everything else combined."""
    report = _set_scores(_as_carla(kinematic_report), baseline=80.0, candidate=20.0)
    text = render_writeup(report, source="carla_report.json")
    # cost 60 > shortfall 20: perception binds.
    assert "binding constraint" in text
    assert "fine-tuning" in text
    assert "recorded, not acted on" in text


def test_fine_tuning_trigger_not_fired_when_controller_binds(kinematic_report):
    report = _set_scores(_as_carla(kinematic_report), baseline=40.0, candidate=35.0)
    text = render_writeup(report, source="carla_report.json")
    # cost 5 < shortfall 60: the controller, not perception, binds.
    assert "binding constraint" in text  # the rule is stated either way
    assert "not the binding constraint" in text


def test_trigger_verdict_excludes_failed_episodes(kinematic_report):
    """A mid-run failure scores the failure, not the perception; the verdict
    must be computed over clean episodes only, and say so."""
    report = _set_scores(_as_carla(kinematic_report), baseline=40.0, candidate=35.0)
    rows = report["difference"]["per_episode_driving_score"]
    # Poison one episode with a failure delta that would flip the verdict.
    rows[0].update(baseline=95.0, candidate=0.0, delta=95.0,
                   baseline_status="completed", candidate_status="failed")
    report["difference"]["failed_episodes"] = [rows[0]["episode_id"]]
    text = render_writeup(report, source="carla_report.json")
    assert "not the binding constraint" in text
    assert rows[0]["episode_id"] in text  # the exclusion is named, not silent


def test_carla_writeup_states_the_near_field_divergence(kinematic_report):
    """The one known measurement divergence that *is* perception (issue #10):
    the Detector floors at the camera's minimum measurable range."""
    text = render_writeup(_as_carla(kinematic_report), source="carla_report.json")
    assert "min_measurable_range_m" in text


def test_writeup_lists_towns_and_weathers_covered(kinematic_report):
    """Criterion: the ablation runs across towns and weather; the write-up
    must show the coverage so 'unseen towns and weather' is checkable."""
    text = render_writeup(kinematic_report, source="kinematic_report.json")
    for value in ("Town01", "Town03", "ClearNoon", "WetNoon"):
        assert value in text


def test_carla_writeup_states_the_domain_is_unseen(kinematic_report):
    """Criterion: 'towns and weather the Detector never trained on'. The CARLA
    write-up must say the coverage is unseen — KITTI-trained, synthetic
    renders — rather than leaving the reader to know the training set."""
    text = render_writeup(_as_carla(kinematic_report), source="carla_report.json")
    assert "unseen" in text
    assert "KITTI" in text


def test_cli_writes_the_writeup_next_to_the_report(tmp_path, monkeypatch):
    """The one command produces both artifacts, so the CARLA sitting cannot
    end with a report but no write-up."""
    import pathfinder.ablation as ablation

    monkeypatch.setattr(ablation, "_candidate_from_args", lambda args: _BlindPerception())
    output = tmp_path / "report.json"

    exit_code = ablation.main(
        [
            "--episodes", "1",
            "--backend", "kinematic",
            "--route-length-m", "60",
            "--max-steps", "30",
            "--output", str(output),
        ]
    )

    assert exit_code == 0
    writeup = output.with_suffix(".md")
    assert writeup.exists()
    text = writeup.read_text()
    assert PERCEPTION_BOUNDARY in text
    assert output.name in text
