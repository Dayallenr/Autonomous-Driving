"""The comparison write-up renderer: report JSON in, honest Markdown out.

Issue #21's acceptance criteria pin the write-up's properties: the student's
observation-boundary sentence and its strictly-harder framing, the
not-project-work labelling on the behaviour agent, the weak-floor caveat on
``pure_pursuit``, the skipped-reference disclosure, and traceability to the
producing run. The reports here are built by the real ``run_comparison`` on
the kinematic backend with untrained weights, so the rendered dict has
exactly the shape the CLI serialises.
"""
from __future__ import annotations

import json

import pytest

import pathfinder.comparison_writeup as writeup
from pathfinder.comparison import STUDENT_OBSERVATION_BOUNDARY, run_comparison
from pathfinder.comparison_writeup import render_writeup
from pathfinder.orchestration import build_episode_specs
from pathfinder.policies import (
    PURE_PURSUIT,
    CarlaBehaviorAgentPolicy,
    CILStudentPolicy,
    build_policy,
)
from pathfinder.sim.kinematic import KinematicSimulator
from tests.support import make_comparison_arms as _arms

REFERENCE = CarlaBehaviorAgentPolicy.NAME


@pytest.fixture(scope="module")
def student(cil_checkpoint):
    return build_policy(CILStudentPolicy.NAME, weights=cil_checkpoint)


@pytest.fixture(scope="module")
def full_report(student):
    specs = build_episode_specs(count=2, route_length_m=60.0, base_seed=5, max_steps=20)
    with KinematicSimulator(render=True) as simulator:
        return run_comparison(simulator, specs, _arms(student))


@pytest.fixture(scope="module")
def skipped_reference_report(student):
    specs = build_episode_specs(count=2, route_length_m=60.0, base_seed=5, max_steps=20)
    with KinematicSimulator(render=True) as simulator:
        return run_comparison(
            simulator,
            specs,
            _arms(student, reference_skip="it needs the CARLA backend"),
        )


def test_the_boundary_sentence_and_strictly_harder_framing_are_stated(full_report):
    text = render_writeup(full_report, source="report.json")
    assert STUDENT_OBSERVATION_BOUNDARY in text
    assert "strictly harder task than either privileged baseline" in text
    assert "infer from pixels" in text


def test_the_behaviour_agent_is_labelled_not_project_work(full_report):
    text = render_writeup(full_report, source="report.json")
    assert REFERENCE in text
    assert "not project work" in text
    assert "presented as this project's driving" in text


def test_the_floor_is_stated_to_be_a_low_bar_with_its_measured_shortfall(full_report):
    text = render_writeup(full_report, source="report.json")
    assert "weak floor" in text
    assert "low bar" in text
    floor = full_report["arms"][0]
    shortfall = 100.0 - floor["summary"]["driving_score"]
    assert f"{shortfall:.2f} points" in text


def test_the_scope_banner_carries_the_pipeline_only_warning(full_report):
    text = render_writeup(full_report, source="report.json")
    assert "**Scope: pipeline-only.**" in text
    assert "never be quoted" in text


def test_a_driving_quality_report_drops_the_pipeline_warning(full_report):
    from pathfinder.comparison import _scope

    scope, scope_note = _scope("carla")
    report = {**full_report, "backend": "carla", "scope": scope, "scope_note": scope_note}
    text = render_writeup(report, source="report.json")
    assert "**Scope: driving-quality.**" in text
    assert "never be quoted" not in text


def test_the_table_columns_carry_each_arms_result_label(full_report, student):
    text = render_writeup(full_report, source="report.json")
    assert f"Floor (`{PURE_PURSUIT}`)" in text
    assert f"Student (`{student.model_version}`)" in text
    assert f"Reference ceiling (`{REFERENCE}`)" in text
    assert str(student.weights_path) in text


def test_a_skipped_reference_is_disclosed_not_omitted(skipped_reference_report):
    text = render_writeup(skipped_reference_report, source="report.json")
    assert "It did not run in this suite: it needs the CARLA backend." in text
    assert "not run" in text
    # The disclaimer still stands even when the column did not run.
    assert "not project work" in text


def test_the_writeup_carries_the_numbers_and_names_its_source(full_report):
    text = render_writeup(full_report, source="results/comparison/kinematic_report.json")
    assert "results/comparison/kinematic_report.json" in text
    assert full_report["generated_at"] in text
    for arm in full_report["arms"]:
        assert str(arm["summary"]["driving_score"]) in text
        for row in arm["episodes"]:
            assert row["episode_id"] in text


def test_regenerator_cli_rewrites_the_sibling_writeup(full_report, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(full_report), encoding="utf-8")
    assert writeup.main([str(path)]) == 0
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Policy comparison" in text


def test_regenerator_refuses_a_foreign_report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"kind": "dagger_run"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a policy_comparison report"):
        writeup.main([str(path)])
