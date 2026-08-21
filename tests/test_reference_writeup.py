"""The reference-baseline write-up: issue #16 requires the artifact itself to
carry the not-project-work label, the scope of its numbers, and the go /
stop-and-reassess gate — so the renderer is tested for exactly those
sentences, the same stance as ``test_comparison_writeup``."""
from __future__ import annotations

import json

import pytest

import pathfinder.reference_writeup as reference_writeup
from pathfinder.orchestration import build_episode_specs
from pathfinder.reference_run import run_reference
from pathfinder.reference_writeup import render_writeup
from pathfinder.sim.kinematic import KinematicSimulator
from tests.support import StubReferencePolicy


@pytest.fixture(scope="module")
def report():
    specs = build_episode_specs(count=2, route_length_m=60.0, base_seed=5, max_steps=20)
    with KinematicSimulator(render=True) as simulator:
        result = run_reference(simulator, specs, StubReferencePolicy())
    result["floor_gate"] = {
        "computed": False,
        "reason": "this run is pipeline-only; only a driving-quality run gates",
    }
    return result


def render(report_dict) -> str:
    return render_writeup(report_dict, source="results/reference/kinematic_report.json")


def test_the_writeup_carries_the_required_sentences(report):
    text = render(report)
    assert "not project work" in text
    assert "Scope: pipeline-only" in text
    assert "results/reference/kinematic_report.json" in text
    assert "carla_builtin_behavior_agent" in text


def test_every_episode_appears_with_its_seed(report):
    text = render(report)
    for spec in report["episodes"]:
        assert spec["episode_id"] in text
        assert str(spec["seed"]) in text


def test_an_uncomputed_gate_states_its_reason(report):
    text = render(report)
    assert "Not computed" in text
    assert "pipeline-only; only a driving-quality run gates" in text


def test_a_computed_gate_states_the_verdict_and_both_numbers(report):
    gated = dict(report)
    gated["floor_gate"] = {
        "computed": True,
        "definition": "pre-registered rule",
        "floor_source": "results/ablation/carla_report.json",
        "floor_policy": "pure_pursuit under privileged perception",
        "floor_driving_score": 25.2,
        "reference_driving_score": 41.0,
        "margin": 15.8,
        "required_margin": 10.0,
        "verdict": "go",
    }
    text = render(gated)
    assert "**Verdict: go**" in text
    assert "25.2" in text
    assert "41.0" in text
    assert "results/ablation/carla_report.json" in text

    gated["floor_gate"] = {**gated["floor_gate"], "verdict": "stop-and-reassess"}
    assert "**Verdict: stop-and-reassess**" in render(gated)


def test_failed_episodes_are_called_out(report):
    flagged = dict(report)
    flagged["failed_episodes"] = [report["episodes"][0]["episode_id"]]
    text = render(flagged)
    assert "failed mid-run" in text


def test_the_cli_regenerates_the_sibling_markdown(report, tmp_path):
    artifact = tmp_path / "report.json"
    artifact.write_text(json.dumps(report), encoding="utf-8")
    assert reference_writeup.main([str(artifact)]) == 0
    assert "not project work" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_the_cli_refuses_a_report_of_the_wrong_kind(tmp_path):
    artifact = tmp_path / "report.json"
    artifact.write_text(json.dumps({"kind": "policy_comparison"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="reference_baseline"):
        reference_writeup.main([str(artifact)])
