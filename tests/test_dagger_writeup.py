"""The DAgger write-up renderer: report JSON in, honest Markdown out.

Issue #20's acceptance criteria pin the write-up's properties: the
smoke-is-not-a-result banner whenever the configuration is below the
documented real-run bar, the disagreement-direction caveat either way, and
traceability to the producing run. The reports here are built by the real
``build_run_report`` from fabricated rows, so the rendered dict has exactly
the shape the CLI serialises — no training required.
"""
from __future__ import annotations

import json

import pytest

from pathfinder.dagger import DAggerConfig, IterationReport, build_run_report
from pathfinder.dagger_writeup import render_writeup
from pathfinder.policies import PURE_PURSUIT, CarlaBehaviorAgentPolicy


def _rows(dataset_sizes=(41, 82, 123)) -> list[IterationReport]:
    return [
        IterationReport(
            iteration=index,
            beta=1.0 if index == 0 else 0.5**index,
            dataset_size=size,
            train_loss=0.51234 - index * 0.1,
            disagreement=float("nan") if index == 0 else 0.21 + index * 0.01,
            driving_score=62.0,
            route_completion=0.62,
            seconds=3.2,
        )
        for index, size in enumerate(dataset_sizes)
    ]


def smoke_report(**overrides) -> dict:
    report = build_run_report(
        backend="kinematic",
        expert_name=PURE_PURSUIT,
        student={"model": "CILModel", "output_mode": "control", "imagenet_pretrained": False},
        config=DAggerConfig(iterations=3, episodes_per_iteration=1, epochs_per_iteration=1),
        rows=_rows(),
        weights="checkpoints/iteration_001.pt",
    )
    report.update(overrides)
    return report


def real_run_report() -> dict:
    """A configuration that clears every documented real-run criterion."""
    config = DAggerConfig(iterations=10, episodes_per_iteration=6, epochs_per_iteration=3)
    rows = _rows(dataset_sizes=tuple(4_000 * (index + 1) for index in range(10)))
    report = build_run_report(
        backend="carla",
        expert_name=CarlaBehaviorAgentPolicy.NAME,
        student={"model": "CILModel", "output_mode": "control", "imagenet_pretrained": True},
        config=config,
        rows=rows,
        weights="checkpoints/iteration_009.pt",
    )
    assert report["real_run_bar"]["met"] is True  # fixture sanity
    return report


def test_below_the_bar_carries_the_smoke_banner_naming_unmet_criteria():
    report = smoke_report()
    assert report["scope"] == "smoke"
    text = render_writeup(report, source="report.json")
    assert "below the documented real-run bar" in text
    assert "loop check" in text
    assert "none may ever be quoted" in text
    # The reader sees which criteria failed, not just that some did.
    for name in ("backend", "imagenet_pretrained_backbone", "final_dataset_size", "total_epochs"):
        assert name in text


def test_meeting_the_bar_drops_the_smoke_banner():
    text = render_writeup(real_run_report(), source="report.json")
    assert "meeting the documented real-run bar" in text
    assert "below the documented real-run bar" not in text


def test_disagreement_direction_caveat_appears_in_both_scopes():
    for report in (smoke_report(), real_run_report()):
        text = render_writeup(report, source="report.json")
        assert "pushing disagreement **up**" in text
        assert "pushing it **down**" in text
        assert "does not by itself mean the loop is broken" in text


def test_rollout_numbers_are_labelled_not_a_driving_score():
    for report in (smoke_report(), real_run_report()):
        text = render_writeup(report, source="report.json")
        assert "not the student's driving score" in text
        assert "scored comparison suite" in text


def test_behaviour_agent_expert_is_labelled_not_project_work():
    text = render_writeup(real_run_report(), source="report.json")
    assert CarlaBehaviorAgentPolicy.NAME in text
    assert "not project work" in text


def test_project_expert_is_not_disclaimed():
    text = render_writeup(smoke_report(), source="report.json")
    assert "not project work" not in text


def test_writeup_names_its_source_artifact():
    report = smoke_report()
    text = render_writeup(report, source="results/dagger/kinematic_smoke/report.json")
    assert "results/dagger/kinematic_smoke/report.json" in text
    assert report["generated_at"] in text


def test_writeup_carries_the_per_iteration_numbers():
    report = smoke_report()
    text = render_writeup(report, source="report.json")
    for row in report["iterations"]:
        assert str(row["dataset_size"]) in text
        assert str(row["train_loss"]) in text
        # The spec names the per-iteration driving score; the write-up must
        # carry it (labelled by the rollout caveat), not silently drop it.
        assert str(row["driving_score"]) in text
    assert str(report["error_reduction_pct"]) in text


def test_single_measurement_renders_not_measurable_rather_than_zero():
    """A two-iteration run has one disagreement measurement; its 'change' is
    undefined, and printing 0.0% would dress that up as 'no change'."""
    report = build_run_report(
        backend="kinematic",
        expert_name=PURE_PURSUIT,
        student={"model": "CILModel", "output_mode": "control", "imagenet_pretrained": False},
        config=DAggerConfig(iterations=2, episodes_per_iteration=1, epochs_per_iteration=1),
        rows=_rows(dataset_sizes=(41, 82)),
        weights="checkpoints/iteration_001.pt",
    )
    assert report["error_reduction_pct"] is None
    text = render_writeup(report, source="report.json")
    assert "not measurable" in text


def test_regenerator_cli_rewrites_the_sibling_writeup(tmp_path):
    import pathfinder.dagger_writeup as writeup

    path = tmp_path / "report.json"
    path.write_text(json.dumps(smoke_report()), encoding="utf-8")
    assert writeup.main([str(path)]) == 0
    assert "loop check" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_regenerator_refuses_a_foreign_report(tmp_path):
    import pathfinder.dagger_writeup as writeup

    path = tmp_path / "report.json"
    path.write_text(json.dumps({"kind": "perception_ablation"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a dagger_run report"):
        writeup.main([str(path)])
