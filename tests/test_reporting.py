"""
The shared report/CLI machinery (issue #29).

`ablation`, `comparison`, and `reference_run` each grew a near-copy of the
same scope labelling, partial-checkpoint insurance, and write-up helpers;
`pathfinder/reporting.py` is now their one home. These tests pin the shared
behaviour directly — the per-report golden tests pin that extraction changed
no artifact shape or sentence.
"""
from __future__ import annotations

import json

import pytest

from pathfinder.reporting import (
    NOT_PROJECT_WORK,
    SCOPE_DRIVING_QUALITY,
    SCOPE_PIPELINE_ONLY,
    ReportArtifact,
    generated_from,
    infraction_table,
    print_cli_tail,
    regenerate_writeup_main,
    scope,
    scope_banner,
    suite_section,
    unique_in_order,
)


def _scope(backend_name: str) -> tuple[str, str]:
    return scope(
        backend_name,
        driving_quality_note="Real numbers.",
        pipeline_limits="neither simulates real physics nor renders real scenes",
        pipeline_label="test pipeline",
    )


class FakeScore:
    def __init__(self, episode_id: str = "ep-000", driving_score: float = 1.0):
        self.episode_id = episode_id
        self.driving_score = driving_score

    def to_dict(self) -> dict:
        return {"episode_id": self.episode_id, "driving_score": self.driving_score}


# ─────────────────────────────────────────────────────────────────────────────
# Scope labelling
# ─────────────────────────────────────────────────────────────────────────────


def test_only_carla_earns_the_driving_quality_scope():
    assert _scope("carla") == (SCOPE_DRIVING_QUALITY, "Real numbers.")
    assert _scope("kinematic")[0] == SCOPE_PIPELINE_ONLY


def test_unknown_backends_default_to_pipeline_only():
    """The safe failure mode for a backend that does not exist yet is
    underclaiming; driving-quality must be earned by name."""
    assert _scope("some_future_backend")[0] == SCOPE_PIPELINE_ONLY


def test_the_pipeline_only_note_names_backend_limits_and_pipeline():
    _, note = _scope("kinematic")
    assert note.startswith(
        "Generated on the kinematic backend, which neither simulates real "
        "physics nor renders real scenes."
    )
    assert "verify the test pipeline end to end" in note
    assert "never be quoted" in note
    assert "The real measurement comes from the CARLA backend." in note


# ─────────────────────────────────────────────────────────────────────────────
# ReportArtifact — partial checkpoints plus the report/write-up landing
# ─────────────────────────────────────────────────────────────────────────────


def test_construction_claims_the_output_path(tmp_path):
    """The parent directory exists afterwards and a stale partial from an
    earlier run is gone — it would otherwise pollute this run's checkpoints."""
    output = tmp_path / "results" / "report.json"
    output.parent.mkdir()
    stale = output.with_suffix(".partial.jsonl")
    stale.write_text("stale row\n", encoding="utf-8")

    artifact = ReportArtifact(output, label_key="model_version")

    assert not stale.exists()
    assert artifact.partial == stale

    nested = tmp_path / "does" / "not" / "exist" / "report.json"
    ReportArtifact(nested, label_key="model_version")
    assert nested.parent.is_dir()


def test_checkpoint_appends_rows_under_the_label_key(tmp_path):
    artifact = ReportArtifact(tmp_path / "report.json", label_key="perception")
    artifact.checkpoint("privileged", FakeScore("ep-000", 25.2))
    artifact.checkpoint("detector", FakeScore("ep-001", 8.86))

    rows = [
        json.loads(line)
        for line in artifact.partial.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {"perception": "privileged", "episode_id": "ep-000", "driving_score": 25.2},
        {"perception": "detector", "episode_id": "ep-001", "driving_score": 8.86},
    ]


def test_finish_lands_both_artifacts_and_retires_the_partial(tmp_path):
    output = tmp_path / "report.json"
    artifact = ReportArtifact(output, label_key="model_version")
    artifact.checkpoint("stub", FakeScore())

    report = {"kind": "test_report", "difference": "− 1"}

    def render(rendered: dict, *, source: str) -> str:
        assert rendered is report
        return f"# Write-up of {source}: − 1\n"

    writeup = artifact.finish(report, render)

    assert not artifact.partial.exists()
    assert writeup == output.with_suffix(".md")
    # The JSON round-trips (written utf-8, indent, trailing newline) and the
    # write-up survives its non-ASCII characters on any platform encoding.
    text = output.read_text(encoding="utf-8")
    assert json.loads(text) == report
    assert text.endswith("\n")
    assert "−" in writeup.read_text(encoding="utf-8")


def test_a_failed_render_still_leaves_report_written_and_partial_gone(tmp_path):
    """The partial is retired the moment the full report exists — a render
    crash afterwards must not resurrect the impression of an unfinished run."""
    output = tmp_path / "report.json"
    artifact = ReportArtifact(output, label_key="model_version")
    artifact.checkpoint("stub", FakeScore())

    def broken_render(report: dict, *, source: str) -> str:
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError):
        artifact.finish({"kind": "test_report"}, broken_render)
    assert output.exists()
    assert not artifact.partial.exists()


# ─────────────────────────────────────────────────────────────────────────────
# The CLI tail
# ─────────────────────────────────────────────────────────────────────────────


def test_the_cli_tail_warns_on_pipeline_only_runs(tmp_path, capsys):
    output, writeup = tmp_path / "r.json", tmp_path / "r.md"
    print_cli_tail({"scope": SCOPE_PIPELINE_ONLY}, output, writeup)
    out = capsys.readouterr().out
    assert "NOTE: pipeline-only run — these numbers are not driving quality." in out
    assert f"report written to {output}" in out
    assert f"write-up written to {writeup}" in out


def test_the_cli_tail_stays_quiet_on_driving_quality_runs(tmp_path, capsys):
    print_cli_tail(
        {"scope": SCOPE_DRIVING_QUALITY}, tmp_path / "r.json", tmp_path / "r.md"
    )
    assert "NOTE" not in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
# Write-up helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_unique_in_order_preserves_first_appearance():
    assert unique_in_order(["Town05", "Town01", "Town05", "Town03"]) == [
        "Town05", "Town01", "Town03",
    ]


def test_the_header_cites_the_artifact_and_its_generator():
    header = generated_from(
        {"generated_at": "2026-08-21T00:00:00+00:00"},
        source="results/reference/carla_report.json",
        generator="pathfinder/reference_writeup.py",
    )
    assert header.startswith("Generated 2026-08-21T00:00:00+00:00 from ")
    # The link target is the sibling file name, not the repo-relative path —
    # the write-up lands next to the report.
    assert "[`results/reference/carla_report.json`](carla_report.json)" in header
    assert "`pathfinder/reference_writeup.py`" in header
    assert "edit the generator, not this file" in header


def test_the_scope_banner_quotes_scope_and_note():
    banner = scope_banner({"scope": "pipeline-only", "scope_note": "Not real."})
    assert banner == "> **Scope: pipeline-only.** Not real."


def test_the_suite_section_summarises_specs_under_the_given_label():
    specs = [
        {"town": "Town01", "weather": "ClearNoon", "seed": 1000},
        {"town": "Town03", "weather": "WetNoon", "seed": 1001},
        {"town": "Town01", "weather": "ClearNoon", "seed": 1002},
    ]
    assert suite_section(specs, episodes_label="Episodes per arm") == [
        "## Suite",
        "",
        "- Episodes per arm: 3",
        "- Towns: Town01, Town03",
        "- Weathers: ClearNoon, WetNoon",
        "- Seeds: 1000–1002",
    ]


def test_an_empty_suite_renders_honestly_instead_of_crashing():
    """The current report kinds refuse empty suites up front, but a future
    kind (the Phase 5 distributed run pulls episodes off a queue) can
    legitimately finish with none — the write-up must still land beside the
    JSON rather than die in min() over no seeds."""
    assert suite_section([], episodes_label="Episodes") == [
        "## Suite",
        "",
        "- Episodes: 0",
    ]


def test_the_infraction_table_unions_and_sorts_names():
    lines = infraction_table(
        [("Baseline", {"collision_vehicle": 2}), ("Candidate", {"agent_blocked": 1})],
        empty_message="unused",
    )
    assert lines == [
        "| Infraction | Baseline | Candidate |",
        "|---|---|---|",
        "| agent_blocked | 0 | 1 |",
        "| collision_vehicle | 2 | 0 |",
    ]


def test_a_skipped_column_renders_not_run():
    lines = infraction_table(
        [("Floor", {"agent_blocked": 3}), ("Reference", None)],
        empty_message="unused",
    )
    assert lines[2] == "| agent_blocked | 3 | not run |"


def test_no_infractions_anywhere_renders_the_empty_message():
    assert infraction_table(
        [("Floor", {}), ("Reference", None)],
        empty_message="No arm committed a scoreable or tracked infraction.",
    ) == ["No arm committed a scoreable or tracked infraction."]


def test_the_not_project_work_sentence_disclaims_the_behaviour_agent():
    assert "**not project work**" in NOT_PROJECT_WORK
    assert "reference upper bound" in NOT_PROJECT_WORK


# ─────────────────────────────────────────────────────────────────────────────
# The write-up regenerator CLI
# ─────────────────────────────────────────────────────────────────────────────


def test_regenerating_writes_the_sibling_markdown(tmp_path, capsys):
    artifact = tmp_path / "report.json"
    artifact.write_text(json.dumps({"kind": "test_report"}), encoding="utf-8")

    exit_code = regenerate_writeup_main(
        [str(artifact)],
        kind="test_report",
        render_writeup=lambda report, *, source: f"# From {source}\n",
        description="Regenerate.",
    )

    assert exit_code == 0
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == (
        f"# From {artifact}\n"
    )
    assert "write-up written to" in capsys.readouterr().out


def test_regenerating_refuses_a_report_of_the_wrong_kind(tmp_path):
    artifact = tmp_path / "report.json"
    artifact.write_text(json.dumps({"kind": "something_else"}), encoding="utf-8")

    with pytest.raises(SystemExit, match="is not a test_report report"):
        regenerate_writeup_main(
            [str(artifact)],
            kind="test_report",
            render_writeup=lambda report, *, source: "",
            description="Regenerate.",
        )
