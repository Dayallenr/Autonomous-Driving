"""The DAgger CLI: one command that cannot end with numbers but no document.

Issue #20's acceptance criteria: the command produces weights, report JSON, and
generated write-up together; killing a run mid-iteration and re-running loses
at most one iteration's work; artifact writes are explicit UTF-8; and a smoke
configuration exercises all of it on CPU in minutes.

The smoke fixture runs the real CLI with the real CILModel once per module.
The crash/resume tests substitute a tiny model at the CLI's model-building
seam (mirroring how the ablation tests substitute ``_candidate_from_args``)
so they stay fast, and simulate the crash by making the training step raise —
the loop dies exactly where a real mid-iteration failure would kill it.
"""
from __future__ import annotations

import json

import pytest
import torch

import pathfinder.dagger as dagger


def _strip_timing(report: dict) -> dict:
    """A report minus wall-clock: what must be identical across a resumed run
    and an uninterrupted one."""
    stripped = json.loads(json.dumps(report))
    stripped.pop("generated_at")
    stripped.pop("total_seconds")
    for row in stripped["iterations"]:
        row.pop("seconds")
    return stripped


def _tiny_model_builder(args):
    from tests.test_dagger import TinyBranchModel

    torch.manual_seed(args.seed)
    return TinyBranchModel()


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """One full --smoke run of the unmodified CLI, shared read-only."""
    out = tmp_path_factory.mktemp("dagger") / "run"
    assert dagger.main(["--smoke", "--output-dir", str(out)]) == 0
    return out


def test_smoke_run_lands_weights_report_and_writeup_together(smoke_run):
    report = json.loads((smoke_run / "report.json").read_text(encoding="utf-8"))
    assert report["kind"] == "dagger_run"
    assert len(report["iterations"]) == report["config"]["iterations"] == 2
    # The recorded weights path resolves to a checkpoint that exists.
    assert (smoke_run / report["weights"]).exists()
    assert (smoke_run / "checkpoints" / "iteration_000.pt").exists()
    assert (smoke_run / "checkpoints" / "iteration_001.pt").exists()
    assert (smoke_run / "report.md").exists()


def test_smoke_run_is_labelled_smoke_never_a_result(smoke_run):
    report = json.loads((smoke_run / "report.json").read_text(encoding="utf-8"))
    assert report["scope"] == "smoke"
    assert report["real_run_bar"]["met"] is False
    writeup = (smoke_run / "report.md").read_text(encoding="utf-8")
    assert "loop check" in writeup
    assert "never" in writeup and "result" in writeup


def test_smoke_run_records_expert_and_student_provenance(smoke_run):
    report = json.loads((smoke_run / "report.json").read_text(encoding="utf-8"))
    assert report["expert"] == "pure_pursuit"
    assert report["backend"] == "kinematic"
    assert report["student"]["model"] == "CILModel"
    assert report["student"]["imagenet_pretrained"] is False


def test_finished_run_cleans_up_its_sample_files(smoke_run):
    # The per-iteration sample archives exist to survive a crash; once the
    # report exists there is nothing to resume, and a lingering archive would
    # suggest the run did not finish (same rule as the ablation's partials).
    assert list((smoke_run / "data").glob("*.npz")) == []


def test_completed_run_refuses_to_be_overwritten(smoke_run):
    with pytest.raises(SystemExit, match="complete"):
        dagger.main(["--smoke", "--output-dir", str(smoke_run)])


def _crash_on_training_call(monkeypatch, crash_call: int):
    real_train = dagger._train
    calls = {"n": 0}

    def exploding(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == crash_call:
            raise KeyboardInterrupt
        return real_train(*args, **kwargs)

    monkeypatch.setattr(dagger, "_train", exploding)


def test_killed_mid_iteration_resumes_losing_at_most_that_iteration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dagger, "_build_model", _tiny_model_builder)
    args = ["--smoke", "--iterations", "3"]

    # The uninterrupted run, for comparison.
    straight = tmp_path / "straight"
    assert dagger.main([*args, "--output-dir", str(straight)]) == 0
    expected = _strip_timing(json.loads((straight / "report.json").read_text(encoding="utf-8")))

    # The same run killed inside iteration 2's training step.
    crashed = tmp_path / "crashed"
    with monkeypatch.context() as inner:
        _crash_on_training_call(inner, crash_call=3)
        with pytest.raises(KeyboardInterrupt):
            dagger.main([*args, "--output-dir", str(crashed)])
    assert not (crashed / "report.json").exists()
    survivors = sorted(p.name for p in (crashed / "checkpoints").glob("*.pt"))
    assert survivors == ["iteration_000.pt", "iteration_001.pt"]

    # Re-running completes the run without redoing the surviving iterations…
    mtimes = {p.name: p.stat().st_mtime_ns for p in (crashed / "checkpoints").glob("*.pt")}
    assert dagger.main([*args, "--output-dir", str(crashed)]) == 0
    for name, mtime in mtimes.items():
        assert (crashed / "checkpoints" / name).stat().st_mtime_ns == mtime

    # …and produces the numbers the uninterrupted run produced.
    resumed = _strip_timing(json.loads((crashed / "report.json").read_text(encoding="utf-8")))
    assert resumed == expected


def test_rerun_with_a_different_config_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(dagger, "_build_model", _tiny_model_builder)
    out = tmp_path / "run"
    with monkeypatch.context() as inner:
        _crash_on_training_call(inner, crash_call=2)
        with pytest.raises(KeyboardInterrupt):
            dagger.main(["--smoke", "--iterations", "3", "--output-dir", str(out)])

    # A resumed run trained under a different configuration would silently mix
    # two experiments into one artifact; the CLI must refuse instead.
    with pytest.raises(SystemExit, match="match"):
        dagger.main(["--smoke", "--iterations", "3", "--max-steps", "99",
                     "--output-dir", str(out)])
