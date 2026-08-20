"""
The ablation runner: one command, two perceptions, one measured difference.

Everything here runs on the kinematic backend with stub perceptions, so it
proves the *mechanism* — report shape, provenance, seed identity, labelling —
on a machine that cannot run CARLA. Deliberately absent: any assertion about
the magnitude of the difference. On this backend the numbers are meaningless,
and a test that pinned them would enshrine a number the project must never
quote.
"""
from __future__ import annotations

import json

import pytest

from pathfinder.ablation import run_ablation
from pathfinder.perception import PerceivedScene, PrivilegedPerception
from pathfinder.sim import EpisodeSpec, KinematicSimulator
from pathfinder.sim.base import FrameState


class _BlindPerception:
    """A candidate that never sees anything — the cheapest possible stand-in
    for the Detector arm. Satisfies the Perception protocol structurally."""

    NAME = "stub_blind"

    def perceive(self, state: FrameState) -> PerceivedScene:
        return PerceivedScene()


def _specs(count: int = 2) -> list[EpisodeSpec]:
    return [
        EpisodeSpec(
            episode_id=f"abl-{index:02d}",
            route_length_m=80.0,
            max_steps=40,
            seed=100 + index,
        )
        for index in range(count)
    ]


@pytest.fixture(scope="module")
def report() -> dict:
    with KinematicSimulator() as simulator:
        return run_ablation(simulator, _specs(), candidate=_BlindPerception())


def test_report_carries_both_aggregate_scores_and_their_difference(report):
    """The point of the report: two driving scores and what separates them,
    where the difference is the arithmetic of the two aggregates — asserted as
    an identity, never as a magnitude."""
    baseline = report["baseline"]["summary"]
    candidate = report["candidate"]["summary"]

    for summary in (baseline, candidate):
        assert isinstance(summary["driving_score"], float)
        assert isinstance(summary["route_completion"], float)

    difference = report["difference"]
    assert difference["driving_score"] == pytest.approx(
        baseline["driving_score"] - candidate["driving_score"], abs=0.01
    )
    assert difference["route_completion"] == pytest.approx(
        baseline["route_completion"] - candidate["route_completion"], abs=1e-4
    )
    assert difference["infraction_penalty"] == pytest.approx(
        baseline["infraction_penalty"] - candidate["infraction_penalty"], abs=1e-4
    )


def test_arms_run_identical_seeds_and_specs_differing_only_in_perception(report):
    """The comparison isolates perception: same episode ids, same seeds, in
    the same order, under two different perception implementations."""
    specs = _specs()
    assert report["episodes"] == [spec.to_dict() for spec in specs]

    for arm in ("baseline", "candidate"):
        episodes = report[arm]["episodes"]
        assert [e["episode_id"] for e in episodes] == [s.episode_id for s in specs]
        assert [e["seed"] for e in episodes] == [s.seed for s in specs]

    assert report["baseline"]["perception"] != report["candidate"]["perception"]


def test_each_episode_records_its_perception_provenance(report):
    """Provenance is observed from the frames that actually ran, per episode —
    a result can never be attributed to a perception that did not produce it."""
    for episode in report["baseline"]["episodes"]:
        assert episode["perception"] == PrivilegedPerception.NAME
    for episode in report["candidate"]["episodes"]:
        assert episode["perception"] == _BlindPerception.NAME
    assert report["baseline"]["perception"] == PrivilegedPerception.NAME
    assert report["candidate"]["perception"] == _BlindPerception.NAME


def test_perception_latency_is_measured_per_arm(report):
    for arm in ("baseline", "candidate"):
        assert report[arm]["mean_perception_latency_ms"] >= 0.0


def test_kinematic_reports_are_labelled_pipeline_only(report):
    """A kinematic number quoted as driving quality would be the exact
    dishonesty this project exists to avoid; the report must say so itself."""
    assert report["backend"] == "kinematic"
    assert report["scope"] == "pipeline-only"
    assert "not driving quality" in report["scope_note"]


def test_report_states_the_perception_boundary(report):
    """Every write-up must say what was perceived and what stayed privileged;
    the report carries the sentence so no downstream author has to remember."""
    assert (
        report["perception_boundary"]
        == "perceived obstacles, privileged localization and traffic lights"
    )


def test_report_is_json_serialisable(report):
    """The report is the artifact; if it cannot be written to disk it does
    not exist."""
    parsed = json.loads(json.dumps(report))
    assert parsed["kind"] == "perception_ablation"


def test_identical_perception_arms_are_rejected():
    """Two arms with the same provenance name would produce a difference of
    zero by construction and a report that measures nothing."""
    with KinematicSimulator() as simulator:
        with pytest.raises(ValueError, match="differ"):
            run_ablation(simulator, _specs(1), candidate=PrivilegedPerception())


def test_empty_spec_list_is_rejected():
    with KinematicSimulator() as simulator:
        with pytest.raises(ValueError, match="episode"):
            run_ablation(simulator, [], candidate=_BlindPerception())


def test_cli_writes_a_labelled_report(tmp_path, monkeypatch):
    """The one command, end to end: build specs, run both arms, write JSON.
    The Detector is stubbed at the CLI's own construction seam so this runs
    on a machine with no weights."""
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
    written = json.loads(output.read_text())
    assert written["scope"] == "pipeline-only"
    assert written["candidate"]["perception"] == _BlindPerception.NAME
    assert written["difference"]["driving_score"] is not None


def test_controller_is_labelled_with_its_registry_name(report):
    """Telemetry and reports must agree with the Policy registry on what drove
    ('pure_pursuit'), or one controller splits into two names in the warehouse."""
    assert report["controller"] == "pure_pursuit"


def test_no_failed_episodes_reads_as_an_empty_flag_list(report):
    assert report["difference"]["failed_episodes"] == []


def test_a_perception_built_for_the_wrong_camera_is_rejected():
    """A camera mismatch mis-scales every range silently; the pairing must
    fail at startup, not surface as a wrong number hours later."""
    from pathfinder.perception.detector import DetectorPerception
    from pathfinder.sim.carla_backend import CARLA_CAMERA

    class _NoBoxes:
        def detect(self, image):
            return []

    wrong_camera = DetectorPerception(_NoBoxes(), camera=CARLA_CAMERA)
    with KinematicSimulator() as simulator:
        with pytest.raises(ValueError, match="camera"):
            run_ablation(simulator, _specs(1), candidate=wrong_camera)


def test_failed_episodes_are_flagged_never_folded_into_the_difference():
    """A mid-run failure scores the failure, not the perception. The report
    must name the affected episodes and carry their statuses on the rows, so
    a failure delta can never be quoted as perception cost."""
    from pathfinder.runner import PurePursuitPolicy

    built = {"count": 0}

    def flaky_controller():
        built["count"] += 1
        if built["count"] in (2, 4):  # second episode of each arm
            class _Crashes(PurePursuitPolicy):
                frames = 0

                def plan(self, state):
                    self.frames += 1
                    if self.frames > 5:
                        raise RuntimeError("injected mid-episode failure")
                    return super().plan(state)

            return _Crashes()
        return PurePursuitPolicy()

    with KinematicSimulator() as simulator:
        result = run_ablation(
            simulator,
            _specs(2),
            candidate=_BlindPerception(),
            controller_factory=flaky_controller,
        )

    assert result["difference"]["failed_episodes"] == ["abl-01"]
    rows = {row["episode_id"]: row for row in result["difference"]["per_episode_driving_score"]}
    assert rows["abl-01"]["baseline_status"] == "failed"
    assert rows["abl-01"]["candidate_status"] == "failed"
    assert "baseline_status" not in rows["abl-00"]
    assert "failed" in result["difference"]["definition"]


def test_on_episode_reports_every_finished_episode_as_it_lands():
    """The persistence seam: a caller checkpointing through this callback
    loses at most one episode to a crash, not the whole run."""
    calls: list[tuple[str, str]] = []

    with KinematicSimulator() as simulator:
        run_ablation(
            simulator,
            _specs(2),
            candidate=_BlindPerception(),
            on_episode=lambda name, score: calls.append((name, score.episode_id)),
        )

    assert calls == [
        (PrivilegedPerception.NAME, "abl-00"),
        (PrivilegedPerception.NAME, "abl-01"),
        (_BlindPerception.NAME, "abl-00"),
        (_BlindPerception.NAME, "abl-01"),
    ]


def test_an_episode_that_fails_before_its_first_frame_is_still_reported():
    """A crash on the very first plan leaves nothing to attribute; the Episode
    is recorded with unobserved provenance and flagged, and the arm survives —
    one crash must not abort a whole benchmark run."""
    from pathfinder.runner import PurePursuitPolicy

    built = {"count": 0}

    def flaky_controller():
        built["count"] += 1
        if built["count"] in (2, 4):
            class _CrashesImmediately:
                def plan(self, state):
                    raise RuntimeError("injected first-frame failure")

            return _CrashesImmediately()
        return PurePursuitPolicy()

    with KinematicSimulator() as simulator:
        result = run_ablation(
            simulator,
            _specs(2),
            candidate=_BlindPerception(),
            controller_factory=flaky_controller,
        )

    assert result["difference"]["failed_episodes"] == ["abl-01"]
    for arm in ("baseline", "candidate"):
        flagged = result[arm]["episodes"][1]
        assert flagged["perception"] == "unobserved"
        assert flagged["status"] == "failed"
