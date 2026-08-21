"""The three-column comparison suite (issue #21), proven CARLA-free.

The comparison's job is attribution: three Policies, one seeded suite, one
artifact. What these tests pin is that the mechanism cannot lie — an arm
cannot claim a Policy or a weights version it does not hold, the reference
ceiling can be skipped only with a recorded reason, failed episodes are
flagged rather than dropped, and the CLI's default suite is byte-identical to
the perception ablation's, so the two artifacts' numbers are comparable.

The reference arm here is a stub declaring the behaviour agent's name, the
same stance as ``test_policies``: the agent is CARLA's, so what is testable
without a server is everything about how the suite treats it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pathfinder.comparison as comparison
from pathfinder.comparison import (
    ROLE_FLOOR,
    ROLE_REFERENCE,
    ROLE_STUDENT,
    ComparisonArm,
    run_comparison,
)
from pathfinder.orchestration import build_episode_specs
from pathfinder.policies import (
    PURE_PURSUIT,
    CarlaBehaviorAgentPolicy,
    CILStudentPolicy,
    build_policy,
)
from pathfinder.runner import ControlOutput, PurePursuitPolicy
from pathfinder.sim.kinematic import KinematicSimulator

REFERENCE = CarlaBehaviorAgentPolicy.NAME


class StubReferencePolicy:
    """Drives blind but declares the behaviour agent's registry name, letting
    the reference column run on a machine with no CARLA."""

    NAME = REFERENCE

    def plan(self, state) -> ControlOutput:
        return ControlOutput(throttle=0.3, steer=0.0, brake=0.0, latency_ms=0.1)


class ExplodingPolicy:
    NAME = REFERENCE

    def plan(self, state) -> ControlOutput:
        raise RuntimeError("deliberate mid-episode failure")


@pytest.fixture(scope="module")
def student(cil_checkpoint):
    return build_policy(CILStudentPolicy.NAME, weights=cil_checkpoint)


def tiny_specs(count: int = 2):
    return build_episode_specs(
        count=count, route_length_m=60.0, base_seed=5, max_steps=20
    )


def three_arms(student, *, reference_policy=None, reference_skip: str = ""):
    reference = (
        ComparisonArm(
            role=ROLE_REFERENCE,
            policy_name=REFERENCE,
            model_version=REFERENCE,
            policy=None,
            skip_reason=reference_skip,
        )
        if reference_policy is None and reference_skip
        else ComparisonArm(
            role=ROLE_REFERENCE,
            policy_name=REFERENCE,
            model_version=REFERENCE,
            policy=reference_policy or StubReferencePolicy(),
        )
    )
    return [
        ComparisonArm(
            role=ROLE_FLOOR,
            policy_name=PURE_PURSUIT,
            model_version=PURE_PURSUIT,
            policy=PurePursuitPolicy(),
        ),
        ComparisonArm(
            role=ROLE_STUDENT,
            policy_name=CILStudentPolicy.NAME,
            model_version=student.model_version,
            policy=student,
            weights=str(student.weights_path),
        ),
        reference,
    ]


def run(student, **kwargs):
    with KinematicSimulator(render=True) as simulator:
        return run_comparison(
            simulator,
            kwargs.pop("specs", tiny_specs()),
            kwargs.pop("arms", three_arms(student)),
            **kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# The report
# ─────────────────────────────────────────────────────────────────────────────


def test_all_three_arms_are_scored_over_identical_specs(student):
    specs = tiny_specs()
    report = run(student, specs=specs)

    assert report["kind"] == "policy_comparison"
    assert [arm["role"] for arm in report["arms"]] == [
        ROLE_FLOOR,
        ROLE_STUDENT,
        ROLE_REFERENCE,
    ]
    assert report["episodes"] == [spec.to_dict() for spec in specs]
    for arm in report["arms"]:
        assert len(arm["episodes"]) == len(specs)
        assert [row["episode_id"] for row in arm["episodes"]] == [
            spec.episode_id for spec in specs
        ]
        assert arm["summary"]["episodes"] == len(specs)


def test_every_arm_records_what_drove_it(student):
    report = run(student)
    by_role = {arm["role"]: arm for arm in report["arms"]}

    assert by_role[ROLE_FLOOR]["model_version"] == PURE_PURSUIT
    assert by_role[ROLE_STUDENT]["model_version"] == student.model_version
    assert student.model_version.startswith("cil_student@")
    assert by_role[ROLE_STUDENT]["weights"] == str(student.weights_path)
    assert by_role[ROLE_REFERENCE]["not_project_work"] is True
    assert by_role[ROLE_FLOOR]["not_project_work"] is False
    assert by_role[ROLE_STUDENT]["not_project_work"] is False


def test_the_kinematic_report_labels_itself_pipeline_only(student):
    report = run(student)
    assert report["backend"] == "kinematic"
    assert report["scope"] == "pipeline-only"
    assert "never be quoted" in report["scope_note"]
    assert report["student_observation_boundary"] == comparison.STUDENT_OBSERVATION_BOUNDARY


def test_only_carla_earns_the_driving_quality_scope():
    assert comparison._scope("carla")[0] == "driving-quality"
    assert comparison._scope("kinematic")[0] == "pipeline-only"
    assert comparison._scope("some_future_backend")[0] == "pipeline-only"


# ─────────────────────────────────────────────────────────────────────────────
# Skipping the reference ceiling
# ─────────────────────────────────────────────────────────────────────────────


def test_the_reference_may_be_skipped_with_a_recorded_reason(student):
    report = run(
        student,
        arms=three_arms(student, reference_skip="needs the CARLA backend"),
    )
    reference = report["arms"][2]
    assert reference["skipped"] is True
    assert reference["skip_reason"] == "needs the CARLA backend"
    assert "summary" not in reference
    # The two runnable columns are still fully scored.
    for arm in report["arms"][:2]:
        assert arm["summary"]["episodes"] == 2


def test_a_skipped_arm_needs_a_reason(student):
    arms = three_arms(student)
    arms[2] = ComparisonArm(
        role=ROLE_REFERENCE, policy_name=REFERENCE, model_version=REFERENCE, policy=None
    )
    with pytest.raises(ValueError, match="reason"):
        run(student, arms=arms)


def test_floor_and_student_cannot_be_skipped(student):
    for index, name in ((0, PURE_PURSUIT), (1, CILStudentPolicy.NAME)):
        arms = three_arms(student)
        arms[index] = ComparisonArm(
            role=arms[index].role,
            policy_name=name,
            model_version=arms[index].model_version,
            policy=None,
            skip_reason="tempting shortcut",
        )
        with pytest.raises(ValueError, match="reference"):
            run(student, arms=arms)


# ─────────────────────────────────────────────────────────────────────────────
# Attribution cannot be faked
# ─────────────────────────────────────────────────────────────────────────────


def test_the_three_roles_are_required_exactly_once(student):
    with pytest.raises(ValueError, match="role"):
        run(student, arms=three_arms(student)[:2])

    arms = three_arms(student)
    arms[2] = ComparisonArm(
        role=ROLE_FLOOR,
        policy_name=REFERENCE,
        model_version=REFERENCE,
        policy=StubReferencePolicy(),
    )
    with pytest.raises(ValueError, match="role"):
        run(student, arms=arms)


def test_an_arm_cannot_claim_a_policy_it_does_not_hold(student):
    arms = three_arms(student)
    arms[2] = ComparisonArm(
        role=ROLE_REFERENCE,
        policy_name=REFERENCE,
        model_version=REFERENCE,
        policy=PurePursuitPolicy(),  # declares NAME "pure_pursuit"
    )
    with pytest.raises(ValueError, match="declares"):
        run(student, arms=arms)


def test_an_arm_cannot_claim_a_weights_version_it_does_not_hold(student):
    arms = three_arms(student)
    arms[1] = ComparisonArm(
        role=ROLE_STUDENT,
        policy_name=CILStudentPolicy.NAME,
        model_version="cil_student@000000000000",
        policy=student,
        weights=str(student.weights_path),
    )
    with pytest.raises(ValueError, match="version"):
        run(student, arms=arms)


def test_two_arms_cannot_share_a_result_label(student):
    arms = three_arms(student)
    arms[2] = ComparisonArm(
        role=ROLE_REFERENCE,
        policy_name=PURE_PURSUIT,
        model_version=PURE_PURSUIT,
        policy=PurePursuitPolicy(),
    )
    with pytest.raises(ValueError, match="label"):
        run(student, arms=arms)


def test_an_empty_suite_is_refused(student):
    with pytest.raises(ValueError, match="episode"):
        run(student, specs=[])


# ─────────────────────────────────────────────────────────────────────────────
# Failures and checkpointing
# ─────────────────────────────────────────────────────────────────────────────


def test_failed_episodes_are_flagged_not_dropped(student):
    report = run(student, arms=three_arms(student, reference_policy=ExplodingPolicy()))
    reference = report["arms"][2]
    assert reference["summary"]["failures"] == 2
    assert report["failed_episodes"] == [spec.episode_id for spec in tiny_specs()]
    # The failed rows are still present, labelled by their status.
    assert {row["status"] for row in reference["episodes"]} == {"failed"}


def test_each_finished_episode_is_checkpointed_in_run_order(student):
    calls: list[tuple[str, str]] = []
    run(
        student,
        on_episode=lambda version, score: calls.append((version, score.episode_id)),
    )
    versions = [version for version, _ in calls]
    assert versions == (
        [PURE_PURSUIT] * 2 + [student.model_version] * 2 + [REFERENCE] * 2
    )
    assert [episode_id for _, episode_id in calls] == [
        spec.episode_id for spec in tiny_specs()
    ] * 3


# ─────────────────────────────────────────────────────────────────────────────
# The CLI
# ─────────────────────────────────────────────────────────────────────────────


def test_the_default_suite_is_the_perception_ablations_exact_suite():
    """The whole point of sharing the suite is that the two artifacts' numbers
    are comparable; the pin is against the real CARLA ablation artifact."""
    recorded = json.loads(
        (Path("results/ablation/carla_report.json")).read_text(encoding="utf-8")
    )["episodes"]
    specs = build_episode_specs(**comparison.DEFAULT_SUITE)
    assert [spec.to_dict() for spec in specs] == recorded


def test_the_cli_runs_end_to_end_on_kinematic_with_untrained_weights(
    cil_checkpoint, tmp_path
):
    output = tmp_path / "report.json"
    assert (
        comparison.main(
            [
                "--episodes", "2",
                "--route-length-m", "60",
                "--max-steps", "20",
                "--weights", str(cil_checkpoint),
                "--output", str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["kind"] == "policy_comparison"
    assert report["scope"] == "pipeline-only"
    # On the kinematic backend the behaviour agent cannot drive; the CLI must
    # record why rather than silently render a two-column comparison.
    reference = report["arms"][2]
    assert reference["skipped"] is True
    assert "CARLA" in reference["skip_reason"]
    # Both artifacts land together, and the partial file is gone.
    assert output.with_suffix(".md").exists()
    assert not output.with_suffix(".partial.jsonl").exists()


def test_the_cli_refuses_to_run_without_weights(capsys):
    with pytest.raises(SystemExit):
        comparison.main(["--episodes", "1"])
    assert "--weights" in capsys.readouterr().err
