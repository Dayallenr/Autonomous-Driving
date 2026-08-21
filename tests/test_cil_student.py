"""The CIL student as a registered Policy (issue #21).

What these tests pin is the scoring side's honesty, not driving quality: the
student is buildable through the same registry call as every other Policy,
refuses to exist without weights, fails eagerly on a missing or mismatched
checkpoint instead of burning a benchmark suite, and stamps a version derived
from the actual weight bytes into every result — so a score can never be
attributed to weights that did not produce it.
"""
from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from pathfinder.policies import POLICY_NAMES, CILStudentPolicy, build_policy
from pathfinder.runner import run_episode
from pathfinder.sim.base import EpisodeSpec
from pathfinder.sim.kinematic import KinematicSimulator
from tests.support import make_frame

# ─────────────────────────────────────────────────────────────────────────────
# Registration and refusal
# ─────────────────────────────────────────────────────────────────────────────


def test_the_student_is_buildable_by_name_from_the_registry(cil_checkpoint):
    assert CILStudentPolicy.NAME in POLICY_NAMES
    policy = build_policy(CILStudentPolicy.NAME, weights=cil_checkpoint)
    assert hasattr(policy, "plan")


def test_the_student_refuses_to_run_without_weights():
    """No default checkpoint path: a student that silently picks up whatever
    weights happen to lie around produces unattributable scores."""
    with pytest.raises(ValueError, match="weights"):
        build_policy(CILStudentPolicy.NAME)


def test_a_missing_checkpoint_fails_eagerly_and_names_the_fix(tmp_path):
    """At construction, not on the first plan: the runner turns mid-Episode
    exceptions into scored failures, so a lazy check would spend a whole
    suite producing zero-score Episodes instead of one error."""
    with pytest.raises(FileNotFoundError, match="dagger"):
        build_policy(CILStudentPolicy.NAME, weights=tmp_path / "absent.pt")


def test_a_mismatched_checkpoint_is_refused_with_a_named_fix(tmp_path):
    from pathfinder.planning.cil_model import CILModel

    torch.manual_seed(0)
    waypoints = CILModel(pretrained=False, output_mode="waypoints")
    path = tmp_path / "waypoints.pt"
    torch.save({"model": waypoints.state_dict()}, path)

    with pytest.raises(ValueError, match="control"):
        build_policy(CILStudentPolicy.NAME, weights=path)


def test_importing_the_registry_does_not_import_torch():
    """Mirrors the CARLA isolation rule: selecting any other Policy — and the
    whole orchestration layer that imports the registry — must work in an
    environment without torch installed."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pathfinder.policies; "
            "assert 'torch' not in sys.modules, 'policies imported torch'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ─────────────────────────────────────────────────────────────────────────────
# Weights version
# ─────────────────────────────────────────────────────────────────────────────


def test_the_weights_version_is_derived_from_the_file_bytes(cil_checkpoint, tmp_path):
    from pathfinder.planning.cil_model import CILModel

    policy = build_policy(CILStudentPolicy.NAME, weights=cil_checkpoint)
    assert policy.model_version.startswith(f"{CILStudentPolicy.NAME}@")

    again = build_policy(CILStudentPolicy.NAME, weights=cil_checkpoint)
    assert again.model_version == policy.model_version

    torch.manual_seed(99)
    other = CILModel(pretrained=False, output_mode="control")
    other_path = tmp_path / "other.pt"
    torch.save({"model": other.state_dict()}, other_path)
    assert (
        build_policy(CILStudentPolicy.NAME, weights=other_path).model_version
        != policy.model_version
    )


def test_a_bare_state_dict_also_loads(cil_checkpoint, tmp_path):
    """The DAgger CLI writes ``{"model": ..., "optimizer": ..., "row": ...}``,
    but the notebook saves a bare state dict; both are real weight sources."""
    payload = torch.load(cil_checkpoint, map_location="cpu", weights_only=True)
    path = tmp_path / "bare.pt"
    torch.save(payload["model"], path)
    assert hasattr(build_policy(CILStudentPolicy.NAME, weights=path), "plan")


# ─────────────────────────────────────────────────────────────────────────────
# Inference discipline
# ─────────────────────────────────────────────────────────────────────────────


def test_the_model_is_in_eval_mode_on_the_pinned_device(cil_checkpoint):
    policy = build_policy(CILStudentPolicy.NAME, weights=cil_checkpoint, device="cpu")
    assert policy.device == "cpu"
    assert policy.model.training is False
    assert {p.device.type for p in policy.model.parameters()} == {"cpu"}


def test_the_student_refuses_frames_without_pixels(cil_checkpoint):
    """The student's whole observation is pixels plus the route command; a
    non-rendering backend is a configuration mistake, not an empty scene."""
    policy = build_policy(CILStudentPolicy.NAME, weights=cil_checkpoint)
    with pytest.raises(ValueError, match="render"):
        policy.plan(make_frame(image=None))


# ─────────────────────────────────────────────────────────────────────────────
# It plugs into the existing loop
# ─────────────────────────────────────────────────────────────────────────────


def test_the_student_drives_a_scored_episode_and_every_frame_carries_its_version(
    cil_checkpoint,
):
    policy = build_policy(CILStudentPolicy.NAME, weights=cil_checkpoint)
    rows: list[dict] = []

    with KinematicSimulator(render=True) as simulator:
        score = run_episode(
            simulator,
            EpisodeSpec(episode_id="student-1", route_length_m=60.0, max_steps=25),
            policy,
            telemetry_sink=rows.append,
            model_version=policy.model_version,
        )

    assert score.status == "completed"
    assert score.frames > 0
    assert rows
    assert {row["model_version"] for row in rows} == {policy.model_version}
