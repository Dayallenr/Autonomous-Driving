"""
Detector-backed perception: the Detector's boxes become the scene the Policy
consumes.

Contract tests only. They drive the seam with a stub detector and assert the
plumbing — empty scenes, failure handling, provenance, determinism, and that
stubbed boxes flow through the geometric estimator — never anything about what
the *trained* Detector finds, because detection quality on the kinematic
renderer's untextured images is explicitly out of scope. One optional test
exercises the real trained weights when they are present locally.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from pathfinder.perception import DetectorPerception, PerceivedScene, PrivilegedPerception
from pathfinder.perception.detector import TRAINED_WEIGHTS
from pathfinder.policies import ModularPolicy
from pathfinder.runner import PurePursuitPolicy, run_episode
from pathfinder.sim import EpisodeSpec, KinematicSimulator
from tests.support import make_frame


class _StubDetector:
    """Satisfies the Detector protocol structurally, like the seam tests'
    perception stubs: no import of the protocol, no inheritance."""

    def __init__(self, boxes) -> None:
        self._boxes = boxes

    def detect(self, image):
        return self._boxes


class _ExplodingDetector:
    def detect(self, image):
        raise RuntimeError("CUDA device disappeared mid-frame")


class _RecordingDetector:
    """Sees nothing, but records what it was shown — so the end-to-end test
    can prove real rendered frames flowed through the seam."""

    def __init__(self) -> None:
        self.image_shapes = []

    def detect(self, image):
        assert isinstance(image, np.ndarray)
        self.image_shapes.append(image.shape)
        return []


def _image():
    return np.zeros((88, 200, 3), dtype=np.uint8)


def test_no_detections_reads_as_an_empty_scene():
    """Nothing seen is infinite range and zero detections — the value the
    speed governor already treats as an unobstructed road."""
    perception = DetectorPerception(_StubDetector([]))
    scene = perception.perceive(make_frame(image=_image()))
    assert math.isinf(scene.nearest_object_m)
    assert scene.detections == 0


def test_nearest_range_comes_from_the_geometric_estimator():
    """Wiring, not quality: the boxes are stubbed, and the expected range is
    the worked example from the geometry module's spec — the kinematic camera
    (f=110 px, h=1.4 m, horizon at row 44) puts a box bottom at the frame's
    bottom edge (row 88) exactly at the 3.5 m floor."""
    at_the_floor = (80.0, 60.0, 120.0, 88.0)
    further_away = (10.0, 40.0, 40.0, 66.0)  # bottom at row 66 -> 7.0 m
    perception = DetectorPerception(_StubDetector([further_away, at_the_floor]))

    scene = perception.perceive(make_frame(image=_image()))

    assert scene.nearest_object_m == pytest.approx(3.5)
    assert scene.detections == 2


def test_a_detector_failure_is_an_empty_scene_and_a_warning_not_an_abort(caplog):
    """Matches the runner's stance on peripheral failures: a perception crash
    must cost the frame its detections, never the Episode its result."""
    perception = DetectorPerception(_ExplodingDetector())
    with caplog.at_level(logging.WARNING, logger="pathfinder.perception.detector"):
        scene = perception.perceive(make_frame(image=_image()))

    assert scene == PerceivedScene()
    assert any("CUDA device disappeared" in record.message for record in caplog.records)


def test_a_frame_without_an_image_is_an_empty_scene_and_a_warning(caplog):
    """Detector-backed perception on a backend that is not rendering is a
    configuration mistake. It must be visible in the log and cost detections,
    not abort the Episode — and the Detector must never be handed None."""
    perception = DetectorPerception(_ExplodingDetector())
    with caplog.at_level(logging.WARNING, logger="pathfinder.perception.detector"):
        scene = perception.perceive(make_frame(image=None))

    assert scene == PerceivedScene()
    assert any("no camera image" in record.message for record in caplog.records)


def test_detector_perception_provenance_is_stamped_on_every_control():
    """The ablation is meaningless unless a frame says which perception drove
    it; the name must differ from the privileged one so the two runs can never
    be conflated in telemetry."""
    policy = ModularPolicy(DetectorPerception(_StubDetector([])), PurePursuitPolicy())
    control = policy.plan(make_frame(image=_image()))

    assert control.perception == "detector"
    assert control.perception != PrivilegedPerception.NAME


def test_an_episode_completes_end_to_end_on_the_rendering_kinematic_backend():
    detector = _RecordingDetector()
    rows: list[dict] = []
    spec = EpisodeSpec(episode_id="detector-e2e", route_length_m=60.0, max_steps=120, seed=3)

    with KinematicSimulator(render=True) as simulator:
        result = run_episode(
            simulator,
            spec,
            ModularPolicy(DetectorPerception(detector), PurePursuitPolicy()),
            telemetry_sink=rows.append,
        )

    # The Episode ran to a verdict, and every frame's camera image reached the
    # Detector at the renderer's documented geometry.
    assert result.frames > 0
    assert len(detector.image_shapes) == result.frames
    assert set(detector.image_shapes) == {(88, 200, 3)}
    assert {row["perception"] for row in rows} == {"detector"}


def _run_seeded_episode(detector) -> object:
    spec = EpisodeSpec(
        episode_id="detector-repeat", route_length_m=120.0, max_steps=300, seed=11
    )
    policy = ModularPolicy(DetectorPerception(detector), PurePursuitPolicy())
    with KinematicSimulator(render=True) as simulator:
        return run_episode(simulator, spec, policy)


def _assert_identical_results(first, second) -> None:
    assert first.driving_score == second.driving_score
    assert first.route_completion == second.route_completion
    assert first.infractions == second.infractions
    assert first.frames == second.frames
    assert first.termination_reason == second.termination_reason


def test_one_seed_run_twice_is_identical_through_detector_perception():
    """Reproducibility must survive perception entering the loop. With the
    Detector stubbed deterministic, any divergence here is the seam's fault;
    the real weights' determinism is the optional test's job."""
    box_at_seven_metres = (10.0, 40.0, 40.0, 66.0)
    first = _run_seeded_episode(_StubDetector([box_at_seven_metres]))
    second = _run_seeded_episode(_StubDetector([box_at_seven_metres]))
    _assert_identical_results(first, second)


def _trained_weights_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return TRAINED_WEIGHTS.exists()


@pytest.mark.skipif(
    not _trained_weights_available(),
    reason="trained Detector weights or ultralytics not present on this machine",
)
def test_real_weights_drive_the_seam_when_present_locally():
    """The one test that touches the trained Detector. It asserts the loading
    contract and within-process determinism — one seed driven twice through
    the real weights must be identical — and nothing about what the Detector
    finds: the kinematic renderer's untextured frames are far outside KITTI's
    domain, so zero detections is an acceptable and likely answer. Determinism
    *across* processes and machines is the CARLA validation's job; a
    same-process check cannot see thread-count or device effects."""
    from pathfinder.perception import YoloDetector
    from pathfinder.perception.detector import load_yolo_model

    # Loaded once per process: the cache hands back the very same object.
    first = load_yolo_model(TRAINED_WEIGHTS, "cpu")
    second = load_yolo_model(TRAINED_WEIGHTS, "cpu")
    assert first is second
    # Evaluation mode, pinned by the loader rather than left to ultralytics.
    assert first.model.training is False

    detector = YoloDetector(TRAINED_WEIGHTS, device="cpu")
    _assert_identical_results(
        _run_short_real_episode(detector), _run_short_real_episode(detector)
    )


def _run_short_real_episode(detector) -> object:
    """Short on purpose: real inference runs on every frame, and this test also
    runs in CI on CPU. Twenty frames is enough to prove the loop is stable and
    repeatable with the network in it."""
    spec = EpisodeSpec(episode_id="real-weights", route_length_m=30.0, max_steps=20, seed=7)
    policy = ModularPolicy(DetectorPerception(detector), PurePursuitPolicy())
    with KinematicSimulator(render=True) as simulator:
        return run_episode(simulator, spec, policy)
