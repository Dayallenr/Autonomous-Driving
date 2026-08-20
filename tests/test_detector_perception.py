"""
Detector-backed perception: the Detector's boxes become the scene the Policy
consumes.

Contract tests only. They drive the seam with a stub detector and assert the
plumbing — empty scenes, failure handling, provenance, determinism — never a
particular range value or detection count, because detection *quality* on the
kinematic renderer's untextured images is explicitly out of scope. One optional
test exercises the real trained weights when they are present locally.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from pathfinder.perception import DetectorPerception, PrivilegedPerception
from pathfinder.policies import ModularPolicy
from pathfinder.runner import PurePursuitPolicy, run_episode
from pathfinder.sim import EpisodeSpec, KinematicSimulator
from pathfinder.sim.base import Command, FrameState


def _frame(**overrides) -> FrameState:
    base = dict(
        frame_index=5,
        simulation_time=0.25,
        x=1.0,
        y=2.0,
        yaw_degrees=3.0,
        speed_mps=6.0,
        command=Command.FOLLOW_LANE,
        distance_travelled_m=12.0,
        nearest_object_m=7.3,
        detections=2,
    )
    base.update(overrides)
    return FrameState(**base)


class _StubDetector:
    """Satisfies the Detector protocol structurally, like the seam tests'
    perception stubs: no import of the protocol, no inheritance."""

    def __init__(self, boxes) -> None:
        self._boxes = boxes

    def detect(self, image):
        return self._boxes


def _image():
    return np.zeros((88, 200, 3), dtype=np.uint8)


def test_no_detections_reads_as_an_empty_scene():
    """Nothing seen is infinite range and zero detections — the value the
    speed governor already treats as an unobstructed road."""
    perception = DetectorPerception(_StubDetector([]))
    scene = perception.perceive(_frame(image=_image()))
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

    scene = perception.perceive(_frame(image=_image()))

    assert scene.nearest_object_m == pytest.approx(3.5)
    assert scene.detections == 2


class _ExplodingDetector:
    def detect(self, image):
        raise RuntimeError("CUDA device disappeared mid-frame")


def test_a_detector_failure_is_an_empty_scene_and_a_warning_not_an_abort(caplog):
    """Matches the runner's stance on peripheral failures: a perception crash
    must cost the frame its detections, never the Episode its result."""
    import logging

    perception = DetectorPerception(_ExplodingDetector())
    with caplog.at_level(logging.WARNING, logger="pathfinder.perception.detector"):
        scene = perception.perceive(_frame(image=_image()))

    assert scene == type(scene)()  # the empty scene, exactly as defaulted
    assert any("CUDA device disappeared" in record.message for record in caplog.records)


def test_a_frame_without_an_image_is_an_empty_scene_and_a_warning(caplog):
    """Detector-backed perception on a backend that is not rendering is a
    configuration mistake. It must be visible in the log and cost detections,
    not abort the Episode — and the Detector must never be handed None."""
    perception = DetectorPerception(_ExplodingDetector())
    with caplog.at_level(logging.WARNING, logger="pathfinder.perception.detector"):
        scene = perception.perceive(_frame(image=None))

    assert scene == type(scene)()
    assert any("no camera image" in record.message for record in caplog.records)


def test_detector_perception_provenance_is_stamped_on_every_control():
    """The ablation is meaningless unless a frame says which perception drove
    it; the name must differ from the privileged one so the two runs can never
    be conflated in telemetry."""
    policy = ModularPolicy(DetectorPerception(_StubDetector([])), PurePursuitPolicy())
    control = policy.plan(_frame(image=_image()))

    assert control.perception == "detector"
    assert control.perception != PrivilegedPerception.NAME


class _RecordingDetector:
    """Sees nothing, but records what it was shown — so the end-to-end test
    can prove real rendered frames flowed through the seam."""

    def __init__(self) -> None:
        self.image_shapes = []

    def detect(self, image):
        assert isinstance(image, np.ndarray)
        self.image_shapes.append(image.shape)
        return []


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


def test_one_seed_run_twice_is_identical_through_detector_perception():
    """Reproducibility must survive perception entering the loop. With the
    Detector stubbed deterministic, any divergence here is the seam's fault;
    the real weights' determinism is the optional test's job."""
    box_at_seven_metres = (10.0, 40.0, 40.0, 66.0)

    def run_once():
        spec = EpisodeSpec(
            episode_id="detector-repeat", route_length_m=120.0, max_steps=300, seed=11
        )
        policy = ModularPolicy(
            DetectorPerception(_StubDetector([box_at_seven_metres])),
            PurePursuitPolicy(),
        )
        with KinematicSimulator(render=True) as simulator:
            return run_episode(simulator, spec, policy)

    first, second = run_once(), run_once()

    assert first.driving_score == second.driving_score
    assert first.route_completion == second.route_completion
    assert first.infractions == second.infractions
    assert first.frames == second.frames
    assert first.termination_reason == second.termination_reason


def _trained_weights_available() -> bool:
    from pathlib import Path

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return Path("results/perception/yolov8m/weights/best.pt").exists()


@pytest.mark.skipif(
    not _trained_weights_available(),
    reason="trained Detector weights or ultralytics not present on this machine",
)
def test_real_weights_drive_the_seam_when_present_locally():
    """The one test that touches the trained Detector. It asserts the loading
    contract and determinism, and nothing about what the Detector finds — the
    kinematic renderer's untextured frames are far outside KITTI's domain, so
    zero detections is an acceptable and likely answer."""
    from pathfinder.perception import YoloDetector
    from pathfinder.perception.detector import load_yolo_model

    weights = "results/perception/yolov8m/weights/best.pt"

    # Loaded once per process: the cache hands back the very same model.
    first = load_yolo_model(weights, "cpu")
    second = load_yolo_model(weights, "cpu")
    assert first is second
    # Evaluation mode, pinned by the loader rather than left to ultralytics.
    assert first.model.training is False

    with KinematicSimulator(render=True) as simulator:
        state = simulator.reset(
            EpisodeSpec(episode_id="real-weights", route_length_m=60.0, max_steps=10, seed=7)
        )

    perception = DetectorPerception(YoloDetector(weights, device="cpu"))
    scene_a = perception.perceive(state)
    scene_b = perception.perceive(state)

    # Determinism: the same frame perceived twice is the same scene.
    assert scene_a == scene_b
    # Contract only — no particular value or count.
    assert scene_a.detections >= 0
    assert scene_a.nearest_object_m > 0 or math.isinf(scene_a.nearest_object_m)
