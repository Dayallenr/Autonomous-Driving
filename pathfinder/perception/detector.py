"""
Detector-backed perception: the trained Detector's boxes become the scene.

This is the measured arm of the ground-truth-versus-Detector ablation. The
Detector runs over the frame's camera image, each detection's range comes from
:func:`~pathfinder.perception.geometry.range_from_box`, and the nearest one is
what the Policy sees. Everything the privileged implementation forwards
perfectly, this one has to earn from pixels — the gap between the two driving
scores is the cost of imperfect perception.

The Detector sits behind its own seam, :class:`Detector`, for the same reason
perception itself does: the contract — empty scenes, failure handling,
provenance, determinism — is testable with a stub on a machine that has no
weights, and the weights are a configuration choice rather than a hard
dependency of the driving loop.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Protocol

from pathfinder.perception.base import PerceivedScene
from pathfinder.perception.geometry import (
    KINEMATIC_CAMERA,
    Box,
    CameraGeometry,
    range_from_box,
)
from pathfinder.sim.base import FrameState

logger = logging.getLogger(__name__)

__all__ = ["Detector", "DetectorPerception", "TRAINED_WEIGHTS", "YoloDetector", "load_yolo_model"]

#: Where the honestly-trained YOLOv8m checkpoint lives in this repo (the
#: sequence-disjoint split; mAP@0.5 = 0.475). Relative to the repo root, so it
#: resolves on any machine that has pulled the weights.
TRAINED_WEIGHTS = Path("results/perception/yolov8m/weights/best.pt")

#: One model per (weights, device) per process. Loading a YOLO checkpoint takes
#: seconds and hundreds of megabytes; doing it per frame — or even per Episode —
#: would dominate every latency figure and thrash memory.
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def load_yolo_model(weights: str | Path, device: str):
    """Load an Ultralytics checkpoint once per process.

    Repeated calls with the same weights and device return the *same* model
    object — that identity is the public contract, and it is what makes "the
    Detector is loaded once per process" checkable rather than hoped.

    The model comes back pinned to ``device`` and in evaluation mode. Set
    explicitly here instead of trusting ultralytics' predict path, because
    reproducibility of every downstream driving score rests on it.

    Raises:
        FileNotFoundError: If the checkpoint is missing. Eager and loud — the
            runner turns mid-Episode exceptions into scored failures, so a lazy
            check would spend a whole benchmark producing zero-score Episodes
            instead of one error.
        ImportError: If ultralytics is not installed.
    """
    path = Path(weights)
    key = (str(path.resolve()), device)
    model = _MODEL_CACHE.get(key)
    if model is None:
        if not path.exists():
            raise FileNotFoundError(
                f"Detector weights not found at {path}. Train them with "
                "scripts/train_detector.py, or point at an existing checkpoint."
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError("ultralytics is required: pip install ultralytics") from error
        model = YOLO(str(path))
        model.to(device)
        model.model.eval()
        _MODEL_CACHE[key] = model
        logger.info("Detector loaded from %s onto %s", path, device)
    return model


class Detector(Protocol):
    """What perception needs from the Detector: boxes in a frame. Structural,
    matching the Perception and Policy protocols, so a test stub satisfies it
    without importing or inheriting anything."""

    def detect(self, image) -> list[Box]: ...


class DetectorPerception:
    """Perception that reports what the Detector found in the camera image."""

    #: Provenance identifier, carried into telemetry on every frame this
    #: implementation produces — the other arm of the ablation from
    #: ``PrivilegedPerception.NAME``.
    NAME = "detector"

    def __init__(
        self, detector: Detector, camera: CameraGeometry = KINEMATIC_CAMERA
    ) -> None:
        self._detector = detector
        self._camera = camera

    def perceive(self, state: FrameState) -> PerceivedScene:
        if state.image is None:
            # The backend is not rendering — a configuration mistake, since
            # this perception has nothing to perceive with. Logged rather than
            # raised so a misconfigured Episode still finishes and shows the
            # problem in its telemetry instead of dying mid-run.
            logger.warning(
                "Detector-backed perception got no camera image on frame %d; "
                "is the backend rendering? Reporting an empty scene.",
                state.frame_index,
            )
            return PerceivedScene()
        try:
            boxes = self._detector.detect(state.image)
        except Exception as error:
            # A perception failure costs the frame its detections, never the
            # Episode its result — the same stance the runner takes on
            # peripheral failures. The empty scene reads as an unobstructed
            # road, which the log line exists to make diagnosable.
            logger.warning(
                "Detector failed on frame %d, reporting an empty scene: %s",
                state.frame_index,
                error,
            )
            return PerceivedScene()
        # min() over per-box estimates; a degenerate box reads as inf from
        # range_from_box and so can never masquerade as the nearest obstacle.
        nearest = min(
            (range_from_box(box, self._camera) for box in boxes),
            default=math.inf,
        )
        return PerceivedScene(nearest_object_m=nearest, detections=len(boxes))


class YoloDetector:
    """The trained Detector behind the :class:`Detector` seam.

    Args:
        weights: Checkpoint path; defaults to the repo's trained YOLOv8m.
        device: Where inference runs. Pinned at construction — never chosen
            per frame — because a device that silently changes mid-benchmark
            changes the numbers. Default ``cpu`` runs identically on every
            machine; pass ``cuda`` explicitly on the GPU box.
        confidence: Detection threshold, pinned for the same reason. The
            ultralytics default, made explicit so a library upgrade cannot
            silently move it.

    Inference runs with test-time augmentation off: TTA trades determinism and
    latency for accuracy, and this seam's contract is determinism first.
    """

    def __init__(
        self,
        weights: str | Path = TRAINED_WEIGHTS,
        *,
        device: str = "cpu",
        confidence: float = 0.25,
    ) -> None:
        self._model = load_yolo_model(weights, device)
        self._device = device
        self._confidence = confidence

    def detect(self, image) -> list[Box]:
        import numpy as np

        # FrameState images are RGB (both backends document this); ultralytics
        # assumes ndarray input is BGR, so flip channels or every red car turns
        # blue on the way into the Detector.
        bgr = np.ascontiguousarray(np.asarray(image)[..., ::-1])
        results = self._model.predict(
            bgr,
            conf=self._confidence,
            augment=False,
            verbose=False,
            device=self._device,
        )
        return [tuple(box) for box in results[0].boxes.xyxy.tolist()]
