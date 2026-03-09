"""
YOLO-based 2D object detector wrapping best_perception_model.pt.

The model was trained with Ultralytics 8.4.14 on the KITTI dataset and
detects 8 classes: car, van, truck, pedestrian, person_sitting, cyclist,
tram, misc.
"""
from __future__ import annotations

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List

from .utils import Detection


# KITTI class index → name
KITTI_CLASSES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "pedestrian",
    4: "person_sitting",
    5: "cyclist",
    6: "tram",
    7: "misc",
}


class YOLODetector:
    """
    Loads the Ultralytics YOLO perception model and runs inference on BGR
    or RGB numpy frames from the CARLA camera sensor.
    """

    def __init__(
        self,
        model_path: str | Path = "best_perception_model.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cpu",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.device = device

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Perception model not found at: {model_path}")

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(model_path))
            self._backend = "ultralytics"
        except ImportError:
            # Fallback: load raw state dict if ultralytics is not installed
            # (should not happen in a correctly set-up environment)
            raise ImportError(
                "ultralytics is required to run the perception model. "
                "Install it with: pip install ultralytics>=8.4.0"
            )

        print(
            f"[Perception] Loaded YOLO model from '{model_path}' "
            f"on device='{device}' | classes={list(KITTI_CLASSES.values())}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single image frame.

        Args:
            frame: BGR or RGB uint8 numpy array of shape (H, W, 3).

        Returns:
            List of Detection objects sorted by descending confidence.
        """
        if frame is None or frame.size == 0:
            return []

        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cid in zip(boxes_xyxy, confs, class_ids):
                detections.append(
                    Detection(
                        class_id=int(cid),
                        class_name=KITTI_CLASSES.get(int(cid), "unknown"),
                        confidence=float(conf),
                        bbox=bbox.astype(np.float32),
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of frames."""
        return [self.detect(f) for f in frames]
