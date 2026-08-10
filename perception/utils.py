"""
Utility helpers for the perception module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    """A single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    # [x1, y1, x2, y2] in pixel coordinates
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2],
            dtype=np.float32,
        )

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    color_map: dict | None = None,
) -> np.ndarray:
    """Draw bounding boxes and labels on an BGR image. Returns a new image."""
    if color_map is None:
        color_map = {
            "car": (0, 255, 0),
            "van": (0, 200, 0),
            "truck": (0, 150, 0),
            "pedestrian": (255, 0, 0),
            "person_sitting": (200, 0, 0),
            "cyclist": (0, 0, 255),
            "tram": (255, 165, 0),
            "misc": (128, 128, 128),
        }
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int)
        color = color_map.get(det.class_name, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            vis, label, (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return vis


def scale_bbox(bbox: np.ndarray, orig_w: int, orig_h: int, new_w: int, new_h: int) -> np.ndarray:
    """Scale a [x1,y1,x2,y2] bbox from one resolution to another."""
    sx, sy = new_w / orig_w, new_h / orig_h
    return np.array([bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy], dtype=np.float32)
