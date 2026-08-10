"""
Agent prediction module.

Maintains an IoU-based multi-object tracker across frames and predicts the
future bounding-box positions of each tracked agent using a constant-velocity
kinematic model.

No deep learning is used here — this keeps the prediction module robust and
fast while the perception (YOLO) and planning (CIL) do the heavy lifting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from perception.utils import Detection, compute_iou


@dataclass
class Track:
    """Single agent track maintained across simulation frames."""
    track_id: int
    class_id: int
    class_name: str

    # Current bounding box [x1, y1, x2, y2] in pixel coords
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    # Velocity of the bbox centre in pixels/frame
    vx: float = 0.0
    vy: float = 0.0

    hits: int = 1          # consecutive frames with a detection match
    age: int = 0           # frames since last matched detection
    is_confirmed: bool = False

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2],
            dtype=np.float32,
        )


@dataclass
class AgentPrediction:
    """Predicted future bounding boxes for one tracked agent."""
    track_id: int
    class_id: int
    class_name: str
    current_bbox: np.ndarray
    # List of predicted bbox centres [(cx, cy), ...] over the horizon
    predicted_centers: list[np.ndarray] = field(default_factory=list)
    # Predicted bounding boxes [x1,y1,x2,y2] at each step
    predicted_bboxes: list[np.ndarray] = field(default_factory=list)


class Predictor:
    """
    Multi-object tracker + constant-velocity predictor.

    Each call to `update(detections)` returns a list of `AgentPrediction`
    objects for all confirmed tracks.
    """

    _next_id: int = 0

    def __init__(self, cfg: dict) -> None:
        pred_cfg = cfg.get("prediction", {})
        self._dt = pred_cfg.get("dt", 0.05)
        self._horizon_s = pred_cfg.get("horizon_seconds", 2.0)
        self._iou_threshold = pred_cfg.get("iou_threshold", 0.3)
        self._max_age = pred_cfg.get("max_age", 10)
        self._min_hits = pred_cfg.get("min_hits", 3)
        self._horizon_steps = int(self._horizon_s / self._dt)

        self._tracks: dict[int, Track] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list[Detection]) -> list[AgentPrediction]:
        """
        Update all tracks with the new detection list.

        Returns a list of AgentPrediction for every confirmed track.
        """
        # Age all existing tracks
        for track in self._tracks.values():
            track.age += 1

        # Match detections to existing tracks using IoU
        matched, unmatched_dets, unmatched_trks = self._match(detections)

        # Update matched tracks
        for det_idx, trk_id in matched:
            trk = self._tracks[trk_id]
            old_center = trk.center
            new_bbox = detections[det_idx].bbox
            new_center = np.array(
                [(new_bbox[0] + new_bbox[2]) / 2, (new_bbox[1] + new_bbox[3]) / 2],
                dtype=np.float32,
            )
            # Velocity: exponential moving average for smoothness
            alpha = 0.6
            trk.vx = alpha * (float(new_center[0]) - float(old_center[0])) + (1 - alpha) * trk.vx
            trk.vy = alpha * (float(new_center[1]) - float(old_center[1])) + (1 - alpha) * trk.vy
            trk.bbox = new_bbox
            trk.age = 0
            trk.hits += 1
            if trk.hits >= self._min_hits:
                trk.is_confirmed = True

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_id = Predictor._next_id
            Predictor._next_id += 1
            self._tracks[new_id] = Track(
                track_id=new_id,
                class_id=det.class_id,
                class_name=det.class_name,
                bbox=det.bbox.copy(),
            )

        # Remove stale tracks
        stale = [tid for tid, t in self._tracks.items() if t.age > self._max_age]
        for tid in stale:
            del self._tracks[tid]

        # Build predictions for confirmed tracks
        predictions: list[AgentPrediction] = []
        for trk in self._tracks.values():
            if not trk.is_confirmed:
                continue
            pred = self._predict_track(trk)
            predictions.append(pred)

        return predictions

    def reset(self) -> None:
        self._tracks.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match(self, detections: list[Detection]):
        """
        Hungarian-like greedy IoU matching between detections and tracks.
        Returns (matched, unmatched_dets, unmatched_trks).
        """
        track_ids = list(self._tracks.keys())
        if not detections or not track_ids:
            return [], list(range(len(detections))), track_ids

        # Build IoU matrix
        iou_matrix = np.zeros((len(detections), len(track_ids)), dtype=np.float32)
        for d_idx, det in enumerate(detections):
            for t_idx, tid in enumerate(track_ids):
                iou_matrix[d_idx, t_idx] = compute_iou(det.bbox, self._tracks[tid].bbox)

        matched: list[tuple] = []
        used_dets = set()
        used_trks = set()

        # Greedy: pick the highest IoU pair repeatedly
        while True:
            if iou_matrix.size == 0:
                break
            flat_idx = int(np.argmax(iou_matrix))
            d_idx, t_idx = divmod(flat_idx, len(track_ids))
            best_iou = iou_matrix[d_idx, t_idx]
            if best_iou < self._iou_threshold:
                break
            if d_idx not in used_dets and t_idx not in used_trks:
                matched.append((d_idx, track_ids[t_idx]))
                used_dets.add(d_idx)
                used_trks.add(t_idx)
            iou_matrix[d_idx, :] = -1
            iou_matrix[:, t_idx] = -1

        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_trks = [track_ids[i] for i in range(len(track_ids)) if i not in used_trks]
        return matched, unmatched_dets, unmatched_trks

    def _predict_track(self, trk: Track) -> AgentPrediction:
        """Constant-velocity forward prediction for one track."""
        cx, cy = float(trk.center[0]), float(trk.center[1])
        w = float(trk.bbox[2] - trk.bbox[0])
        h = float(trk.bbox[3] - trk.bbox[1])

        predicted_centers: list[np.ndarray] = []
        predicted_bboxes: list[np.ndarray] = []

        for step in range(1, self._horizon_steps + 1):
            ncx = cx + trk.vx * step
            ncy = cy + trk.vy * step
            predicted_centers.append(np.array([ncx, ncy], dtype=np.float32))
            predicted_bboxes.append(
                np.array([ncx - w / 2, ncy - h / 2, ncx + w / 2, ncy + h / 2], dtype=np.float32)
            )

        return AgentPrediction(
            track_id=trk.track_id,
            class_id=trk.class_id,
            class_name=trk.class_name,
            current_bbox=trk.bbox.copy(),
            predicted_centers=predicted_centers,
            predicted_bboxes=predicted_bboxes,
        )
