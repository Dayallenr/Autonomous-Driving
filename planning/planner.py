"""
CIL-based motion planner.

Wraps the trained CILModel and exposes a `plan(image, command)` method that
returns a list of (x, y) waypoint coordinates in the vehicle's local frame.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple

from .cil_model import CILModel


# Normalisation constants (ImageNet) — must match training preprocessing
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INPUT_SIZE = (224, 224)  # (W, H) for cv2.resize


class CILPlanner:
    """
    High-level motion planner based on Conditional Imitation Learning.

    Loads a trained CILModel and produces `num_waypoints` future (x, y)
    positions in the vehicle's local frame given an RGB camera frame and
    the current navigation command (follow lane / left / right / straight).
    """

    def __init__(self, cfg: dict) -> None:
        import cv2
        self._cv2 = cv2

        planning_cfg = cfg.get("planning", {})
        model_path = Path(planning_cfg.get("cil_model_path", "models/cil_model.pt"))
        self._num_waypoints = planning_cfg.get("waypoint_lookahead", 5)

        device_str = cfg.get("perception", {}).get("device", "cpu")
        self._device = torch.device(device_str if device_str != "mps" or
                                    torch.backends.mps.is_available() else "cpu")

        self._model = CILModel(num_waypoints=self._num_waypoints, pretrained=False)
        self._model.to(self._device)

        if model_path.exists():
            state = torch.load(str(model_path), map_location=self._device)
            self._model.load_state_dict(state)
            print(f"[CILPlanner] Loaded model from '{model_path}'")
        else:
            print(
                f"[CILPlanner] WARNING: model not found at '{model_path}'. "
                "Predictions will be random until the model is trained."
            )

        self._model.eval()

    def plan(
        self,
        bgr_frame: np.ndarray,
        command: int,
    ) -> List[Tuple[float, float]]:
        """
        Predict future waypoints given the current camera frame and command.

        Args:
            bgr_frame: BGR uint8 numpy array (H, W, 3) from CARLA camera.
            command:   High-level command index {0:lane, 1:left, 2:right, 3:straight}.

        Returns:
            List of (x, y) tuples in vehicle-local metres.
            x = forward, y = left.
        """
        # Pre-process: resize → RGB → normalise → tensor
        img = self._cv2.cvtColor(bgr_frame, self._cv2.COLOR_BGR2RGB)
        img = self._cv2.resize(img, _INPUT_SIZE)
        img = img.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            pred = self._model.predict(img_t, command)  # (1, num_wp*2)

        flat = pred.squeeze(0).cpu().numpy()  # (num_wp*2,)
        waypoints = [(float(flat[i * 2]), float(flat[i * 2 + 1]))
                     for i in range(self._num_waypoints)]
        return waypoints

    def plan_fallback(self, num_waypoints: int, spacing: float = 2.0) -> List[Tuple[float, float]]:
        """
        Straight-ahead fallback waypoints used before the CIL model is trained
        or when the camera frame is unavailable.
        """
        return [(float(spacing * (i + 1)), 0.0) for i in range(num_waypoints)]
