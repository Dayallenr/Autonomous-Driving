"""
Conditional Imitation Learning (CIL) model.

Architecture:
  - ResNet-18 image encoder (pretrained on ImageNet, fine-tuned on demos)
  - Four separate MLP branches — one per high-level command:
      0: FOLLOW_LANE
      1: TURN_LEFT
      2: TURN_RIGHT
      3: GO_STRAIGHT
  - Each branch outputs `num_waypoints * 2` values (x, y per waypoint)
    in the vehicle's local frame.

Reference: "End-to-end Driving via Conditional Imitation Learning"
           Codevilla et al., ICRA 2018.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# High-level command indices (must match config.yaml and collect_data.py)
COMMAND_FOLLOW_LANE = 0
COMMAND_TURN_LEFT   = 1
COMMAND_TURN_RIGHT  = 2
COMMAND_GO_STRAIGHT = 3

NUM_COMMANDS = 4


class _Branch(nn.Module):
    """MLP prediction branch for one command."""

    def __init__(self, in_features: int, num_waypoints: int, output_dim: int | None = None) -> None:
        super().__init__()
        out = output_dim if output_dim is not None else num_waypoints * 2
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CILModel(nn.Module):
    """
    Conditional Imitation Learning model.

    Forward pass returns the waypoints for ALL branches so that the correct
    branch can be selected by the caller using the current command index.
    This makes training straightforward (loss only on the active branch).

    Args:
        num_waypoints: Number of future (x, y) waypoints to predict per branch.
        pretrained:    If True, initialise the ResNet-18 backbone with ImageNet
                       weights (requires torchvision).

    Input:
        image: (B, 3, H, W) float tensor, normalised to ImageNet mean/std.

    Output:
        Tuple of four tensors, each (B, num_waypoints*2):
        (follow_lane_wp, turn_left_wp, turn_right_wp, go_straight_wp)
    """

    def __init__(
        self,
        num_waypoints: int = 5,
        pretrained: bool = True,
        output_mode: str = "waypoints",
    ) -> None:
        """
        Args:
            num_waypoints: Waypoints per branch, when ``output_mode`` is "waypoints".
            pretrained: Initialise the ResNet-18 backbone from ImageNet weights.
            output_mode: ``"waypoints"`` predicts (x, y) pairs in the vehicle
                frame; ``"control"`` predicts (steer, throttle, brake) directly,
                as in Codevilla et al. Direct control is what the DAgger loop
                uses, because the expert it imitates produces controls — going
                via waypoints would insert a hand-tuned tracking controller
                between student and expert and make the imitation error measure
                that controller as much as the network.
        """
        super().__init__()
        if output_mode not in ("waypoints", "control"):
            raise ValueError(
                f"output_mode must be 'waypoints' or 'control', got {output_mode!r}"
            )
        self.num_waypoints = num_waypoints
        self.output_mode = output_mode
        self.output_dim = 3 if output_mode == "control" else num_waypoints * 2

        # ---- Backbone ----
        try:
            from torchvision.models import ResNet18_Weights, resnet18
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        except ImportError:
            # torchvision not available — build a lightweight CNN from scratch
            backbone = _SimpleCNNBackbone()
            self._features_dim = 512
            self.backbone = backbone
            self._build_branches(self._features_dim, num_waypoints)
            return

        # Replace the final FC with an identity so we keep the 512-d feature vec
        self._features_dim = backbone.fc.in_features  # 512 for ResNet-18
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self._build_branches(self._features_dim, num_waypoints)

    def _build_branches(self, feat_dim: int, num_waypoints: int) -> None:
        out = self.output_dim
        self.branch_follow   = _Branch(feat_dim, num_waypoints, out)
        self.branch_left     = _Branch(feat_dim, num_waypoints, out)
        self.branch_right    = _Branch(feat_dim, num_waypoints, out)
        self.branch_straight = _Branch(feat_dim, num_waypoints, out)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone(image)
        return (
            self.branch_follow(features),
            self.branch_left(features),
            self.branch_right(features),
            self.branch_straight(features),
        )

    def predict(self, image: torch.Tensor, command: int) -> torch.Tensor:
        """
        Convenience method: run forward and return only the active branch output.

        Args:
            image:   (1, 3, H, W) normalised tensor.
            command: int in {0, 1, 2, 3}.

        Returns:
            (1, num_waypoints * 2) tensor.
        """
        with torch.no_grad():
            all_branches = self.forward(image)
        return all_branches[command]


# ---------------------------------------------------------------------------
# Fallback backbone when torchvision is not installed
# ---------------------------------------------------------------------------

class _SimpleCNNBackbone(nn.Module):
    """
    Lightweight CNN backbone producing a 512-d feature vector.
    Used when torchvision is unavailable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # (3, H, W) → (32, H/2, W/2)
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # → (64, H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # → (128, H/8, W/8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # → (256, H/16, W/16)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))
