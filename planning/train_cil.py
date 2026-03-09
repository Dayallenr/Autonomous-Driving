"""
CIL training script.

Loads demonstration data collected by collect_data.py and trains the
CILModel via supervised regression on the active command branch.

Usage:
    python planning/train_cil.py --config config/config.yaml

The best model checkpoint is saved to models/cil_model.pt.
"""
from __future__ import annotations

import argparse
import sys
import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import yaml
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planning.cil_model import CILModel, NUM_COMMANDS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DemonstrationDataset(Dataset):
    """
    Loads all episode data from data/demonstrations/.

    Each sample is (image_tensor, command, waypoints_tensor) where:
      image_tensor: (3, H, W) float32 normalised
      command:      int in {0,1,2,3}
      waypoints:    (num_waypoints * 2,) float32 in vehicle-local metres
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, data_dir: str | Path, img_size: tuple = (224, 224)) -> None:
        self.img_size = img_size
        self.samples: list[tuple[Path, int, np.ndarray]] = []

        data_dir = Path(data_dir)
        for ep_dir in sorted(data_dir.glob("episode_*")):
            data_file = ep_dir / "data.npy"
            images_dir = ep_dir / "images"
            if not data_file.exists() or not images_dir.exists():
                continue

            records = np.load(str(data_file))  # (T, 1 + num_waypoints*2)
            for idx, row in enumerate(records):
                command = int(row[0])
                waypoints = row[1:]  # (num_waypoints*2,)
                img_path = images_dir / f"{idx:06d}.png"
                if img_path.exists() and 0 <= command < NUM_COMMANDS:
                    self.samples.append((img_path, command, waypoints))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, command, waypoints = self.samples[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((*self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)

        return img_tensor, torch.tensor(command, dtype=torch.long), torch.tensor(waypoints, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    train_cfg = cfg.get("training", {})
    planning_cfg = cfg.get("planning", {})
    dc_cfg = cfg.get("data_collection", {})

    data_dir = dc_cfg.get("output_dir", "data/demonstrations")
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cil_path = checkpoint_dir / "cil_model.pt"

    num_waypoints = planning_cfg.get("waypoint_lookahead", 5)
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 1e-4)
    num_epochs = train_cfg.get("num_epochs", 50)
    val_split = train_cfg.get("validation_split", 0.1)
    log_interval = train_cfg.get("log_interval", 50)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # Dataset split
    dataset = DemonstrationDataset(data_dir)
    if len(dataset) == 0:
        print("[Train] No training data found. Run collect_data.py first.")
        return

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CILModel(num_waypoints=num_waypoints, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for step, (images, commands, waypoints) in enumerate(train_loader):
            images    = images.to(device)
            commands  = commands.to(device)
            waypoints = waypoints.to(device)

            all_branches = model(images)  # tuple of 4 tensors (B, wp*2)

            # Gather the active branch output for each sample in the batch
            preds = torch.stack(all_branches, dim=1)  # (B, 4, wp*2)
            cmd_idx = commands.view(-1, 1, 1).expand(-1, 1, preds.size(2))
            active_pred = preds.gather(1, cmd_idx).squeeze(1)  # (B, wp*2)

            loss = criterion(active_pred, waypoints)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            if step % log_interval == 0:
                print(
                    f"  [Epoch {epoch}/{num_epochs}] step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, commands, waypoints in val_loader:
                images    = images.to(device)
                commands  = commands.to(device)
                waypoints = waypoints.to(device)

                all_branches = model(images)
                preds = torch.stack(all_branches, dim=1)
                cmd_idx = commands.view(-1, 1, 1).expand(-1, 1, preds.size(2))
                active_pred = preds.gather(1, cmd_idx).squeeze(1)
                val_loss += criterion(active_pred, waypoints).item()

        avg_train = train_loss / max(1, len(train_loader))
        avg_val   = val_loss   / max(1, len(val_loader))
        scheduler.step(avg_val)

        print(f"[Epoch {epoch}/{num_epochs}] train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), str(cil_path))
            print(f"  *** Saved best model → {cil_path} (val_loss={best_val_loss:.4f})")

    print(f"\n[Train] Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"[Train] Best model saved at: {cil_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CIL planning model")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
