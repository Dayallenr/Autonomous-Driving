#!/usr/bin/env python
"""
Evaluate a detector checkpoint on the sequence-disjoint KITTI validation split.

    python scripts/eval_detector.py --weights models/yolov8m.pt

Writes a per-class report with instance, image, and drive counts to
``results/perception/<name>/report.json`` and prints it.

A warning about evaluating checkpoints trained elsewhere
--------------------------------------------------------
This split is only leak-free for models **trained on its training half**. A
checkpoint trained on a random split of KITTI has already seen frames from the
drives now held out — around 80% of all 141 drives appear somewhere in its
training data — so scoring it here does not measure generalisation either. It is
still leaked; only the bookkeeping changed.

Pass ``--trained-elsewhere`` to acknowledge this. The flag does not alter the
computation; it stamps the caveat into the report so the number cannot later be
quoted as a clean held-out result.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder.data.kitti import load_drive_index, support_by_class  # noqa: E402
from pathfinder.detection.evaluate import evaluate  # noqa: E402

logger = logging.getLogger("eval_detector")

LEAK_WARNING = (
    "checkpoint was trained on a different split, so the validation drives here "
    "were very likely in its training data — this figure is NOT a clean held-out "
    "result and must not be reported as one"
)


def load_val_support(root: Path) -> dict[str, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    val_drives = set(manifest["split"]["val_drives"])

    index = load_drive_index(root / "raw" / "train_mapping.txt", root / "raw" / "train_rand.txt")
    labels: dict[int, list[int]] = {}
    for split in ("train", "val"):
        for label_path in (root / "raw" / "labels" / split).glob("*.txt"):
            labels[int(label_path.stem)] = [
                int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()
            ]

    val_frames = {frame for drive in val_drives for frame in index.frames_of_drive.get(drive, ())}
    return support_by_class(index, labels, val_frames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a detector on held-out KITTI drives")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--name", default=None, help="report directory name")
    parser.add_argument("--device", default="", help="cpu, mps, or a CUDA index")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("results/perception"))
    parser.add_argument(
        "--trained-elsewhere",
        action="store_true",
        help="checkpoint was not trained on this split; stamp the leak caveat into the report",
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_yaml = arguments.data_root / "kitti_seqdisjoint.yaml"
    if not data_yaml.exists():
        logger.error("no dataset config at %s\nrun: python scripts/prepare_kitti.py", data_yaml)
        return 1
    if not arguments.weights.exists():
        logger.error("no checkpoint at %s", arguments.weights)
        return 1

    notes = []
    if arguments.trained_elsewhere:
        notes.append(LEAK_WARNING)
        logger.warning("\n!! %s\n", LEAK_WARNING)

    report = evaluate(
        arguments.weights,
        data_yaml,
        support=load_val_support(arguments.data_root),
        device=arguments.device,
        image_size=arguments.image_size,
        batch=arguments.batch,
        notes=notes,
    )

    name = arguments.name or f"eval-{arguments.weights.stem}"
    destination = arguments.out / name / "report.json"
    report.write(destination)

    print(f"\n{report.table()}\n")
    for note in report.notes:
        print(f"note: {note}")
    print(f"\nreport: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
