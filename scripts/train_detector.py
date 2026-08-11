#!/usr/bin/env python
"""
Train a YOLO detector on the sequence-disjoint KITTI split.

    python scripts/prepare_kitti.py      # once, builds the split
    python scripts/train_detector.py     # trains yolov8m, ~1h on an RTX 5070

On finishing it evaluates the best checkpoint on the held-out drives and writes
a per-class report carrying instance, image, **and drive** counts, so a rare
class's AP cannot be read as though it were as well-supported as ``car``.

Outputs
-------
``results/perception/<name>/report.json``   metrics with support counts
``results/perception/<name>/curves.csv``    per-epoch training curves
``models/<name>.pt``                        the best checkpoint

Why the numbers here differ from the old checkpoint
---------------------------------------------------
``models/best_perception_model.pt`` reports mAP@0.5 = 0.919, but it was trained
and validated on a *random* split of frames drawn from continuous video — 34% of
its validation frames sit 0.1 s from a training frame (see ``docs/DATA.md``).
This script validates on held-out drives. The resulting number is lower and
means something. Do not compare the two.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder.data.kitti import (  # noqa: E402
    load_drive_index,
    support_by_class,
)
from pathfinder.detection.evaluate import evaluate  # noqa: E402

logger = logging.getLogger("train_detector")


def load_val_support(root: Path) -> dict[str, dict]:
    """Per-class instances/images/drives over the validation split."""
    manifest = json.loads((root / "manifest.json").read_text())
    val_drives = set(manifest["split"]["val_drives"])

    index = load_drive_index(root / "raw" / "train_mapping.txt", root / "raw" / "train_rand.txt")
    labels: dict[int, list[int]] = {}
    for split in ("train", "val"):
        for label_path in (root / "raw" / "labels" / split).glob("*.txt"):
            labels[int(label_path.stem)] = [
                int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()
            ]

    val_frames = {
        frame for drive in val_drives for frame in index.frames_of_drive.get(drive, ())
    }
    return support_by_class(index, labels, val_frames)


def select_device(requested: str) -> str:
    """Resolve ``auto`` to the best device present, and say what was chosen."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        architectures = torch.cuda.get_arch_list()
        logger.info("CUDA device: %s (sm_%d%d)", name, *capability)
        # A Blackwell card on a pre-12.8 build silently falls back or dies with
        # "no kernel image is available". Checking here turns a confusing
        # runtime failure into a message that names the fix.
        target = f"sm_{capability[0]}{capability[1]}"
        if architectures and target not in architectures:
            logger.warning(
                "this torch build (%s, CUDA %s) has no %s kernels — compiled for %s.\n"
                "Reinstall with a matching CUDA build, e.g.:\n"
                "  pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu130",
                torch.__version__, torch.version.cuda, target, " ".join(architectures),
            )
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        logger.info("no CUDA device; using Apple MPS (training will be slow)")
        return "mps"
    logger.info("no GPU detected; using CPU (training will be very slow)")
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO detector on KITTI")
    parser.add_argument("--model", default="yolov8m.pt", help="base weights to fine-tune")
    parser.add_argument("--name", default=None, help="run name (defaults to the model stem)")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or a CUDA index")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "epochs without improvement before stopping; 0 disables early stopping "
            "(the default, and deliberately so — see the note in this file)"
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/perception"), help="report directory"
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_yaml = arguments.data_root / "kitti_seqdisjoint.yaml"
    if not data_yaml.exists():
        logger.error(
            "no dataset config at %s\nrun: python scripts/prepare_kitti.py", data_yaml
        )
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is required: pip install ultralytics")
        return 1

    name = arguments.name or Path(arguments.model).stem
    device = select_device(arguments.device)
    support = load_val_support(arguments.data_root)

    logger.info("\ntraining %s for %d epochs on device %s", arguments.model, arguments.epochs, device)
    logger.info("dataset: %s", data_yaml)

    # Ultralytics resolves a *relative* `project` against its own runs_dir
    # setting, so "results/perception" becomes "runs/detect/results/perception".
    # An absolute path is interpreted literally.
    output_root = arguments.out.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(arguments.model)
    model.train(
        data=str(data_yaml),
        epochs=arguments.epochs,
        batch=arguments.batch,
        imgsz=arguments.image_size,
        device=device,
        seed=arguments.seed,
        deterministic=True,
        workers=arguments.workers,
        # Early stopping defaults OFF, which is not the usual advice and is worth
        # the paragraph.
        #
        # A YOLO run's best epochs are its last ones, for two reasons that both
        # land at the end of the schedule: the learning rate anneals to lr0*lrf,
        # and `close_mosaic` switches mosaic augmentation off for the final 10
        # epochs so the model fine-tunes on undistorted images. Validation mAP
        # while mosaic is on is therefore not a reading of the model you will
        # ship, and it plateaus or wanders for long stretches mid-run.
        #
        # Early stopping reads that plateau as convergence. A first run here
        # stopped at epoch 30 of 60 with the LR still at 4.3x its final value and
        # the mosaic-off phase never reached — costing an hour of GPU time to
        # measure a model that was never allowed to finish. Ultralytics maps
        # patience=0 to infinity, so the schedule always completes.
        patience=arguments.patience,
        project=str(output_root),
        name=name,
        exist_ok=True,
        plots=True,
        val=True,
        # Geometry augmentation is left at Ultralytics defaults. Class balance is
        # handled upstream by oversampling rare-class images in the training file
        # list (scripts/prepare_kitti.py) rather than by pixel-level tricks:
        # `copy_paste` needs segmentation masks, which KITTI detection labels do
        # not carry.
        #
        # Vertical flip stays off: KITTI is a forward-facing road camera, so an
        # upside-down frame is not a pose the detector will ever be shown, and
        # training on it spends capacity on an impossible input.
        flipud=0.0,
        fliplr=0.5,
    )

    # Ask the trainer where it actually wrote rather than reconstructing the
    # path. Reconstructing is what silently lost an hour of training once: the
    # run landed under runs/detect/, this looked in results/perception/, found
    # nothing, and exited before writing the report — and runs/ is gitignored,
    # so the checkpoint was invisible to git as well.
    run_directory = Path(getattr(model.trainer, "save_dir", output_root / name))
    best = run_directory / "weights" / "best.pt"
    if not best.exists():
        logger.error(
            "training finished but no checkpoint at %s\n"
            "the trainer reported save_dir=%s — if that differs from the "
            "--out directory, evaluate it directly with:\n"
            "  python scripts/eval_detector.py --weights %s --name %s",
            best, run_directory, best, name,
        )
        return 1

    logger.info("\nevaluating %s on held-out drives", best)
    report = evaluate(
        best,
        data_yaml,
        support=support,
        device=device,
        image_size=arguments.image_size,
        batch=arguments.batch,
        notes=[
            f"trained {arguments.epochs} epochs from {arguments.model}, seed {arguments.seed}",
            "validation split is sequence-disjoint: no source drive appears in both "
            "train and val, so this number is not comparable to a random-split figure",
        ],
    )
    report.write(run_directory / "report.json")

    curves = run_directory / "results.csv"
    if curves.exists():
        shutil.copy(curves, run_directory / "curves.csv")

    models = Path("models")
    models.mkdir(exist_ok=True)
    shutil.copy(best, models / f"{name}.pt")

    print(f"\n{report.table()}\n")
    for note in report.notes:
        print(f"note: {note}")
    print(f"\nreport:     {run_directory / 'report.json'}")
    print(f"checkpoint: {models / f'{name}.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
