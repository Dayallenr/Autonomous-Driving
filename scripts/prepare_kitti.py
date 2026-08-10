#!/usr/bin/env python
"""
Build the KITTI training corpus: download, verify, split by drive, rebalance.

    python scripts/prepare_kitti.py

Idempotent — an existing, complete download is reused rather than re-fetched, so
re-running is cheap and safe.

Output (all under ``data/``)
----------------------------
``data/raw/``                 images, labels, and the two devkit mapping files
``data/splits/train.txt``     training image paths, rare classes oversampled
``data/splits/val.txt``       validation image paths, one line each
``data/kitti_seqdisjoint.yaml``  Ultralytics dataset config pointing at the above
``data/manifest.json``        split, class distribution, and leakage measurements

The manifest is the point: it records which drives went where and what the
resulting distribution was, so a model's metrics can be traced back to the exact
data that produced them.

Why not just use Ultralytics' bundled ``kitti.yaml``
---------------------------------------------------
Because its split is random over frames drawn from 10 Hz video, and this script
measures exactly how badly that leaks (``--report-baseline``). The headline: 34%
of its validation frames are within one raw frame — 0.1 s — of a training frame.
"""
from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfinder.data.kitti import (  # noqa: E402
    CLASS_NAMES,
    build_split,
    class_distribution,
    leakage_report,
    load_drive_index,
    oversample_for_balance,
    write_manifest,
)

logger = logging.getLogger("prepare_kitti")

KITTI_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip"
KITTI_BYTES = 390_529_329
DEVKIT_BASE = (
    "https://raw.githubusercontent.com/bostondiditeam/kitti/master/resources/devkit_object/mapping"
)
EXPECTED_FRAMES = 7481


def download(url: str, destination: Path, *, expected_bytes: int | None = None) -> Path:
    """Fetch ``url`` to ``destination`` unless a complete copy is already there."""
    if destination.exists() and (
        expected_bytes is None or destination.stat().st_size == expected_bytes
    ):
        logger.info("reusing %s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", url)
    # Write to a temporary name so an interrupted transfer is never mistaken for
    # a complete one on the next run.
    partial = destination.with_suffix(destination.suffix + ".partial")
    urllib.request.urlretrieve(url, partial)  # noqa: S310 - fixed https URLs above
    partial.replace(destination)
    logger.info("saved %s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)
    return destination


def ensure_dataset(root: Path) -> None:
    """Download and extract images/labels plus the devkit mapping files."""
    archive = download(KITTI_URL, root / "kitti.zip", expected_bytes=KITTI_BYTES)

    if not (root / "images" / "train").exists():
        logger.info("extracting %s", archive.name)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(root)

    for name in ("train_mapping.txt", "train_rand.txt"):
        download(f"{DEVKIT_BASE}/{name}", root / name)


def load_labels(root: Path) -> tuple[dict[int, list[int]], dict[int, Path]]:
    """Read every YOLO label file, keyed by the original KITTI frame index.

    Raises:
        ValueError: If the frame count is not exactly ``EXPECTED_FRAMES``. A short
            read here would quietly shrink the dataset and every downstream number.
    """
    labels: dict[int, list[int]] = {}
    images: dict[int, Path] = {}

    for split in ("train", "val"):
        for label_path in sorted((root / "labels" / split).glob("*.txt")):
            index = int(label_path.stem)
            image_path = root / "images" / split / f"{label_path.stem}.png"
            if not image_path.exists():
                raise ValueError(f"label {label_path} has no matching image at {image_path}")
            labels[index] = [
                int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()
            ]
            images[index] = image_path

    if len(labels) != EXPECTED_FRAMES:
        raise ValueError(f"found {len(labels)} frames, expected {EXPECTED_FRAMES}")
    return labels, images


def format_distribution(title: str, distribution: dict[str, dict]) -> str:
    lines = [f"\n{title}", f"{'class':<16}{'instances':>11}{'share':>9}{'images':>9}", "-" * 45]
    for name, stats in distribution.items():
        lines.append(
            f"{name:<16}{stats['instances']:>11,}{stats['share'] * 100:>8.2f}%{stats['images']:>9,}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", type=Path, default=Path("data"), help="data directory")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--oversample-cap",
        type=int,
        default=8,
        help="max repeats for a rare-class image (1 disables oversampling)",
    )
    parser.add_argument(
        "--report-baseline",
        action="store_true",
        help="also measure leakage in Ultralytics' bundled random split",
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = arguments.root
    raw = root / "raw"
    ensure_dataset(raw)

    labels, images = load_labels(raw)
    index = load_drive_index(raw / "train_mapping.txt", raw / "train_rand.txt")
    logger.info("\n%d frames across %d source drives", len(labels), index.num_drives)

    manifest: dict = {"frames": len(labels), "drives": index.num_drives}

    if arguments.report_baseline:
        baseline_train = {int(p.stem) for p in (raw / "images" / "train").glob("*.png")}
        baseline_val = {int(p.stem) for p in (raw / "images" / "val").glob("*.png")}
        baseline = leakage_report(index, baseline_train, baseline_val)
        manifest["baseline_random_split"] = baseline
        logger.info(
            "\nbundled random split: %d/%d drives straddle the split",
            baseline["drives_shared"], baseline["drives_total"],
        )
        for gap, stats in baseline["temporal_adjacency"].items():
            logger.info(
                "  val frames within +/-%2s raw frames of train: %5d (%.1f%%)",
                gap, stats["frames"], stats["fraction"] * 100,
            )

    plan = build_split(index, labels, val_fraction=arguments.val_fraction)
    leakage = leakage_report(index, plan.train, plan.val)
    manifest["split"] = plan.to_dict()
    manifest["leakage"] = leakage

    logger.info(
        "\nsequence-disjoint split: %d/%d drives straddle the split (must be 0)",
        leakage["drives_shared"], leakage["drives_total"],
    )
    for gap, stats in leakage["temporal_adjacency"].items():
        logger.info(
            "  val frames within +/-%2s raw frames of train: %5d (%.1f%%)",
            gap, stats["frames"], stats["fraction"] * 100,
        )

    train_distribution = class_distribution({i: labels[i] for i in plan.train})
    val_distribution = class_distribution({i: labels[i] for i in plan.val})
    manifest["class_distribution"] = {"train": train_distribution, "val": val_distribution}
    print(format_distribution("train split", train_distribution))
    print(format_distribution("val split", val_distribution))

    listing = (
        oversample_for_balance(plan.train, labels, cap=arguments.oversample_cap)
        if arguments.oversample_cap > 1
        else sorted(plan.train)
    )
    manifest["oversampling"] = {
        "cap": arguments.oversample_cap,
        "unique_images": len(plan.train),
        "sampled_entries": len(listing),
    }

    splits = root / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "train.txt").write_text(
        "\n".join(str(images[i].resolve()) for i in listing) + "\n", encoding="utf-8"
    )
    (splits / "val.txt").write_text(
        "\n".join(str(images[i].resolve()) for i in sorted(plan.val)) + "\n", encoding="utf-8"
    )

    names = "\n".join(f"  {cid}: {name}" for cid, name in sorted(CLASS_NAMES.items()))
    (root / "kitti_seqdisjoint.yaml").write_text(
        "# Generated by scripts/prepare_kitti.py — do not edit by hand.\n"
        "# Sequence-disjoint split: no source drive appears in both train and val.\n"
        f"path: {root.resolve()}\n"
        "train: splits/train.txt\n"
        "val: splits/val.txt\n\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )

    write_manifest(root / "manifest.json", manifest)
    logger.info(
        "\nwrote %s (%d entries), %s (%d), %s, %s",
        splits / "train.txt", len(listing),
        splits / "val.txt", len(plan.val),
        root / "kitti_seqdisjoint.yaml",
        root / "manifest.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
