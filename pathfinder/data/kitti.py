"""
KITTI ingestion: sequence-disjoint splitting and rare-class rebalancing.

Why this module exists
----------------------
KITTI's 7,481 detection images are not independent samples. They are frames
sampled from 141 continuous 10 Hz video drives, and the conventional way to
split them — shuffle the 7,481 filenames, take 80/20 — puts frames captured
0.1 s apart on opposite sides of the split.

That is not a subtle leak. Measured on the standard split shipped with
Ultralytics' ``kitti.yaml`` (see :func:`leakage_report`), **34% of validation
frames sit within one raw frame of a training frame** and 76% within two. A
detector does not have to generalise to score well on that; it has to remember.
Published KITTI numbers are consequently much lower than what a random split
reports, and the two are not comparable.

So this module splits by *drive*: every frame from a given drive lands wholly in
train or wholly in val. The resulting number is lower and means something.

The mapping that makes it possible
----------------------------------
KITTI ships two devkit files that together recover each detection frame's
provenance:

* ``train_rand.txt`` — a permutation; entry *i* (0-based) gives the 1-based line
  in ``train_mapping.txt`` for detection image ``{i:06d}.png``.
* ``train_mapping.txt`` — one line per frame: ``<date> <drive> <raw_frame>``.

Composing them recovers (drive, raw frame index) per image, which is the only
thing needed to group frames by source video.

The rare-class constraint
-------------------------
KITTI is severely imbalanced — 28,742 ``car`` instances against 222
``person_sitting``, a 129:1 ratio, with ``person_sitting`` appearing in just 99
images. Splitting by drive while ignoring that will happily put most of those 99
images on one side, leaving a class with near-zero support in val whose AP is
then pure noise. :func:`build_split` therefore packs drives with a
rarity-weighted objective rather than by size alone.
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "CLASS_NAMES",
    "DriveIndex",
    "SplitPlan",
    "build_split",
    "class_distribution",
    "leakage_report",
    "load_drive_index",
    "oversample_for_balance",
    "support_by_class",
    "write_manifest",
]

#: KITTI's 8 detection classes, in the label-file id order.
CLASS_NAMES: dict[int, str] = {
    0: "car",
    1: "van",
    2: "truck",
    3: "pedestrian",
    4: "person_sitting",
    5: "cyclist",
    6: "tram",
    7: "misc",
}

#: Total detection frames in the KITTI training set.
NUM_FRAMES = 7481

#: Cost added per class left with no instances on one side of the split. Set far
#: above any achievable share deviation (which is bounded by 1.0) so the
#: optimiser treats starvation as forbidden rather than merely expensive.
STARVED_CLASS_PENALTY = 100.0


@dataclass(frozen=True)
class DriveIndex:
    """Provenance for every detection frame.

    Attributes:
        drive_of: image index -> source drive id (e.g. ``2011_09_26_drive_0005_sync``).
        raw_frame_of: image index -> frame number within that drive.
        frames_of_drive: drive id -> sorted image indices belonging to it.
    """

    drive_of: dict[int, str]
    raw_frame_of: dict[int, int]
    frames_of_drive: dict[str, list[int]]

    @property
    def num_drives(self) -> int:
        return len(self.frames_of_drive)


def load_drive_index(mapping_path: Path, rand_path: Path) -> DriveIndex:
    """Recover each detection frame's source drive from the KITTI devkit files.

    Args:
        mapping_path: ``train_mapping.txt`` — one ``<date> <drive> <frame>`` per line.
        rand_path: ``train_rand.txt`` — comma-separated 1-based line numbers.

    Raises:
        ValueError: If either file does not describe exactly ``NUM_FRAMES``
            frames, or an index points outside the mapping. Both would silently
            corrupt provenance, so they are checked rather than trusted.
    """
    mapping = mapping_path.read_text(encoding="utf-8").splitlines()
    if len(mapping) != NUM_FRAMES:
        raise ValueError(f"{mapping_path.name} has {len(mapping)} lines, expected {NUM_FRAMES}")

    raw = rand_path.read_text(encoding="utf-8").replace("\n", "").strip().strip(",")
    order = [int(token) for token in raw.split(",")]
    if len(order) != NUM_FRAMES:
        raise ValueError(f"{rand_path.name} has {len(order)} entries, expected {NUM_FRAMES}")

    drive_of: dict[int, str] = {}
    raw_frame_of: dict[int, int] = {}
    frames_of_drive: dict[str, list[int]] = defaultdict(list)

    for image_index, line_number in enumerate(order):
        if not 1 <= line_number <= NUM_FRAMES:
            raise ValueError(
                f"{rand_path.name} entry {image_index} is {line_number}, "
                f"outside 1..{NUM_FRAMES}"
            )
        _date, drive, frame = mapping[line_number - 1].split()
        drive_of[image_index] = drive
        raw_frame_of[image_index] = int(frame)
        frames_of_drive[drive].append(image_index)

    for indices in frames_of_drive.values():
        indices.sort()

    return DriveIndex(
        drive_of=drive_of,
        raw_frame_of=raw_frame_of,
        frames_of_drive=dict(frames_of_drive),
    )


def class_distribution(labels: dict[int, list[int]]) -> dict[str, dict]:
    """Instance and image counts per class.

    Args:
        labels: image index -> list of class ids present in that image.
    """
    instances: Counter[int] = Counter()
    images: dict[int, set[int]] = defaultdict(set)
    for index, class_ids in labels.items():
        for class_id in class_ids:
            instances[class_id] += 1
            images[class_id].add(index)

    total = sum(instances.values())
    return {
        CLASS_NAMES[class_id]: {
            "instances": instances[class_id],
            "images": len(images[class_id]),
            "share": instances[class_id] / total if total else 0.0,
        }
        for class_id in sorted(CLASS_NAMES)
    }


def support_by_class(
    index: DriveIndex,
    labels: dict[int, list[int]],
    frames: set[int],
) -> dict[str, dict]:
    """Per-class support within ``frames``, counted three ways.

    Instances alone overstate how much evidence a class carries. KITTI frames
    come from continuous video, so 56 instances spread over one drive are
    repeated views of one scene, not 56 independent observations. Reporting the
    **drive** count beside the instance count is what makes an AP figure
    interpretable — a class backed by a single drive has an effective sample
    size far below its instance count, and its AP should be read as noisy.
    """
    instances: Counter[int] = Counter()
    images: dict[int, set[int]] = defaultdict(set)
    drives: dict[int, set[str]] = defaultdict(set)

    for image_index in frames:
        for class_id in labels.get(image_index, ()):
            instances[class_id] += 1
            images[class_id].add(image_index)
            drives[class_id].add(index.drive_of[image_index])

    return {
        CLASS_NAMES[class_id]: {
            "instances": instances[class_id],
            "images": len(images[class_id]),
            "drives": len(drives[class_id]),
        }
        for class_id in sorted(CLASS_NAMES)
    }


def leakage_report(
    index: DriveIndex,
    train: set[int],
    val: set[int],
    *,
    gaps: tuple[int, ...] = (1, 2, 3, 5, 10),
) -> dict:
    """Quantify how much of ``val`` is near-duplicated in ``train``.

    Two measures, because they answer different questions:

    * **Shared drives** — how many source videos straddle the split at all. A
      coarse yes/no on whether the split is sequence-disjoint.
    * **Temporal adjacency** — for each gap *g*, the fraction of val frames
      within *g* raw frames of some train frame in the same drive. KITTI raw is
      10 Hz, so g=1 is 0.1 s: effectively the same photograph.

    A sequence-disjoint split scores zero on both, by construction.
    """
    train_frames_by_drive: dict[str, set[int]] = defaultdict(set)
    for image_index in train:
        train_frames_by_drive[index.drive_of[image_index]].add(index.raw_frame_of[image_index])

    train_drives = {index.drive_of[i] for i in train}
    val_drives = {index.drive_of[i] for i in val}

    adjacency: dict[str, dict] = {}
    for gap in gaps:
        leaked = 0
        for image_index in val:
            drive = index.drive_of[image_index]
            frame = index.raw_frame_of[image_index]
            neighbours = train_frames_by_drive.get(drive, ())
            if any(frame + offset in neighbours for offset in range(-gap, gap + 1) if offset):
                leaked += 1
        adjacency[str(gap)] = {
            "frames": leaked,
            "fraction": leaked / len(val) if val else 0.0,
        }

    return {
        "val_frames": len(val),
        "train_frames": len(train),
        "drives_total": index.num_drives,
        "drives_shared": len(train_drives & val_drives),
        "temporal_adjacency": adjacency,
    }


@dataclass
class SplitPlan:
    """A drive-level assignment of frames to train/val."""

    train_drives: list[str] = field(default_factory=list)
    val_drives: list[str] = field(default_factory=list)
    train: set[int] = field(default_factory=set)
    val: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "train_drives": sorted(self.train_drives),
            "val_drives": sorted(self.val_drives),
            "train_frames": len(self.train),
            "val_frames": len(self.val),
            "val_fraction": len(self.val) / max(len(self.train) + len(self.val), 1),
        }


def build_split(
    index: DriveIndex,
    labels: dict[int, list[int]],
    *,
    val_fraction: float = 0.2,
    class_weight: float = 4.0,
    max_passes: int = 50,
    restarts: int = 40,
    seed: int = 0,
) -> SplitPlan:
    """Assign whole drives to train/val, balancing frame count and rare classes.

    Two stages:

    1. **Greedy seed.** Drives placed largest-first — the big, least-flexible
       items go while there is still room to compensate.
    2. **Local search.** Repeatedly move any single drive to the other side if
       doing so lowers total cost, until nothing improves. The greedy pass is
       order-dependent and commits to early choices it cannot revisit; with only
       141 drives, sweeping for improving moves is cheap and measurably better.

    The cost is a squared deviation from the target share, in two parts: overall
    frame count, and per-class instance share averaged with ``1/sqrt(instances)``
    weights so rare classes count for more. ``class_weight`` scales the class
    term against the frame-count term — without it the frame term dominates by
    two orders of magnitude and rare classes land wherever chance puts them.

    A caveat this function cannot engineer away: a class confined to a handful of
    drives has only a handful of achievable val shares. ``person_sitting`` lives
    in 3 of 141 drives, so no sequence-disjoint split gives it a balanced,
    low-variance validation set. :func:`build_split` gets as close as the data
    allows; reporting should carry the caveat rather than hide it.

    Raises:
        ValueError: If ``val_fraction`` is not strictly between 0 and 1.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    totals: Counter[int] = Counter()
    for class_ids in labels.values():
        totals.update(class_ids)
    # Rare classes carry more weight; sqrt keeps a 129:1 count ratio from
    # becoming a 129:1 weight ratio and drowning out frame-count balance.
    #
    # Only classes that actually occur are weighted. An absent class has no
    # share to balance, and folding it in at weight 1.0 would inflate the
    # normaliser and silently scale the whole class term towards zero — which is
    # invisible on full KITTI, where all eight classes are present, and wrong
    # everywhere else.
    weight = {c: 1.0 / totals[c] ** 0.5 for c in CLASS_NAMES if totals[c]}
    weight_sum = sum(weight.values()) or 1.0

    drive_classes: dict[str, Counter[int]] = {}
    drive_size: dict[str, int] = {}
    for drive, frames in index.frames_of_drive.items():
        counter: Counter[int] = Counter()
        for frame in frames:
            counter.update(labels.get(frame, ()))
        drive_classes[drive] = counter
        drive_size[drive] = len(frames)

    total_frames = sum(drive_size.values())

    def cost(in_val: set[str]) -> float:
        """Squared deviation from every target share, given a val drive set."""
        n_val = sum(drive_size[d] for d in in_val)
        penalty = (n_val / total_frames - val_fraction) ** 2

        class_penalty = 0.0
        starved = 0
        for class_id in weight:
            in_val_count = sum(drive_classes[d][class_id] for d in in_val)
            share = in_val_count / totals[class_id]
            class_penalty += weight[class_id] * (share - val_fraction) ** 2
            # Squared deviation alone will happily starve a class. When a class
            # occupies whole drives, its achievable val shares are coarse — say
            # {0, 0.5, 1.0} — and |0 - 0.2| beats |0.5 - 0.2|, so the optimiser
            # picks *zero validation instances* as the tidier number. AP is then
            # undefined for that class and the split is useless for the thing it
            # was built to measure. Zero support on either side is a
            # constraint violation, not a slightly worse share.
            if in_val_count == 0 or in_val_count == totals[class_id]:
                starved += 1

        return penalty + class_weight * class_penalty / weight_sum + STARVED_CLASS_PENALTY * starved

    ordered = sorted(index.frames_of_drive, key=lambda d: (-drive_size[d], d))

    def refine(seed_set: set[str]) -> tuple[set[str], float]:
        """Hill-climb from ``seed_set`` over single-drive moves *and* swaps.

        Swaps are not an optimisation nicety, they are required. A class living
        in a few drives has only a few achievable val shares, and reaching the
        best one usually means exchanging one drive for another: each half of
        that exchange, taken alone, raises the cost, so a move-only search sits
        in its local optimum forever.
        """
        current = set(seed_set)
        best_cost = cost(current)

        for _ in range(max_passes):
            improved = False

            for drive in ordered:
                candidate = current - {drive} if drive in current else current | {drive}
                candidate_cost = cost(candidate)
                if candidate_cost < best_cost - 1e-12:
                    current, best_cost = candidate, candidate_cost
                    improved = True

            for outgoing in sorted(current):
                if outgoing not in current:
                    continue
                for incoming in ordered:
                    if incoming in current:
                        continue
                    candidate = (current - {outgoing}) | {incoming}
                    candidate_cost = cost(candidate)
                    if candidate_cost < best_cost - 1e-12:
                        current, best_cost = candidate, candidate_cost
                        improved = True
                        break

            if not improved:
                break
        return current, best_cost

    # Stage 1 — greedy seed, largest drive first.
    greedy: set[str] = set()
    for drive in ordered:
        if cost(greedy | {drive}) < cost(greedy):
            greedy.add(drive)

    # Stage 2 — hill-climb, then restart from randomised seeds and keep the best.
    #
    # Even move+swap hill-climbing stalls here, because escaping some optima
    # needs a *compound* move. Concretely: the drive holding 67% of
    # ``person_sitting`` also holds the only ``tram`` instances on the val side,
    # so trading it away starves ``tram`` unless another tram drive moves in at
    # the same instant. No single move or swap does both. Restarts sidestep that
    # without the machinery of a full annealer, and the RNG is seeded so the
    # chosen split stays reproducible run to run.
    in_val, best = refine(greedy)

    rng = random.Random(seed)
    for _ in range(restarts):
        start = {drive for drive in ordered if rng.random() < val_fraction}
        candidate, candidate_cost = refine(start)
        if candidate_cost < best - 1e-12:
            in_val, best = candidate, candidate_cost

    plan = SplitPlan()
    for drive in ordered:
        if drive in in_val:
            plan.val_drives.append(drive)
            plan.val.update(index.frames_of_drive[drive])
        else:
            plan.train_drives.append(drive)
            plan.train.update(index.frames_of_drive[drive])

    logger.info(
        "split: %d train frames (%d drives) / %d val frames (%d drives), cost %.6f",
        len(plan.train), len(plan.train_drives), len(plan.val), len(plan.val_drives), best,
    )
    return plan


def oversample_for_balance(
    train: set[int],
    labels: dict[int, list[int]],
    *,
    cap: int = 8,
    target_share: float = 0.03,
) -> list[int]:
    """Repeat rare-class training images to rebalance the sampler.

    Returns a training list in which images containing under-represented classes
    appear more than once. Repetition is the honest lever for detection: YOLO's
    ``copy_paste`` augmentation needs segmentation masks, which KITTI detection
    labels do not carry, so pasting boxes would mean fabricating composites with
    wrong occlusion and lighting.

    Each image's repeat count is driven by its *rarest* class — an image with one
    ``person_sitting`` and six ``car`` is valuable for the former, and the cars it
    drags along are already abundant enough not to matter.

    Args:
        cap: Maximum repeats for any one image. Uncapped, a 129:1 ratio demands
            ~129 copies of 99 images, and the model overfits those exact scenes
            instead of learning the class.
        target_share: Instance share each class is pulled toward.

    Raises:
        ValueError: If ``cap`` is below 1.
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")

    totals: Counter[int] = Counter()
    for index in train:
        totals.update(labels.get(index, ()))
    total = sum(totals.values())
    if not total:
        return sorted(train)

    repeats_for_class = {
        class_id: min(cap, max(1, round(target_share / max(totals[class_id] / total, 1e-9))))
        for class_id in CLASS_NAMES
    }

    listing: list[int] = []
    for index in sorted(train):
        present = labels.get(index, ())
        repeats = max((repeats_for_class[c] for c in present), default=1)
        listing.extend([index] * repeats)

    logger.info(
        "oversampling: %d images -> %d entries (repeat factors %s)",
        len(train), len(listing),
        {CLASS_NAMES[c]: r for c, r in repeats_for_class.items() if r > 1},
    )
    return listing


def write_manifest(path: Path, payload: dict) -> None:
    """Persist a run's split and statistics so a result can be traced to its data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
