"""
Tests for KITTI provenance recovery, sequence-disjoint splitting, and rebalancing.

These use a small synthetic dataset rather than the real 390 MB download so the
suite runs in CI without it. The properties under test — that no drive straddles
the split, that adjacency leakage is zero by construction, that rare classes keep
val support — are structural and hold at any scale.
"""
from __future__ import annotations

import pytest

from pathfinder.data.kitti import (
    NUM_FRAMES,
    DriveIndex,
    build_split,
    class_distribution,
    leakage_report,
    load_drive_index,
    oversample_for_balance,
)


def make_index(drives: dict[str, list[int]]) -> DriveIndex:
    """Build a DriveIndex directly, with consecutive raw frame numbers per drive."""
    drive_of, raw_frame_of, frames_of_drive = {}, {}, {}
    for drive, indices in drives.items():
        frames_of_drive[drive] = sorted(indices)
        for position, image_index in enumerate(sorted(indices)):
            drive_of[image_index] = drive
            raw_frame_of[image_index] = position
    return DriveIndex(drive_of, raw_frame_of, frames_of_drive)


# ─────────────────────────────────────────────────────────────────────────────
# Provenance recovery
# ─────────────────────────────────────────────────────────────────────────────


def write_devkit(tmp_path, mapping_lines: list[str], order: list[int]):
    mapping = tmp_path / "train_mapping.txt"
    rand = tmp_path / "train_rand.txt"
    mapping.write_text("\n".join(mapping_lines) + "\n")
    rand.write_text(",".join(str(value) for value in order) + "\n")
    return mapping, rand


def test_drive_index_composes_the_two_devkit_files(tmp_path):
    """``train_rand`` is a permutation *into* ``train_mapping``, not a direct map.

    Getting this composition backwards still yields a plausible-looking index, so
    it is pinned explicitly.
    """
    mapping_lines = [f"2011_09_26 drive_{i:02d} {i * 10:010d}" for i in range(1, NUM_FRAMES + 1)]
    order = list(range(NUM_FRAMES, 0, -1))  # reverse permutation
    mapping, rand = write_devkit(tmp_path, mapping_lines, order)

    index = load_drive_index(mapping, rand)

    # Image 0 -> order[0] = NUM_FRAMES -> the *last* mapping line.
    assert index.drive_of[0] == f"drive_{NUM_FRAMES:02d}"
    assert index.raw_frame_of[0] == NUM_FRAMES * 10


def test_drive_index_rejects_wrong_frame_count(tmp_path):
    mapping, rand = write_devkit(tmp_path, ["2011_09_26 drive_01 0000000000"], [1])
    with pytest.raises(ValueError, match="expected 7481"):
        load_drive_index(mapping, rand)


def test_drive_index_rejects_out_of_range_pointer(tmp_path):
    mapping_lines = [f"2011_09_26 drive_01 {i:010d}" for i in range(NUM_FRAMES)]
    order = [NUM_FRAMES + 1] + list(range(2, NUM_FRAMES + 1))
    mapping, rand = write_devkit(tmp_path, mapping_lines, order)
    with pytest.raises(ValueError, match="outside 1"):
        load_drive_index(mapping, rand)


# ─────────────────────────────────────────────────────────────────────────────
# Leakage measurement
# ─────────────────────────────────────────────────────────────────────────────


def test_random_split_over_video_leaks_adjacent_frames():
    """The failure mode this whole module exists to prevent."""
    index = make_index({"drive_a": list(range(100))})
    train = {i for i in range(100) if i % 2 == 0}
    val = {i for i in range(100) if i % 2 == 1}

    report = leakage_report(index, train, val)

    assert report["drives_shared"] == 1
    # Every odd frame sits between two even ones.
    assert report["temporal_adjacency"]["1"]["fraction"] == pytest.approx(1.0)


def test_sequence_disjoint_split_has_zero_leakage_by_construction():
    index = make_index({"drive_a": list(range(50)), "drive_b": list(range(50, 100))})
    report = leakage_report(index, set(range(50)), set(range(50, 100)))

    assert report["drives_shared"] == 0
    for stats in report["temporal_adjacency"].values():
        assert stats["frames"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────


def test_split_never_straddles_a_drive():
    drives = {f"drive_{d:02d}": list(range(d * 20, d * 20 + 20)) for d in range(10)}
    index = make_index(drives)
    labels = {i: [0] for i in range(200)}

    plan = build_split(index, labels, val_fraction=0.2)

    assert not (set(plan.train_drives) & set(plan.val_drives))
    for drive, frames in drives.items():
        in_train = [f in plan.train for f in frames]
        assert all(in_train) or not any(in_train), f"{drive} straddles the split"


def test_split_approximates_the_requested_fraction():
    drives = {f"drive_{d:02d}": list(range(d * 10, d * 10 + 10)) for d in range(20)}
    index = make_index(drives)
    labels = {i: [0] for i in range(200)}

    plan = build_split(index, labels, val_fraction=0.25)

    assert plan.val and plan.train
    assert abs(len(plan.val) / 200 - 0.25) < 0.1


def test_split_keeps_val_support_for_a_rare_class():
    """A class in few drives must still reach val — otherwise its AP is undefined.

    Without a class-aware objective the rare drive is placed on frame count alone
    and lands in train, leaving the class with zero validation instances.
    """
    drives = {f"drive_{d:02d}": list(range(d * 20, d * 20 + 20)) for d in range(10)}
    index = make_index(drives)
    labels: dict[int, list[int]] = {i: [0] for i in range(200)}
    # Class 4 exists only in drives 0 and 1.
    for i in list(range(0, 20)) + list(range(20, 40)):
        labels[i] = [0, 4]

    plan = build_split(index, labels, val_fraction=0.2)

    val_rare = sum(labels[i].count(4) for i in plan.val)
    assert val_rare > 0, "rare class has no validation support"


def test_split_is_deterministic():
    drives = {f"drive_{d:02d}": list(range(d * 20, d * 20 + 20)) for d in range(10)}
    index = make_index(drives)
    labels = {i: [i % 3] for i in range(200)}

    first = build_split(index, labels, val_fraction=0.2)
    second = build_split(index, labels, val_fraction=0.2)

    assert sorted(first.val_drives) == sorted(second.val_drives)


def test_split_rejects_degenerate_fraction():
    index = make_index({"drive_00": [0, 1]})
    with pytest.raises(ValueError, match="val_fraction"):
        build_split(index, {0: [0], 1: [0]}, val_fraction=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rebalancing
# ─────────────────────────────────────────────────────────────────────────────


def test_oversampling_repeats_rare_class_images():
    labels = {i: [0] for i in range(100)}
    labels[99] = [4]  # one image holds the only rare-class instance

    listing = oversample_for_balance(set(labels), labels, cap=8)

    assert listing.count(99) > 1
    assert listing.count(0) == 1


def test_oversampling_respects_the_cap():
    """Uncapped, a 129:1 ratio asks for ~129 copies of the same few scenes, and
    the model memorises those instead of learning the class."""
    labels = {i: [0] for i in range(1000)}
    labels[999] = [4]

    listing = oversample_for_balance(set(labels), labels, cap=3)

    assert listing.count(999) == 3


def test_oversampling_is_driven_by_the_rarest_class_present():
    labels = {i: [0] for i in range(100)}
    labels[99] = [0, 0, 0, 4]  # abundant and rare together

    listing = oversample_for_balance(set(labels), labels, cap=8)

    assert listing.count(99) > 1, "presence of common classes must not cancel the repeat"


def test_oversampling_rejects_bad_cap():
    with pytest.raises(ValueError, match="cap"):
        oversample_for_balance({0}, {0: [0]}, cap=0)


def test_class_distribution_counts_instances_and_images():
    labels = {0: [0, 0, 1], 1: [1], 2: []}
    distribution = class_distribution(labels)

    assert distribution["car"]["instances"] == 2
    assert distribution["car"]["images"] == 1
    assert distribution["van"]["instances"] == 2
    assert distribution["van"]["images"] == 2
    assert distribution["car"]["share"] == pytest.approx(0.5)
