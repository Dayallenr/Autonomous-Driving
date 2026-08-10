"""
Tests for the detector report: per-class alignment, support, and caveats.

The alignment test is the important one. Ultralytics returns per-class arrays
indexed by ``ap_class_index`` — the classes that actually appeared — not by
class id. Zipping those arrays against a static 8-class list produces a report
where every row below an absent class is attributed to the wrong class, and it
looks entirely plausible.
"""
from __future__ import annotations

import numpy as np
import pytest

from pathfinder.detection.evaluate import MIN_DRIVES_FOR_CONFIDENCE, build_report


class FakeBox:
    """Stands in for ultralytics' Metric object."""

    def __init__(self, ap50, ap, precision, recall):
        self.ap50 = np.asarray(ap50, dtype=float)
        self.ap = np.asarray(ap, dtype=float)
        self._p = np.asarray(precision, dtype=float)
        self._r = np.asarray(recall, dtype=float)

    @property
    def map50(self):
        return float(self.ap50.mean())

    @property
    def map(self):
        return float(self.ap.mean())

    @property
    def mp(self):
        return float(self._p.mean())

    @property
    def mr(self):
        return float(self._r.mean())

    def class_result(self, i):
        return float(self._p[i]), float(self._r[i]), 0.0, 0.0


class FakeMetrics:
    def __init__(self, ap_class_index, box):
        self.ap_class_index = np.asarray(ap_class_index, dtype=int)
        self.box = box


def make_support(**overrides):
    support = {
        name: {"instances": 100, "images": 50, "drives": 5}
        for name in (
            "car", "van", "truck", "pedestrian",
            "person_sitting", "cyclist", "tram", "misc",
        )
    }
    support.update(overrides)
    return support


def test_per_class_rows_follow_ap_class_index_not_position():
    """With class 1 (van) absent, index 2 must still resolve to truck."""
    metrics = FakeMetrics(
        ap_class_index=[0, 2, 3],  # car, truck, pedestrian — van missing
        box=FakeBox(ap50=[0.9, 0.8, 0.7], ap=[0.7, 0.6, 0.5],
                    precision=[0.9, 0.9, 0.9], recall=[0.8, 0.8, 0.8]),
    )
    report = build_report(
        metrics, weights="w.pt", dataset="d.yaml",
        support=make_support(
            car={"instances": 300, "images": 100, "drives": 9},
            truck={"instances": 200, "images": 80, "drives": 7},
            pedestrian={"instances": 100, "images": 60, "drives": 5},
        ),
    )

    by_name = {result.name: result for result in report.per_class}
    assert set(by_name) == {"car", "truck", "pedestrian"}
    assert by_name["car"].ap50 == pytest.approx(0.9)
    assert by_name["truck"].ap50 == pytest.approx(0.8)
    assert by_name["pedestrian"].ap50 == pytest.approx(0.7)


def test_absent_classes_are_reported_as_a_note():
    """An n-class mAP wearing an 8-class label is the failure being prevented."""
    metrics = FakeMetrics(
        ap_class_index=[0],
        box=FakeBox(ap50=[0.9], ap=[0.7], precision=[0.9], recall=[0.8]),
    )
    report = build_report(metrics, weights="w.pt", dataset="d.yaml", support=make_support())

    assert any("absent from the validation set" in note for note in report.notes)
    assert "tram" in " ".join(report.notes)


def test_single_drive_class_is_flagged_low_confidence():
    metrics = FakeMetrics(
        ap_class_index=[0, 4],
        box=FakeBox(ap50=[0.9, 0.95], ap=[0.7, 0.8],
                    precision=[0.9, 0.9], recall=[0.8, 0.8]),
    )
    report = build_report(
        metrics, weights="w.pt", dataset="d.yaml",
        support=make_support(person_sitting={"instances": 56, "images": 27, "drives": 1}),
    )

    by_name = {result.name: result for result in report.per_class}
    assert by_name["person_sitting"].low_confidence
    assert not by_name["car"].low_confidence
    assert report.low_confidence_classes == ["person_sitting"]
    assert any("high-variance" in note for note in report.notes)


def test_a_high_ap_does_not_clear_the_low_confidence_flag():
    """Support, not score, decides trustworthiness — a class can score 0.99 on
    one drive and still mean nothing."""
    metrics = FakeMetrics(
        ap_class_index=[4],
        box=FakeBox(ap50=[0.99], ap=[0.95], precision=[1.0], recall=[1.0]),
    )
    report = build_report(
        metrics, weights="w.pt", dataset="d.yaml",
        support=make_support(person_sitting={"instances": 56, "images": 27, "drives": 1}),
    )
    assert report.per_class[0].low_confidence


def test_multi_drive_class_is_not_flagged():
    metrics = FakeMetrics(
        ap_class_index=[6],
        box=FakeBox(ap50=[0.9], ap=[0.7], precision=[0.9], recall=[0.8]),
    )
    report = build_report(
        metrics, weights="w.pt", dataset="d.yaml",
        support=make_support(tram={"instances": 98, "images": 98,
                                   "drives": MIN_DRIVES_FOR_CONFIDENCE}),
    )
    assert not report.per_class[0].low_confidence


def test_rows_are_ordered_by_support():
    metrics = FakeMetrics(
        ap_class_index=[0, 4, 3],
        box=FakeBox(ap50=[0.9, 0.9, 0.9], ap=[0.7, 0.7, 0.7],
                    precision=[0.9] * 3, recall=[0.8] * 3),
    )
    report = build_report(
        metrics, weights="w.pt", dataset="d.yaml",
        support=make_support(
            car={"instances": 6663, "images": 1376, "drives": 13},
            pedestrian={"instances": 917, "images": 293, "drives": 10},
            person_sitting={"instances": 56, "images": 27, "drives": 1},
        ),
    )
    assert [r.name for r in report.per_class] == ["car", "pedestrian", "person_sitting"]


def test_report_roundtrips_through_json(tmp_path):
    metrics = FakeMetrics(
        ap_class_index=[0],
        box=FakeBox(ap50=[0.9], ap=[0.7], precision=[0.9], recall=[0.8]),
    )
    report = build_report(metrics, weights="w.pt", dataset="d.yaml", support=make_support())

    destination = tmp_path / "nested" / "report.json"
    report.write(destination)

    import json
    loaded = json.loads(destination.read_text())
    assert loaded["map50"] == pytest.approx(0.9)
    assert loaded["per_class"][0]["name"] == "car"
    assert "notes" in loaded


def test_table_renders_every_class_with_support():
    metrics = FakeMetrics(
        ap_class_index=[0, 4],
        box=FakeBox(ap50=[0.9, 0.5], ap=[0.7, 0.3],
                    precision=[0.9, 0.5], recall=[0.8, 0.4]),
    )
    report = build_report(
        metrics, weights="w.pt", dataset="d.yaml",
        support=make_support(person_sitting={"instances": 56, "images": 27, "drives": 1}),
    )
    table = report.table()

    assert "car" in table and "person_sitting" in table
    assert "drives" in table
    assert "low confidence" in table
