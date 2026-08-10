"""
Detector evaluation on the sequence-disjoint KITTI split.

Ultralytics' own validator computes the headline mAP, deliberately: it is the
implementation everyone else's KITTI numbers come from, and reimplementing it
would produce a figure that is not comparable to anything. (``pathfinder.metrics
.detection`` exists for the closed-loop ablation, where mAP has to be computed
over prediction sets the validator never sees.)

What this module adds is **support**. A per-class AP table with no denominators
invites the reader to weigh ``person_sitting`` AP equally against ``car`` AP,
when one rests on 56 instances from a single video drive and the other on
thousands across most of the corpus. Every row here carries instances, images,
and drives, and rows whose support comes from one drive are flagged as
low-confidence rather than quietly listed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pathfinder.data.kitti import CLASS_NAMES

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_DRIVES_FOR_CONFIDENCE",
    "ClassResult",
    "DetectionReport",
    "build_report",
    "evaluate",
]

#: A class backed by fewer than this many distinct drives has an effective
#: sample size far below its instance count — its frames are repeated views of
#: one scene. Two is the minimum at which a per-class AP reflects anything
#: beyond a single recording session.
MIN_DRIVES_FOR_CONFIDENCE = 2


@dataclass(frozen=True)
class ClassResult:
    """One class's AP alongside the evidence behind it."""

    name: str
    ap50: float
    ap50_95: float
    precision: float
    recall: float
    instances: int
    images: int
    drives: int

    @property
    def low_confidence(self) -> bool:
        """True when the AP rests on too few independent recordings to trust."""
        return self.drives < MIN_DRIVES_FOR_CONFIDENCE

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ap50": round(self.ap50, 5),
            "ap50_95": round(self.ap50_95, 5),
            "precision": round(self.precision, 5),
            "recall": round(self.recall, 5),
            "instances": self.instances,
            "images": self.images,
            "drives": self.drives,
            "low_confidence": self.low_confidence,
        }


@dataclass
class DetectionReport:
    """A checkpoint's measured performance, with provenance."""

    weights: str
    dataset: str
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    per_class: list[ClassResult] = field(default_factory=list)
    images: int = 0
    instances: int = 0
    device: str = ""
    evaluated_at: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def low_confidence_classes(self) -> list[str]:
        return [result.name for result in self.per_class if result.low_confidence]

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "dataset": self.dataset,
            "map50": round(self.map50, 5),
            "map50_95": round(self.map50_95, 5),
            "precision": round(self.precision, 5),
            "recall": round(self.recall, 5),
            "images": self.images,
            "instances": self.instances,
            "device": self.device,
            "evaluated_at": self.evaluated_at,
            "per_class": [result.to_dict() for result in self.per_class],
            "low_confidence_classes": self.low_confidence_classes,
            "notes": self.notes,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def table(self) -> str:
        """A fixed-width per-class table, support included."""
        header = (
            f"{'class':<16}{'AP@0.5':>9}{'AP@.5:.95':>11}"
            f"{'inst':>8}{'imgs':>7}{'drives':>8}  note"
        )
        lines = [header, "-" * len(header)]
        for result in self.per_class:
            note = "low confidence: single drive" if result.low_confidence else ""
            lines.append(
                f"{result.name:<16}{result.ap50:>9.3f}{result.ap50_95:>11.3f}"
                f"{result.instances:>8,}{result.images:>7,}{result.drives:>8}  {note}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'mAP (all)':<16}{self.map50:>9.3f}{self.map50_95:>11.3f}"
            f"{self.instances:>8,}{self.images:>7,}"
        )
        return "\n".join(lines)


def build_report(
    metrics,
    *,
    weights: str,
    dataset: str,
    support: dict[str, dict],
    device: str = "",
    notes: list[str] | None = None,
) -> DetectionReport:
    """Assemble a report from an Ultralytics metrics object plus support counts.

    ``metrics.ap_class_index`` lists only the classes that actually appeared in
    the validation set, and it indexes the per-class arrays. Zipping those arrays
    against a static class list instead would silently misalign every row the
    moment one class is absent.
    """
    report = DetectionReport(
        weights=weights,
        dataset=dataset,
        map50=float(metrics.box.map50),
        map50_95=float(metrics.box.map),
        precision=float(metrics.box.mp),
        recall=float(metrics.box.mr),
        device=device,
        evaluated_at=datetime.now(UTC).isoformat(),
        notes=list(notes or ()),
    )

    for position, class_index in enumerate(metrics.ap_class_index):
        name = CLASS_NAMES.get(int(class_index), str(class_index))
        counts = support.get(name, {})
        precision, recall, _f1, _ = metrics.box.class_result(position)
        report.per_class.append(
            ClassResult(
                name=name,
                ap50=float(metrics.box.ap50[position]),
                ap50_95=float(metrics.box.ap[position]),
                precision=float(precision),
                recall=float(recall),
                instances=int(counts.get("instances", 0)),
                images=int(counts.get("images", 0)),
                drives=int(counts.get("drives", 0)),
            )
        )

    report.per_class.sort(key=lambda result: -result.instances)
    report.instances = sum(result.instances for result in report.per_class)
    report.images = max((result.images for result in report.per_class), default=0)

    missing = set(CLASS_NAMES.values()) - {result.name for result in report.per_class}
    if missing:
        # Absent classes are excluded from mAP by the validator. Saying so is the
        # difference between an 8-class number and an n-class number wearing an
        # 8-class label.
        report.notes.append(
            f"classes absent from the validation set and excluded from mAP: "
            f"{', '.join(sorted(missing))}"
        )
    if report.low_confidence_classes:
        report.notes.append(
            f"per-class AP for {', '.join(report.low_confidence_classes)} rests on a "
            f"single video drive and is high-variance"
        )
    return report


def evaluate(
    weights: Path,
    data_yaml: Path,
    *,
    support: dict[str, dict],
    device: str = "",
    image_size: int = 640,
    batch: int = 16,
    notes: list[str] | None = None,
):
    """Run the Ultralytics validator and wrap the result in a report.

    Raises:
        FileNotFoundError: If the checkpoint or dataset config is missing.
        ImportError: If ultralytics is not installed.
    """
    if not weights.exists():
        raise FileNotFoundError(f"checkpoint not found: {weights}")
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"dataset config not found: {data_yaml}. Run scripts/prepare_kitti.py first."
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("ultralytics is required: pip install ultralytics") from error

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=image_size,
        batch=batch,
        device=device or None,
        verbose=False,
        plots=False,
    )
    return build_report(
        metrics,
        weights=str(weights),
        dataset=str(data_yaml),
        support=support,
        device=device,
        notes=notes,
    )
