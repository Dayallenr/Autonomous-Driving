"""
Detection metrics: mAP and inference throughput.

Two numbers are claimed for the perception model — mAP on KITTI and FPS — and
they are measured very differently, so they are separated here.

mAP
---
Computed with the standard VOC/COCO protocol: per class, sort detections by
confidence, greedily match each to the highest-IoU unmatched ground-truth box
above the IoU threshold, then integrate precision over recall. Unmatched
detections are false positives; unmatched ground truth are false negatives.

The greedy-by-confidence matching is not a detail — matching by IoU alone would
let a low-confidence duplicate steal a ground-truth box from the confident
detection that should have claimed it, which inflates recall at low precision
and quietly changes the whole curve.

``mAP@0.5`` uses a single IoU threshold. ``mAP@0.5:0.95`` averages over ten
thresholds and is the stricter COCO-style number; both are reported because a
model tuned for loose localization looks much better under the former.

FPS
---
Throughput is **hardware- and batch-specific**, so a number without both stated
is meaningless. The harness records device, batch size, input resolution, and
precision alongside the measurement, and discards warmup iterations — the first
few passes include lazy kernel compilation and allocator growth, and including
them understates steady-state throughput substantially.

Latency is reported as percentiles rather than a mean. A mean hides the tail,
and for a 20 Hz control loop the tail is what causes dropped frames.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BoxMatch",
    "DetectionMetrics",
    "ThroughputMetrics",
    "average_precision",
    "compute_map",
    "iou_matrix",
    "measure_throughput",
]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``[x1, y1, x2, y2]`` boxes.

    Returns an ``(len(a), len(b))`` matrix. Vectorized because the naive double
    loop dominates evaluation time on a few thousand images.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)

    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])

    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - intersection

    # Degenerate (zero-area) boxes would divide by zero; IoU with them is 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


@dataclass
class BoxMatch:
    """One detection's outcome after matching."""

    confidence: float
    true_positive: bool


def average_precision(matches: list[BoxMatch], num_ground_truth: int) -> float:
    """Area under the precision-recall curve, VOC-2010 / COCO style.

    Uses all-point interpolation (precision is made monotonically decreasing
    before integrating) rather than the older 11-point sampling. The 11-point
    version systematically overestimates on models with sharp PR curves, and
    mixing the two makes numbers incomparable across papers.
    """
    if num_ground_truth == 0:
        # No ground truth for this class. AP is undefined; returning 0 would
        # drag mAP down for a class that simply is not present, so callers must
        # exclude these classes rather than average them in.
        return 0.0
    if not matches:
        return 0.0

    ordered = sorted(matches, key=lambda match: -match.confidence)
    true_positives = np.cumsum([1 if m.true_positive else 0 for m in ordered], dtype=np.float64)
    false_positives = np.cumsum([0 if m.true_positive else 1 for m in ordered], dtype=np.float64)

    recall = true_positives / num_ground_truth
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-12)

    # Make precision monotonically decreasing from the right.
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    # Integrate with the sentinel endpoints (recall 0 and the final recall).
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


@dataclass
class DetectionMetrics:
    """mAP and per-class AP at one or more IoU thresholds."""

    map_50: float
    map_50_95: float
    per_class_ap_50: dict[str, float] = field(default_factory=dict)
    per_class_support: dict[str, int] = field(default_factory=dict)
    images: int = 0
    detections: int = 0
    ground_truths: int = 0
    #: Classes with no ground truth, excluded from the mAP average.
    absent_classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mAP@0.5": round(self.map_50, 4),
            "mAP@0.5:0.95": round(self.map_50_95, 4),
            "per_class_AP@0.5": {k: round(v, 4) for k, v in self.per_class_ap_50.items()},
            "per_class_support": dict(self.per_class_support),
            "images": self.images,
            "detections": self.detections,
            "ground_truths": self.ground_truths,
            "absent_classes": list(self.absent_classes),
        }


def _match_at_threshold(
    predictions: list[tuple[np.ndarray, float]],
    ground_truth: np.ndarray,
    iou_threshold: float,
) -> list[BoxMatch]:
    """Greedily match predictions (already confidence-sorted) to ground truth."""
    matched = np.zeros(len(ground_truth), dtype=bool)
    matches: list[BoxMatch] = []

    for box, confidence in predictions:
        if len(ground_truth) == 0:
            matches.append(BoxMatch(confidence, False))
            continue
        ious = iou_matrix(box[None, :], ground_truth)[0]
        # Only unmatched ground truth is eligible: one box cannot satisfy two
        # detections, and allowing it would let duplicate boxes both count.
        ious[matched] = -1.0
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            matched[best] = True
            matches.append(BoxMatch(confidence, True))
        else:
            matches.append(BoxMatch(confidence, False))
    return matches


def compute_map(
    predictions_per_image: list[dict],
    ground_truth_per_image: list[dict],
    class_names: dict[int, str],
    *,
    iou_thresholds: tuple[float, ...] | None = None,
) -> DetectionMetrics:
    """Compute mAP over a dataset.

    Args:
        predictions_per_image: One dict per image with keys ``boxes`` (N,4),
            ``scores`` (N,), ``classes`` (N,).
        ground_truth_per_image: One dict per image with ``boxes`` and ``classes``.
        class_names: Class index to name.
        iou_thresholds: Defaults to ``0.50..0.95`` in steps of 0.05.

    Raises:
        ValueError: If the prediction and ground-truth lists differ in length —
            a silent mismatch would score predictions against the wrong images.
    """
    if len(predictions_per_image) != len(ground_truth_per_image):
        raise ValueError(
            f"{len(predictions_per_image)} prediction sets but "
            f"{len(ground_truth_per_image)} ground-truth sets"
        )
    thresholds = iou_thresholds or tuple(round(0.50 + 0.05 * i, 2) for i in range(10))

    ap_by_threshold: dict[float, dict[str, float]] = {}
    support: dict[str, int] = {}
    total_detections = 0

    for threshold in thresholds:
        per_class: dict[str, float] = {}
        for class_index, class_name in class_names.items():
            predictions: list[tuple[np.ndarray, float]] = []
            ground_truth_boxes: list[np.ndarray] = []

            for prediction, truth in zip(
                predictions_per_image, ground_truth_per_image, strict=True
            ):
                p_classes = np.asarray(prediction.get("classes", []))
                p_boxes = np.asarray(prediction.get("boxes", [])).reshape(-1, 4)
                p_scores = np.asarray(prediction.get("scores", []))
                keep = p_classes == class_index
                for box, score in zip(p_boxes[keep], p_scores[keep], strict=True):
                    predictions.append((np.asarray(box, dtype=np.float64), float(score)))

                t_classes = np.asarray(truth.get("classes", []))
                t_boxes = np.asarray(truth.get("boxes", [])).reshape(-1, 4)
                ground_truth_boxes.extend(t_boxes[t_classes == class_index])

            support[class_name] = len(ground_truth_boxes)
            # Sort once globally by confidence: AP is defined over the whole
            # dataset's ranked detections, not per image.
            predictions.sort(key=lambda item: -item[1])
            matches = _match_at_threshold(
                predictions,
                np.asarray(ground_truth_boxes, dtype=np.float64).reshape(-1, 4),
                threshold,
            )
            per_class[class_name] = average_precision(matches, len(ground_truth_boxes))
            if threshold == thresholds[0]:
                total_detections += len(predictions)
        ap_by_threshold[threshold] = per_class

    # Classes absent from the ground truth are excluded rather than counted as
    # zero: averaging in a class that does not appear penalizes the model for
    # something it was never asked to do.
    present = [name for name, count in support.items() if count > 0]
    absent = [name for name, count in support.items() if count == 0]

    def mean_over_present(per_class: dict[str, float]) -> float:
        return sum(per_class[name] for name in present) / len(present) if present else 0.0

    map_50 = mean_over_present(ap_by_threshold.get(0.5, {}))
    map_50_95 = (
        sum(mean_over_present(per_class) for per_class in ap_by_threshold.values())
        / len(ap_by_threshold)
        if ap_by_threshold
        else 0.0
    )

    return DetectionMetrics(
        map_50=map_50,
        map_50_95=map_50_95,
        per_class_ap_50={name: ap_by_threshold.get(0.5, {}).get(name, 0.0) for name in present},
        per_class_support={name: support[name] for name in present},
        images=len(predictions_per_image),
        detections=total_detections,
        ground_truths=sum(support.values()),
        absent_classes=absent,
    )


@dataclass
class ThroughputMetrics:
    """Inference throughput, with the context that makes it interpretable."""

    fps: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    iterations: int
    warmup_iterations: int
    batch_size: int
    input_size: int
    device: str
    precision: str

    def to_dict(self) -> dict:
        return {
            "fps": round(self.fps, 1),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p90_latency_ms": round(self.p90_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "device": self.device,
            "precision": self.precision,
        }

    def summary(self) -> str:
        return (
            f"{self.fps:.1f} FPS ({self.p50_latency_ms:.1f} ms p50, "
            f"{self.p99_latency_ms:.1f} ms p99) @ batch={self.batch_size} "
            f"{self.input_size}px {self.precision} on {self.device}"
        )


def measure_throughput(
    infer,
    *,
    iterations: int = 50,
    warmup: int = 10,
    batch_size: int = 1,
    input_size: int = 640,
    device: str = "cpu",
    precision: str = "fp32",
) -> ThroughputMetrics:
    """Time an inference callable.

    Args:
        infer: Zero-argument callable running one forward pass.
        iterations: Timed iterations.
        warmup: Untimed iterations run first. Non-negotiable: the first passes
            include lazy kernel compilation, autotuning, and allocator growth,
            and counting them understates steady-state throughput badly.

    Raises:
        ValueError: If ``iterations`` is not positive.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")

    for _ in range(max(0, warmup)):
        infer()

    latencies: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        infer()
        latencies.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(latencies)

    def percentile(p: float) -> float:
        rank = max(1, int(round(p / 100.0 * len(ordered))))
        return ordered[min(rank, len(ordered)) - 1]

    mean_latency_ms = sum(latencies) / len(latencies)
    return ThroughputMetrics(
        # FPS counts frames, so a batch of N in one pass is N frames.
        fps=(batch_size * 1000.0 / mean_latency_ms) if mean_latency_ms > 0 else 0.0,
        p50_latency_ms=percentile(50),
        p90_latency_ms=percentile(90),
        p99_latency_ms=percentile(99),
        iterations=iterations,
        warmup_iterations=max(0, warmup),
        batch_size=batch_size,
        input_size=input_size,
        device=device,
        precision=precision,
    )
