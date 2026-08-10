"""
Benchmark the KITTI perception model: mAP and throughput.

    python -m pathfinder.benchmark_detector --throughput
    python -m pathfinder.benchmark_detector --map --images data/kitti/val

Throughput runs anywhere — it needs only the model. mAP needs labelled images,
so it reports honestly that it cannot run when none are present rather than
printing a number derived from nothing.

On reading the FPS number
-------------------------
Throughput is a property of (model, hardware, batch size, resolution,
precision), not of the model alone. The harness prints all five next to the
number for exactly that reason. A figure measured on CPU or Apple MPS is not
comparable to one measured on a discrete NVIDIA GPU with TensorRT, and quoting
one as though it were the other is the most common way an FPS claim becomes
false.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from pathfinder.metrics.detection import compute_map, measure_throughput

logger = logging.getLogger(__name__)

DEFAULT_MODEL = Path("models/best_perception_model.pt")

#: KITTI classes the shipped model was trained on.
KITTI_CLASSES = {
    0: "car", 1: "van", 2: "truck", 3: "pedestrian",
    4: "person_sitting", 5: "cyclist", 6: "tram", 7: "misc",
}


def select_device(requested: str = "auto") -> str:
    """Pick an inference device, preferring the fastest available.

    MPS is included because this is commonly developed on Apple Silicon, where
    it is markedly faster than CPU — but it is reported explicitly so the number
    is never mistaken for a CUDA measurement.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path: Path, device: str):
    """Load the Ultralytics detector.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        ImportError: If ultralytics is not installed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"perception model not found at {path}. "
            "Train one with the KITTI pipeline (see the Perception section of "
            "README.md), or point --model at an existing checkpoint."
        )
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "ultralytics is required to run the detector benchmark: pip install ultralytics"
        ) from error

    model = YOLO(str(path))
    model.to(device)
    return model


def run_throughput(
    model, *, device: str, batch_size: int, input_size: int, iterations: int, warmup: int
):
    """Measure steady-state inference throughput on synthetic frames.

    Synthetic input is correct here: throughput depends on tensor shape and
    dtype, not on pixel content. Using real images would add JPEG decode time to
    a number that is meant to isolate the forward pass.
    """
    rng = np.random.default_rng(0)
    batch = [
        rng.integers(0, 255, (input_size, input_size, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]

    def infer() -> None:
        model.predict(batch, imgsz=input_size, device=device, verbose=False)

    return measure_throughput(
        infer,
        iterations=iterations,
        warmup=warmup,
        batch_size=batch_size,
        input_size=input_size,
        device=device,
        precision="fp32",
    )


def _load_yolo_labels(label_path: Path, width: int, height: int) -> dict:
    """Read a YOLO-format label file into absolute xyxy boxes.

    YOLO labels are ``class cx cy w h`` normalized to [0,1]; the metric works in
    pixels, so they are denormalized here.
    """
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not label_path.exists():
        return {"boxes": np.zeros((0, 4)), "classes": np.zeros(0, dtype=int)}

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_index = int(float(parts[0]))
        cx, cy, w, h = (float(value) for value in parts[1:5])
        boxes.append(
            [(cx - w / 2) * width, (cy - h / 2) * height,
             (cx + w / 2) * width, (cy + h / 2) * height]
        )
        classes.append(class_index)
    return {
        "boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        "classes": np.asarray(classes, dtype=int),
    }


def run_map(model, images_dir: Path, *, device: str, input_size: int, confidence: float):
    """Compute mAP over a YOLO-layout dataset directory.

    Expects ``<dir>/images/*.png`` and ``<dir>/labels/*.txt``, or images and
    labels side by side.

    Raises:
        FileNotFoundError: If no images are found.
    """
    image_dir = images_dir / "images" if (images_dir / "images").is_dir() else images_dir
    label_dir = images_dir / "labels" if (images_dir / "labels").is_dir() else images_dir

    image_paths = sorted(
        path
        for extension in ("*.png", "*.jpg", "*.jpeg")
        for path in image_dir.glob(extension)
    )
    if not image_paths:
        raise FileNotFoundError(f"no images found under {image_dir}")

    import cv2

    predictions: list[dict] = []
    ground_truth: list[dict] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("could not read %s, skipping", image_path)
            continue
        height, width = image.shape[:2]

        result = model.predict(
            image, imgsz=input_size, conf=confidence, device=device, verbose=False
        )[0]
        boxes = result.boxes
        predictions.append(
            {
                "boxes": boxes.xyxy.cpu().numpy() if boxes is not None else np.zeros((0, 4)),
                "scores": boxes.conf.cpu().numpy() if boxes is not None else np.zeros(0),
                "classes": (
                    boxes.cls.cpu().numpy().astype(int)
                    if boxes is not None
                    else np.zeros(0, dtype=int)
                ),
            }
        )
        ground_truth.append(
            _load_yolo_labels(label_dir / f"{image_path.stem}.txt", width, height)
        )

    return compute_map(predictions, ground_truth, KITTI_CLASSES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the KITTI perception model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--throughput", action="store_true", help="measure FPS")
    parser.add_argument("--map", action="store_true", help="compute mAP (needs labelled images)")
    parser.add_argument("--images", type=Path, help="dataset directory for --map")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.001,
                        help="low by default: mAP integrates the full PR curve, "
                             "and a high threshold truncates it")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if not args.throughput and not args.map:
        args.throughput = True  # the one that always works

    device = select_device(args.device)
    try:
        model = load_model(args.model, device)
    except (FileNotFoundError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    report: dict = {"model": str(args.model), "device": device, "classes": KITTI_CLASSES}

    if args.throughput:
        throughput = run_throughput(
            model,
            device=device,
            batch_size=args.batch_size,
            input_size=args.input_size,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        report["throughput"] = throughput.to_dict()
        if args.output == "text":
            print("\nThroughput")
            print(f"  {throughput.summary()}")
            print(
                f"  measured over {throughput.iterations} iterations "
                f"after {throughput.warmup_iterations} warmup passes"
            )
            print(
                "\n  NOTE: FPS is a property of (model, hardware, batch, resolution,\n"
                "        precision). This number is not comparable to one measured on\n"
                "        different hardware or with TensorRT."
            )

    if args.map:
        if not args.images:
            print("error: --map requires --images pointing at a labelled dataset", file=sys.stderr)
            return 1
        try:
            metrics = run_map(
                model, args.images, device=device,
                input_size=args.input_size, confidence=args.confidence,
            )
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        report["detection"] = metrics.to_dict()
        if args.output == "text":
            print(f"\nDetection quality ({metrics.images} images, {metrics.ground_truths} labels)")
            print(f"  mAP@0.5       {metrics.map_50:.4f}")
            print(f"  mAP@0.5:0.95  {metrics.map_50_95:.4f}")
            print("\n  per-class AP@0.5:")
            for name, value in sorted(
                metrics.per_class_ap_50.items(), key=lambda item: -item[1]
            ):
                print(f"    {name:16} {value:.4f}   (n={metrics.per_class_support[name]})")
            if metrics.absent_classes:
                print(
                    f"\n  excluded from mAP (no ground truth): "
                    f"{', '.join(metrics.absent_classes)}"
                )

    if args.output == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
