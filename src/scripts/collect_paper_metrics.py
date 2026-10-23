"""Collect native AP, fixed-threshold contours/structure/application metrics and provenance.

Example: python src/scripts/collect_paper_metrics.py --weights baps=results/train/baps_seed0/weights/best.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path

from quality_scoring import QUALITY_MODES, set_quality_scoring
from project_paths import project_path, relative_or_absolute
from paper_runtime import EVALUATOR_VERSION, benchmark, code_provenance, configuration_of, environment, sha256, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--data", default="data/datasets.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--project", default="results/paper")
    parser.add_argument("--metadata-csv", help="Columns: image,source,country,resolution_m,pixel_size_x_m,pixel_size_y_m")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--conf", type=float, default=0.25, help="Fixed reporting threshold; never tuned on test.")
    parser.add_argument("--predict-conf", type=float, default=0.001, help="Low score floor for native AP and prediction.")
    parser.add_argument("--iou", type=float, default=0.70, help="Box NMS IoU.")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=710)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--small-area", type=int, default=1024, help="GT area in square letterboxed evaluation pixels.")
    scoring = parser.add_mutually_exclusive_group()
    scoring.add_argument("--quality-blend", type=float, choices=(0.0, 1.0), help="Legacy C0/C2 switch.")
    scoring.add_argument("--quality-mode", choices=(*QUALITY_MODES, "all"),
                         help="class/region/boundary/joint; all evaluates each checkpoint four times.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--plots", action="store_true", help="Save native PR/F1 plots; off by default.")
    args = parser.parse_args()
    if args.imgsz <= 0 or args.imgsz % 32 or min(args.batch, args.max_det, args.iterations) < 1:
        parser.error("imgsz must be positive and divisible by 32; batch/max-det/iterations must be positive")
    if not 0 <= args.predict_conf <= args.conf <= 1 or not 0 < args.match_iou <= 1 or not 0 < args.iou <= 1:
        parser.error("Require 0 <= predict-conf <= conf <= 1 and IoU thresholds in (0,1]")
    if min(args.boundary_tolerance, args.small_area, args.warmup) < 0:
        parser.error("boundary-tolerance, small-area and warmup must be nonnegative")
    return args


def load_metadata(path, root, images):
    if not path:
        return {}
    metadata = {}
    with project_path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image" not in (reader.fieldnames or []):
            raise ValueError("metadata CSV requires an image column, relative to dataset root or absolute")
        for row in reader:
            image = Path(row["image"])
            image = (root / image).resolve() if not image.is_absolute() else image.resolve()
            if image in metadata:
                raise ValueError(f"Duplicate metadata: {image}")
            metadata[image] = row
    missing = [str(image) for image in images if image not in metadata]
    if missing:
        raise ValueError(f"Metadata missing for {len(missing)} evaluation images; first: {missing[0]}")
    return metadata


def load_targets(label, original_shape, size):
    import cv2
    import numpy as np
    height, width = original_shape
    scale = min(size / height, size / width)
    left = round((size - round(width * scale)) / 2 - 0.1)
    top = round((size - round(height * scale)) / 2 - 0.1)
    classes, masks = [], []
    for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = np.asarray([float(item) for item in line.split()])
        if len(values) < 7 or (len(values) - 1) % 2 or not np.isfinite(values).all():
            raise ValueError(f"Malformed polygon: {label}:{line_number}")
        if values[0] < 0 or values[0] != int(values[0]) or (values[1:] < 0).any() or (values[1:] > 1).any():
            raise ValueError(f"Class must be an integer and polygon coordinates normalized: {label}:{line_number}")
        points = values[1:].reshape(-1, 2) * [width * scale, height * scale] + [left, top]
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        if not mask.any():
            raise ValueError(f"Polygon has no rasterized area: {label}:{line_number}")
        classes.append(int(values[0]))
        masks.append(mask.astype(bool))
    return np.asarray(masks, dtype=bool).reshape(-1, size, size), np.asarray(classes, dtype=int), scale


def square_evaluation_masks(masks, original_shape, size):
    """Translate rect input-grid masks to the square GT grid, preserving every pixel.

    Prediction uses the same longest-side scale as the square grid, but can omit
    stride-aligned padding. Restore only that padding; never stretch contours.
    """
    import numpy as np
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or size <= 0 or min(original_shape) <= 0:
        raise ValueError("Expected N,H,W masks and positive original/image sizes")
    height, width = original_shape
    gain = min(size / height, size / width)
    content_h, content_w = round(height * gain), round(width * gain)
    mask_h, mask_w = masks.shape[1:]
    if not (content_h <= mask_h <= size and content_w <= mask_w <= size):
        raise ValueError(f"Mask grid {masks.shape[1:]} is incompatible with imgsz={size}, original={original_shape}")
    square_top = round((size - content_h) / 2 - 0.1)
    square_left = round((size - content_w) / 2 - 0.1)
    rect_top = round((mask_h - content_h) / 2 - 0.1)
    rect_left = round((mask_w - content_w) / 2 - 0.1)
    top, left = square_top - rect_top, square_left - rect_left
    if top < 0 or left < 0 or top + mask_h > size or left + mask_w > size:
        raise ValueError("Rectangular padding does not align with the square evaluation grid")
    if (mask_h, mask_w) == (size, size):
        return masks
    output = np.zeros((len(masks), size, size), dtype=bool)
    output[:, top:top + mask_h, left:left + mask_w] = masks
    return output


def metric_values(result):
    output = {}
    for key, value in (("box", result.box), ("mask", result.seg)):
        p, r = float(value.mp), float(value.mr)
        output[key] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                       "map50": float(value.map50), "map75": float(value.map75), "map50_95": float(value.map)}
    return output


def collect_one(args, name, weights, data_path, root, images, labels, metadata, manifest, dataset_hash):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from paper_metrics import aggregate, image_metrics
    output = project_path(args.project) / name
    if output.exists():
        raise FileExistsError(f"Refusing to mix evaluations in existing directory: {output}; choose another --project/name")
    wrapper = YOLO(str(weights))
    actual = configuration_of(wrapper.model)
    scoring = set_quality_scoring(wrapper.model, args.quality_mode, args.quality_blend)
    output.mkdir(parents=True)
    stored_args = dict((wrapper.ckpt or {}).get("train_args", getattr(wrapper.model, "args", {})))
    checkpoint_hash = sha256(weights)
    half = args.precision == "fp16"
    common = dict(imgsz=args.imgsz, batch=args.batch, device=args.device, half=half,
                  conf=args.predict_conf, iou=args.iou, max_det=args.max_det, rect=True, augment=False, verbose=False)
    native = wrapper.val(data=str(data_path), split=args.split, workers=args.workers,
                         project=str(output), name="native", plots=args.plots, **common)
    native_values = metric_values(native)
    # Validation constructs a backend, which can change the model dtype/device. Prediction
    # uses a fresh wrapper, with the same explicit quality-blend override.
    del wrapper
    gc.collect()
    wrapper = YOLO(str(weights))
    set_quality_scoring(wrapper.model, args.quality_mode, args.quality_blend)
    records, groups, image_rows, sample_images = [], {}, [], []
    for start in range(0, len(images), args.batch):
        selected = images[start:start + args.batch]
        predictions = wrapper.predict(source=[str(p) for p in selected], save=False, retina_masks=False, **common)
        if len(predictions) != len(selected):
            raise RuntimeError("Prediction count does not match requested image count")
        for path, prediction in zip(selected, predictions):
            gt, gt_class, scale = load_targets(labels[path], prediction.orig_shape, args.imgsz)
            if len(sample_images) < args.batch:
                sample_images.append(prediction.orig_img.copy())
            if prediction.masks is None:
                masks = np.zeros((0, args.imgsz, args.imgsz), dtype=bool)
                scores, classes = np.zeros(0), np.zeros(0, dtype=int)
            else:
                masks = prediction.masks.data.cpu().numpy().astype(bool)
                scores = prediction.boxes.conf.cpu().numpy()
                classes = prediction.boxes.cls.cpu().numpy().astype(int)
                masks = square_evaluation_masks(masks, prediction.orig_shape, args.imgsz)
            keep = scores >= args.conf
            meta = metadata.get(path, {})
            px, py = meta.get("pixel_size_x_m"), meta.get("pixel_size_y_m")
            sampling = None
            if bool(px) != bool(py):
                raise ValueError(f"Both pixel_size_x_m and pixel_size_y_m are required: {path}")
            if px and py:
                sampling = (float(py) / scale, float(px) / scale)
                if not all(np.isfinite(v) and v > 0 for v in sampling):
                    raise ValueError(f"Pixel sizes must be positive finite metres per original pixel: {path}")
            record = image_metrics(masks[keep], gt, scores[keep], classes[keep], gt_class,
                                   args.boundary_tolerance, args.small_area, args.match_iou, sampling)
            records.append(record)
            for column in ("source", "country", "resolution_m"):
                value = meta.get(column)
                if value:
                    groups.setdefault(f"{column}={value}", []).append(record)
            row = {"image": relative_or_absolute(path, root),
                   **{key: meta.get(key) for key in ("source", "country", "resolution_m")},
                   **{key: value for key, value in record.items() if key != "calibration_bins"}}
            image_rows.append(row)
        print(f"{name}: fixed-threshold metrics {min(start + args.batch, len(images))}/{len(images)}", flush=True)
    with (output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)
    fixed = aggregate(records)
    protocol = {"version": EVALUATOR_VERSION, "imgsz": args.imgsz, "split": args.split,
                "rect": True, "batch": args.batch, "augment": False,
                "grid": "rectangular predictions translated without resampling to square letterbox; integer rasterized GT polygons",
                "conf": args.conf, "predict_conf": args.predict_conf, "nms_iou": args.iou,
                "match_iou": args.match_iou, "max_det": args.max_det, "small_area": args.small_area,
                "boundary_tolerance": args.boundary_tolerance, "precision": args.precision,
                "boundary_aggregation": "matched contours with unmatched contour pixels in denominators; square tolerance",
                "band_iou": "sum matched band intersections / (sum per-instance band areas - intersections)",
                "ece": "10 equal-width bins; score vs best same-class mask IoU over retained predictions, including FP",
                "native_ap": "Ultralytics native mask evaluator, separate from fixed-threshold matching",
                "distance_and_matched_area": "matched instances only; consult recall and FP/FN alongside these errors"}
    training_metadata = weights.parent.parent / "run_metadata.json"
    training = json.loads(training_metadata.read_text(encoding="utf-8")) if training_metadata.is_file() else None
    report = {
        "run_name": name, "configuration": actual, "seed": stored_args.get("seed"),
        "checkpoint": str(weights), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_train_args": stored_args, "training_metadata": training,
        "environment": environment(), "code": code_provenance(),
        "data": str(data_path), "data_yaml_sha256": sha256(data_path),
        "dataset_content_sha256": dataset_hash, "metadata_csv_sha256": sha256(project_path(args.metadata_csv)) if args.metadata_csv else None,
        "evaluation_protocol": protocol, **scoring,
        "native": native_values, "native_speed_ms": native.speed,
        "fixed": fixed, "groups": {key: aggregate(value) for key, value in groups.items()},
        "efficiency": None,
    }
    write_json(output / "dataset_manifest.json", manifest)
    write_json(output / "paper_report.json", report)  # preserve accuracy results if timing later fails
    del native, predictions, wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not args.skip_benchmark:
        report["efficiency"] = benchmark(weights, args.imgsz, args.batch, args.device, half,
                                         args.warmup, args.iterations, args.predict_conf, args.iou,
                                         args.max_det, sample_images, args.quality_blend, args.quality_mode, rect=True)
        write_json(output / "paper_report.json", report)
    print(f"Saved {output / 'paper_report.json'}", flush=True)


def main():
    args = parse_args()
    import torch
    from ultralytics.data.utils import img2label_paths
    from val_by_source import _resolve_root, resolve_images
    if args.precision == "fp16" and (args.device == "cpu" or not torch.cuda.is_available()):
        raise ValueError("FP16 requires CUDA; choose --precision fp32 for CPU")
    data_path, config, images = resolve_images(args.data, args.split)
    if not images:
        raise ValueError("Evaluation image list is empty")
    root = _resolve_root(data_path, config)
    labels = {image: Path(label) for image, label in zip(images, img2label_paths([str(p) for p in images]))}
    metadata = load_metadata(args.metadata_csv, root, images)
    manifest = []
    for image in images:
        label = labels[image]
        if not label.is_file():
            raise FileNotFoundError(f"Missing label {label}; negative images require an explicit empty label file")
        manifest.append({"image": relative_or_absolute(image, root),
                         "image_sha256": sha256(image), "label_sha256": sha256(label)})
    dataset_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    names = set()
    models = []
    for item in args.weights:
        name, separator, weight = item.partition("=")
        if not separator:
            weight, name = item, Path(item).parent.parent.name
        if not name or name in names or any(c in name for c in ('/', '\\', ':')) or name in {".", ".."}:
            raise ValueError("Supply unique --weights NAME=CHECKPOINT names without path separators")
        names.add(name)
        weights = project_path(weight)
        if not weights.is_file():
            raise FileNotFoundError(weights)
        models.append((name, weights))
    for name, weights in models:
        modes = QUALITY_MODES if args.quality_mode == "all" else (args.quality_mode,)
        for mode in modes:
            selected = copy.copy(args)
            selected.quality_mode = mode
            run_name = name + "_q" + mode if args.quality_mode == "all" else name
            collect_one(selected, run_name, weights, data_path, root, images, labels, metadata, manifest, dataset_hash)


if __name__ == "__main__":
    main()
