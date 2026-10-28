#!/usr/bin/env python3
"""Evaluate boundary, merge, over-segmentation, and score calibration metrics.

This script is intended to run from the root of the modified Ultralytics project.
It evaluates multiple instance-segmentation checkpoints with one fixed protocol and
produces dataset-level, per-image, and calibration-bin results.

The four comparable metrics reported for every model are:
  1. tolerant Boundary F1;
  2. merge (adhesion) rate;
  3. over-segmentation rate;
  4. final score-to-true-mask-IoU calibration error (ECE).

For checkpoints whose segmentation head contains the two-channel MQC ``cv5``
branch, the script also captures the raw mask-IoU and boundary-F1 quality scores
before NMS and reports their calibration separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Suppress non-fatal OpenCV warnings for GeoTIFF vendor metadata tags.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}

DEFAULT_MODELS = {
    "round1_baseline": "/home/root1/projects/students/lzw/实验结果/一轮/exp3/weights/best.pt",
    "round1_a": "/home/root1/projects/students/lzw/实验结果/一轮/exp_BoundaryDice/weights/best.pt",
    "round1_ab": "/home/root1/projects/students/lzw/实验结果/一轮/exp_BAS_BoundaryDice/weights/best.pt",
    "round1_abc": (
        "/home/root1/projects/students/lzw/实验结果/一轮/"
        "bas_btq_fusion/f1_bas_mqc_seed0/weights/best.pt"
    ),
    "round2_baseline": "/home/root1/projects/students/lzw/实验结果/二轮/exp/weights/best.pt",
    "round2_a": "/home/root1/projects/students/lzw/实验结果/二轮/a_seed0/weights/best.pt",
    "round2_ab": "/home/root1/projects/students/lzw/实验结果/二轮/ab_seed0/weights/best.pt",
    "round2_abc": "/home/root1/projects/students/lzw/实验结果/二轮/abc_seed0/weights/best.pt",
    "round3_baseline": "/home/root1/projects/students/lzw/实验结果/三轮/baseline_seed0/weights/best.pt",
    "round3_a": "/home/root1/projects/students/lzw/实验结果/三轮/a_seed02/weights/best.pt",
    "round3_ab": "/home/root1/projects/students/lzw/实验结果/三轮/ab_seed02/weights/best.pt",
    "round3_abc": "/home/root1/projects/students/lzw/实验结果/三轮/abc_seed02/weights/best.pt",
}


@dataclass
class InstanceSet:
    masks: List[np.ndarray]
    classes: np.ndarray
    scores: np.ndarray
    mqc_mask_iou: Optional[np.ndarray] = None
    mqc_boundary_f1: Optional[np.ndarray] = None
    mqc_combined: Optional[np.ndarray] = None


@dataclass
class Aggregate:
    images: int = 0
    gt_instances: int = 0
    pred_instances: int = 0
    gt_boundary_pixels: int = 0
    pred_boundary_pixels: int = 0
    gt_boundary_hits: int = 0
    pred_boundary_hits: int = 0
    merge_count: int = 0
    overseg_count: int = 0
    final_scores: List[float] = field(default_factory=list)
    final_iou_targets: List[float] = field(default_factory=list)
    mqc_mask_scores: List[float] = field(default_factory=list)
    mqc_boundary_scores: List[float] = field(default_factory=list)
    mqc_combined_scores: List[float] = field(default_factory=list)
    mqc_iou_targets: List[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate four mechanism metrics for Baseline, A, A+B, and A+B+C checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=WEIGHTS",
        help="Model name and checkpoint. Repeat this argument for multiple models.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/root1/projects/students/lzw/field/datasets"),
        help="Dataset root containing images/<split> and labels/<split>.",
    )
    parser.add_argument("--split", default="test", help="Dataset split directory name.")
    parser.add_argument("--output", type=Path, default=Path("runs/mechanism_metrics"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25, help="Fixed prediction confidence threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=710)
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Boundary matching tolerance in pixels at original image resolution.",
    )
    parser.add_argument(
        "--merge-gt-coverage",
        type=float,
        default=0.50,
        help="A prediction covers a GT instance when intersection/GT area reaches this value.",
    )
    parser.add_argument(
        "--overseg-pred-purity",
        type=float,
        default=0.50,
        help="A fragment belongs to a GT when intersection/prediction area reaches this value.",
    )
    parser.add_argument(
        "--overseg-min-gt-coverage",
        type=float,
        default=0.05,
        help="Minimum intersection/GT area for a predicted fragment.",
    )
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N images for a smoke test; 0 evaluates the full split.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic metric tests and exit.")
    return parser.parse_args()


def parse_models(specs: Sequence[str]) -> Dict[str, Path]:
    if not specs:
        return {name: Path(path) for name, path in DEFAULT_MODELS.items()}
    models: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --model value {spec!r}; expected NAME=WEIGHTS")
        name, weights = spec.split("=", 1)
        name, weights = name.strip(), weights.strip()
        if not name or not weights:
            raise ValueError(f"Invalid --model value {spec!r}; both NAME and WEIGHTS are required")
        if name in models:
            raise ValueError(f"Duplicate model name: {name}")
        models[name] = Path(weights)
    return models


def find_images(image_dir: Path, limit: int = 0) -> List[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No supported images found under: {image_dir}")
    return images[:limit] if limit > 0 else images


def label_path_for_image(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    return (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")


def read_yolo_masks(label_path: Path, height: int, width: int) -> InstanceSet:
    masks: List[np.ndarray] = []
    classes: List[int] = []
    if not label_path.exists():
        return InstanceSet(masks, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"Invalid segmentation label at {label_path}:{line_number}")
        class_id = int(float(fields[0]))
        coords = np.asarray([float(value) for value in fields[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] = np.clip(np.rint(coords[:, 0] * width), 0, width - 1)
        coords[:, 1] = np.clip(np.rint(coords[:, 1] * height), 0, height - 1)
        polygon = coords.astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        if mask.any():
            masks.append(mask.astype(bool))
            classes.append(class_id)
    return InstanceSet(masks, np.asarray(classes, dtype=np.int64), np.empty(0, dtype=np.float32))


def result_instances(result: Any) -> InstanceSet:
    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return InstanceSet([], np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

    masks = result.masks.data.detach().cpu().numpy() > 0.5
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float64)

    keep = np.asarray([bool(mask.any()) for mask in masks], dtype=bool)
    mask_list = [mask.astype(bool, copy=False) for mask, valid in zip(masks, keep) if valid]
    classes = classes[keep]
    scores = scores[keep]

    def optional_scores(attribute: str) -> Optional[np.ndarray]:
        values = getattr(result, attribute, None)
        if values is None:
            return None
        values = np.asarray(values, dtype=np.float64)
        return values[keep]

    return InstanceSet(
        mask_list,
        classes,
        scores,
        mqc_mask_iou=optional_scores("mqc_mask_iou_scores"),
        mqc_boundary_f1=optional_scores("mqc_boundary_f1_scores"),
        mqc_combined=optional_scores("mqc_combined_scores"),
    )


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return 0, 0, 0, 0
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def pairwise_overlap(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    gt_classes: np.ndarray,
    pred_classes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return IoU, GT coverage, and prediction purity matrices."""
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    intersections = np.zeros((n_gt, n_pred), dtype=np.float64)
    gt_areas = np.asarray([np.count_nonzero(mask) for mask in gt_masks], dtype=np.float64)
    pred_areas = np.asarray([np.count_nonzero(mask) for mask in pred_masks], dtype=np.float64)
    gt_boxes = [mask_bbox(mask) for mask in gt_masks]
    pred_boxes = [mask_bbox(mask) for mask in pred_masks]

    for gt_index, (gt_mask, gt_box) in enumerate(zip(gt_masks, gt_boxes)):
        gx1, gy1, gx2, gy2 = gt_box
        for pred_index, (pred_mask, pred_box) in enumerate(zip(pred_masks, pred_boxes)):
            if gt_classes[gt_index] != pred_classes[pred_index]:
                continue
            px1, py1, px2, py2 = pred_box
            x1, y1, x2, y2 = max(gx1, px1), max(gy1, py1), min(gx2, px2), min(gy2, py2)
            if x1 >= x2 or y1 >= y2:
                continue
            intersections[gt_index, pred_index] = np.count_nonzero(
                gt_mask[y1:y2, x1:x2] & pred_mask[y1:y2, x1:x2]
            )

    union = gt_areas[:, None] + pred_areas[None, :] - intersections
    iou = np.divide(intersections, union, out=np.zeros_like(intersections), where=union > 0)
    gt_coverage = np.divide(
        intersections, gt_areas[:, None], out=np.zeros_like(intersections), where=gt_areas[:, None] > 0
    )
    pred_purity = np.divide(
        intersections, pred_areas[None, :], out=np.zeros_like(intersections), where=pred_areas[None, :] > 0
    )
    return iou, gt_coverage, pred_purity


def instance_boundary_union(masks: Sequence[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    boundary_union = np.zeros_like(masks[0], dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for mask in masks:
        mask_u8 = mask.astype(np.uint8, copy=False)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        boundary_union |= mask_u8 & (1 - eroded)
    return boundary_union.astype(bool)


def boundary_counts(
    gt_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray], tolerance: int
) -> Dict[str, int]:
    if gt_masks:
        shape = gt_masks[0].shape
    elif pred_masks:
        shape = pred_masks[0].shape
    else:
        return {"gt_pixels": 0, "pred_pixels": 0, "gt_hits": 0, "pred_hits": 0}

    gt_boundary = instance_boundary_union(gt_masks) if gt_masks else np.zeros(shape, dtype=bool)
    pred_boundary = instance_boundary_union(pred_masks) if pred_masks else np.zeros(shape, dtype=bool)
    if tolerance > 0:
        size = 2 * tolerance + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        dilated_gt = cv2.dilate(gt_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
        dilated_pred = cv2.dilate(pred_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        dilated_gt, dilated_pred = gt_boundary, pred_boundary
    return {
        "gt_pixels": int(np.count_nonzero(gt_boundary)),
        "pred_pixels": int(np.count_nonzero(pred_boundary)),
        "gt_hits": int(np.count_nonzero(gt_boundary & dilated_pred)),
        "pred_hits": int(np.count_nonzero(pred_boundary & dilated_gt)),
    }


def precision_recall_f1(pred_hits: int, pred_total: int, gt_hits: int, gt_total: int) -> Tuple[float, float, float]:
    if pred_total == 0:
        precision = 1.0 if gt_total == 0 else 0.0
    else:
        precision = pred_hits / pred_total
    if gt_total == 0:
        recall = 1.0 if pred_total == 0 else 0.0
    else:
        recall = gt_hits / gt_total
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray, rank: bool = False) -> Optional[float]:
    if len(x) < 2:
        return None
    if rank:
        x, y = average_ranks(x), average_ranks(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def calibration_metrics(scores: Sequence[float], targets: Sequence[float], bins: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    score_array = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    target_array = np.clip(np.asarray(targets, dtype=np.float64), 0.0, 1.0)
    if len(score_array) == 0:
        return {
            "count": 0,
            "ece": None,
            "mae": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
        }, []

    bin_ids = np.minimum((score_array * bins).astype(np.int64), bins - 1)
    rows: List[Dict[str, Any]] = []
    ece = 0.0
    for bin_index in range(bins):
        selected = bin_ids == bin_index
        count = int(np.count_nonzero(selected))
        mean_score = float(score_array[selected].mean()) if count else None
        mean_target = float(target_array[selected].mean()) if count else None
        gap = abs(mean_score - mean_target) if count else None
        if count:
            ece += (count / len(score_array)) * gap
        rows.append(
            {
                "bin": bin_index,
                "lower": bin_index / bins,
                "upper": (bin_index + 1) / bins,
                "count": count,
                "mean_score": mean_score,
                "mean_true_iou": mean_target,
                "absolute_gap": gap,
            }
        )
    errors = score_array - target_array
    metrics = {
        "count": int(len(score_array)),
        "ece": float(ece),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "pearson": correlation(score_array, target_array),
        "spearman": correlation(score_array, target_array, rank=True),
    }
    return metrics, rows


def analyse_image(
    gt: InstanceSet,
    pred: InstanceSet,
    boundary_tolerance: int,
    merge_gt_coverage: float,
    overseg_pred_purity: float,
    overseg_min_gt_coverage: float,
) -> Dict[str, Any]:
    iou, gt_coverage, pred_purity = pairwise_overlap(gt.masks, pred.masks, gt.classes, pred.classes)
    best_pred_iou = iou.max(axis=0) if iou.shape[0] else np.zeros(len(pred.masks), dtype=np.float64)

    merge_flags = (gt_coverage >= merge_gt_coverage).sum(axis=0) >= 2 if len(pred.masks) else np.empty(0, bool)
    fragment_flags = (pred_purity >= overseg_pred_purity) & (gt_coverage >= overseg_min_gt_coverage)
    overseg_flags = fragment_flags.sum(axis=1) >= 2 if len(gt.masks) else np.empty(0, bool)
    boundary = boundary_counts(gt.masks, pred.masks, boundary_tolerance)
    bp, br, bf1 = precision_recall_f1(
        boundary["pred_hits"], boundary["pred_pixels"], boundary["gt_hits"], boundary["gt_pixels"]
    )
    return {
        "gt_count": len(gt.masks),
        "pred_count": len(pred.masks),
        "merge_count": int(np.count_nonzero(merge_flags)),
        "overseg_count": int(np.count_nonzero(overseg_flags)),
        "boundary_precision": bp,
        "boundary_recall": br,
        "boundary_f1": bf1,
        **{f"boundary_{key}": value for key, value in boundary.items()},
        "best_pred_iou": best_pred_iou,
    }


def find_head(auto_backend: Any) -> Optional[Any]:
    model = getattr(auto_backend, "model", None)
    layers = getattr(model, "model", None)
    if layers is not None and len(layers):
        return layers[-1]
    return None


class QualityCapture:
    def __init__(self, head: Any):
        self.head = head
        self.outputs: Dict[int, Any] = {}
        self.handles: List[Any] = []
        self.handles.append(head.register_forward_pre_hook(self._clear))
        for index, branch in enumerate(head.cv5):
            self.handles.append(branch.register_forward_hook(self._make_hook(index)))

    def _clear(self, _module: Any, _inputs: Any) -> None:
        self.outputs.clear()

    def _make_hook(self, index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self.outputs[index] = output.detach()

        return hook

    def logits(self) -> Optional[Any]:
        if len(self.outputs) != len(self.head.cv5):
            return None
        tensors = [self.outputs[index] for index in range(len(self.head.cv5))]
        batch_size = tensors[0].shape[0]
        return __import__("torch").cat([tensor.view(batch_size, 2, -1) for tensor in tensors], dim=2)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def build_quality_predictor_class():
    """Build a predictor compatible with the repository's local Ultralytics version."""
    import torch
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops

    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:  # Ultralytics 8.3.x
        non_max_suppression = ops.non_max_suppression

    class QualitySegmentationPredictor(SegmentationPredictor):
        def setup_model(self, model: Any, verbose: bool = True) -> None:
            super().setup_model(model, verbose=verbose)
            head = find_head(self.model)
            self.quality_capture = QualityCapture(head) if head is not None and hasattr(head, "cv5") else None
            self.mqc_head_name = type(head).__name__ if self.quality_capture is not None else None

        def postprocess(self, preds: Any, img: Any, orig_imgs: Any) -> List[Any]:
            # Ultralytics 8.3.x returns (prediction, auxiliary_tuple), while
            # newer releases may wrap (prediction, prototypes) in preds[0].
            if isinstance(preds[0], tuple):
                prediction, proto = preds[0]
            else:
                prediction = preds[0]
                proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
            quality_logits = self.quality_capture.logits() if self.quality_capture is not None else None
            capture_quality = quality_logits is not None and quality_logits.shape[2] == prediction.shape[2]

            if capture_quality:
                # NMS preserves mask extras. Appending the anchor id lets us map raw MQC
                # logits to the retained instances without changing model source code.
                prediction = prediction.float()
                anchor_ids = torch.arange(prediction.shape[2], device=prediction.device, dtype=prediction.dtype)
                anchor_ids = anchor_ids.view(1, 1, -1).expand(prediction.shape[0], 1, -1)
                prediction = torch.cat((prediction, anchor_ids), dim=1)

            detections = non_max_suppression(
                prediction,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=len(self.model.names),
                classes=self.args.classes,
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

            results = []
            for batch_index, (pred, orig_img, img_path) in enumerate(zip(detections, orig_imgs, self.batch[0])):
                if capture_quality:
                    kept_anchor_ids = pred[:, -1].long()
                    mask_coefficients = pred[:, 6:-1]
                else:
                    kept_anchor_ids = None
                    mask_coefficients = pred[:, 6:]

                if not len(pred):
                    masks = None
                elif self.args.retina_masks:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    masks = ops.process_mask_native(proto[batch_index], mask_coefficients, pred[:, :4], orig_img.shape[:2])
                else:
                    masks = ops.process_mask(
                        proto[batch_index], mask_coefficients, pred[:, :4], img.shape[2:], upsample=True
                    )
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

                result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
                if capture_quality and len(pred):
                    raw = quality_logits[batch_index, :, kept_anchor_ids].sigmoid().float()
                    result.mqc_mask_iou_scores = raw[0].detach().cpu().numpy()
                    result.mqc_boundary_f1_scores = raw[1].detach().cpu().numpy()
                    result.mqc_combined_scores = raw.prod(0).sqrt().detach().cpu().numpy()
                results.append(result)
            return results

    return QualitySegmentationPredictor


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_model(
    name: str,
    weights: Path,
    image_paths: Sequence[Path],
    source_list: Path,
    image_dir: Path,
    label_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import torch
    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint for {name} does not exist: {weights}")

    predictor_class = build_quality_predictor_class()
    model = YOLO(str(weights), task="segment")
    aggregate = Aggregate()
    per_image_rows: List[Dict[str, Any]] = []

    predictions = model.predict(
        # A Python list is interpreted as one in-memory batch by Ultralytics.
        # A text source keeps normal batched streaming for the full test split.
        source=str(source_list),
        stream=True,
        predictor=predictor_class,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        workers=args.workers,
        retina_masks=True,
        save=False,
        verbose=False,
    )

    for image_index, result in enumerate(predictions, start=1):
        result_path = Path(result.path)
        try:
            relative_path = result_path.resolve().relative_to(image_dir.resolve())
            canonical_image_path = image_dir / relative_path
        except ValueError:
            canonical_image_path = next(
                (path for path in image_paths if path.name == result_path.name), result_path
            )
        label_path = label_path_for_image(canonical_image_path, image_dir, label_dir)
        height, width = result.orig_shape
        gt = read_yolo_masks(label_path, height, width)
        pred = result_instances(result)
        image_metrics = analyse_image(
            gt,
            pred,
            args.boundary_tolerance,
            args.merge_gt_coverage,
            args.overseg_pred_purity,
            args.overseg_min_gt_coverage,
        )

        best_iou = image_metrics.pop("best_pred_iou")
        aggregate.images += 1
        aggregate.gt_instances += image_metrics["gt_count"]
        aggregate.pred_instances += image_metrics["pred_count"]
        aggregate.merge_count += image_metrics["merge_count"]
        aggregate.overseg_count += image_metrics["overseg_count"]
        aggregate.gt_boundary_pixels += image_metrics["boundary_gt_pixels"]
        aggregate.pred_boundary_pixels += image_metrics["boundary_pred_pixels"]
        aggregate.gt_boundary_hits += image_metrics["boundary_gt_hits"]
        aggregate.pred_boundary_hits += image_metrics["boundary_pred_hits"]
        aggregate.final_scores.extend(pred.scores.tolist())
        aggregate.final_iou_targets.extend(best_iou.tolist())

        if pred.mqc_mask_iou is not None:
            aggregate.mqc_mask_scores.extend(pred.mqc_mask_iou.tolist())
            aggregate.mqc_boundary_scores.extend(pred.mqc_boundary_f1.tolist())
            aggregate.mqc_combined_scores.extend(pred.mqc_combined.tolist())
            aggregate.mqc_iou_targets.extend(best_iou.tolist())

        per_image_rows.append(
            {
                "model": name,
                "image": str(canonical_image_path.relative_to(image_dir)),
                **image_metrics,
                "merge_rate": safe_rate(image_metrics["merge_count"], image_metrics["pred_count"]),
                "overseg_rate": safe_rate(image_metrics["overseg_count"], image_metrics["gt_count"]),
            }
        )
        if image_index % 100 == 0 or image_index == len(image_paths):
            print(f"[{name}] {image_index}/{len(image_paths)} images", flush=True)

    boundary_precision, boundary_recall, boundary_f1 = precision_recall_f1(
        aggregate.pred_boundary_hits,
        aggregate.pred_boundary_pixels,
        aggregate.gt_boundary_hits,
        aggregate.gt_boundary_pixels,
    )
    final_calibration, final_bins = calibration_metrics(
        aggregate.final_scores, aggregate.final_iou_targets, args.calibration_bins
    )
    mqc_mask_calibration, mqc_mask_bins = calibration_metrics(
        aggregate.mqc_mask_scores, aggregate.mqc_iou_targets, args.calibration_bins
    )
    mqc_combined_calibration, mqc_combined_bins = calibration_metrics(
        aggregate.mqc_combined_scores, aggregate.mqc_iou_targets, args.calibration_bins
    )

    summary = {
        "model": name,
        "weights": str(weights),
        "images": aggregate.images,
        "gt_instances": aggregate.gt_instances,
        "pred_instances": aggregate.pred_instances,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "merge_count": aggregate.merge_count,
        "merge_rate": safe_rate(aggregate.merge_count, aggregate.pred_instances),
        "overseg_count": aggregate.overseg_count,
        "overseg_rate": safe_rate(aggregate.overseg_count, aggregate.gt_instances),
        "score_iou_ece": final_calibration["ece"],
        "score_iou_mae": final_calibration["mae"],
        "score_iou_rmse": final_calibration["rmse"],
        "score_iou_pearson": final_calibration["pearson"],
        "score_iou_spearman": final_calibration["spearman"],
        "mqc_available": bool(aggregate.mqc_mask_scores),
        "mqc_mask_iou_ece": mqc_mask_calibration["ece"],
        "mqc_mask_iou_mae": mqc_mask_calibration["mae"],
        "mqc_mask_iou_rmse": mqc_mask_calibration["rmse"],
        "mqc_mask_iou_pearson": mqc_mask_calibration["pearson"],
        "mqc_mask_iou_spearman": mqc_mask_calibration["spearman"],
        "mqc_combined_iou_ece": mqc_combined_calibration["ece"],
        "mqc_combined_iou_mae": mqc_combined_calibration["mae"],
        "mqc_combined_iou_pearson": mqc_combined_calibration["pearson"],
    }

    calibration_rows = []
    for score_type, rows in (
        ("final_score", final_bins),
        ("mqc_mask_iou", mqc_mask_bins),
        ("mqc_combined", mqc_combined_bins),
    ):
        calibration_rows.extend({"model": name, "score_type": score_type, **row} for row in rows)

    del predictions, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, per_image_rows, calibration_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    def metric(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    lines = [
        "# 四项机制指标汇总",
        "",
        "Boundary F1 越高越好；粘连率、过分割率和校准 ECE 越低越好。",
        "",
        "| 模型 | Boundary F1 ↑ | 粘连率 ↓ | 过分割率 ↓ | 分数-IoU ECE ↓ | MQC-IoU ECE ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {boundary} | {merge} | {overseg} | {score_ece} | {mqc_ece} |".format(
                model=row["model"],
                boundary=metric(row["boundary_f1"]),
                merge=metric(row["merge_rate"]),
                overseg=metric(row["overseg_rate"]),
                score_ece=metric(row["score_iou_ece"]),
                mqc_ece=metric(row["mqc_mask_iou_ece"]),
            )
        )
    lines.extend(
        [
            "",
            "说明：分数-IoU ECE 对所有模型计算；MQC-IoU ECE 仅对含 MQC 质量头的模型计算。",
            "具体阈值和权重路径见 `run_config.json`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_calibration(path: Path, rows: Sequence[Dict[str, Any]], model_names: Sequence[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; calibration CSV was saved but PNG plotting was skipped.")
        return

    columns = 2
    rows_count = math.ceil(len(model_names) / columns)
    figure, axes = plt.subplots(rows_count, columns, figsize=(6.5 * columns, 5 * rows_count), squeeze=False)
    for axis, model_name in zip(axes.flat, model_names):
        axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
        for score_type, label, marker in (
            ("final_score", "final score", "o"),
            ("mqc_mask_iou", "raw MQC mask-IoU", "s"),
            ("mqc_combined", "raw MQC combined", "^"),
        ):
            selected = [
                row
                for row in rows
                if row["model"] == model_name and row["score_type"] == score_type and row["count"]
            ]
            if selected:
                axis.plot(
                    [row["mean_score"] for row in selected],
                    [row["mean_true_iou"] for row in selected],
                    marker=marker,
                    linewidth=1.5,
                    label=label,
                )
        axis.set(title=model_name, xlabel="Mean predicted score", ylabel="Mean true Mask IoU", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(model_names) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_self_test() -> None:
    shape = (32, 32)

    def rectangle(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask

    gt_masks = [rectangle(2, 2, 10, 10), rectangle(12, 2, 20, 10)]
    merged_prediction = [rectangle(2, 2, 20, 10)]
    metrics = analyse_image(
        InstanceSet(gt_masks, np.zeros(2, dtype=np.int64), np.empty(0)),
        InstanceSet(merged_prediction, np.zeros(1, dtype=np.int64), np.asarray([0.9])),
        boundary_tolerance=0,
        merge_gt_coverage=0.5,
        overseg_pred_purity=0.5,
        overseg_min_gt_coverage=0.05,
    )
    assert metrics["merge_count"] == 1, metrics

    gt = [rectangle(2, 14, 20, 24)]
    fragments = [rectangle(2, 14, 10, 24), rectangle(12, 14, 20, 24)]
    metrics = analyse_image(
        InstanceSet(gt, np.zeros(1, dtype=np.int64), np.empty(0)),
        InstanceSet(fragments, np.zeros(2, dtype=np.int64), np.asarray([0.9, 0.9])),
        boundary_tolerance=0,
        merge_gt_coverage=0.5,
        overseg_pred_purity=0.5,
        overseg_min_gt_coverage=0.05,
    )
    assert metrics["overseg_count"] == 1, metrics

    identical = [rectangle(4, 4, 16, 16)]
    counts = boundary_counts(identical, identical, tolerance=0)
    _, _, f1 = precision_recall_f1(
        counts["pred_hits"], counts["pred_pixels"], counts["gt_hits"], counts["gt_pixels"]
    )
    assert abs(f1 - 1.0) < 1e-12, f1
    calibration, _ = calibration_metrics([0.2, 0.8], [0.2, 0.8], bins=5)
    assert calibration["ece"] < 1e-12, calibration

    try:
        import torch

        class FakeQualityHead(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cv5 = torch.nn.ModuleList([torch.nn.Conv2d(3, 2, 1), torch.nn.Conv2d(3, 2, 1)])

            def forward(self, features: Sequence[Any]) -> List[Any]:
                return [branch(feature) for branch, feature in zip(self.cv5, features)]

        fake_head = FakeQualityHead()
        capture = QualityCapture(fake_head)
        fake_head([torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 2, 2)])
        assert tuple(capture.logits().shape) == (1, 2, 20)
        capture.close()
    except ImportError:
        pass
    print("Synthetic self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.boundary_tolerance < 0:
        raise ValueError("--boundary-tolerance must be non-negative")
    if args.calibration_bins < 2:
        raise ValueError("--calibration-bins must be at least 2")

    models = parse_models(args.model)
    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split
    image_paths = find_images(image_dir, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    source_list = args.output / "evaluation_images.txt"
    source_list.write_text("\n".join(str(path.resolve()) for path in image_paths) + "\n", encoding="utf-8")

    config = {
        "models": {name: str(path) for name, path in models.items()},
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "image_count": len(image_paths),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "boundary_tolerance": args.boundary_tolerance,
        "merge_gt_coverage": args.merge_gt_coverage,
        "overseg_pred_purity": args.overseg_pred_purity,
        "overseg_min_gt_coverage": args.overseg_min_gt_coverage,
        "calibration_bins": args.calibration_bins,
    }
    (args.output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summaries: List[Dict[str, Any]] = []
    per_image_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    print(f"Evaluating {len(models)} models on {len(image_paths)} images")
    for index, (name, weights) in enumerate(models.items(), start=1):
        print(f"\n[{index}/{len(models)}] {name}: {weights}", flush=True)
        summary, image_rows, bin_rows = evaluate_model(
            name, weights, image_paths, source_list, image_dir, label_dir, args
        )
        summaries.append(summary)
        per_image_rows.extend(image_rows)
        calibration_rows.extend(bin_rows)
        write_csv(args.output / "mechanism_summary.csv", summaries)
        write_csv(args.output / "per_image_metrics.csv", per_image_rows)
        write_csv(args.output / "calibration_bins.csv", calibration_rows)
        (args.output / "mechanism_metrics.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    plot_calibration(args.output / "calibration_curves.png", calibration_rows, list(models))
    write_markdown_summary(args.output / "mechanism_summary.md", summaries)
    print("\nFour-metric summary")
    print("model,boundary_f1,merge_rate,overseg_rate,score_iou_ece,mqc_mask_iou_ece")
    for row in summaries:
        def format_optional(value: Optional[float]) -> str:
            return "NA" if value is None else f"{value:.6f}"

        values = [
            row["model"],
            f'{row["boundary_f1"]:.6f}',
            f'{row["merge_rate"]:.6f}',
            f'{row["overseg_rate"]:.6f}',
            format_optional(row["score_iou_ece"]),
            format_optional(row["mqc_mask_iou_ece"]),
        ]
        print(",".join(values))
    print(f"\nSaved results to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
