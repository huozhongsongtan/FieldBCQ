"""Offline instance-aware Boundary IoU and Boundary-PQ-like evaluation at 2 px.

Reference: Cheng et al., CVPR 2021, https://arxiv.org/abs/2103.16562.
Uses min(mask IoU, Boundary IoU) for boundary matching/scoring. Our overlapping
instances require one-to-one assignment, so this is not official panoptic PQ.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_agent_experiments import instances, write_csv
from evaluate_mechanism_metrics import pairwise_overlap, read_yolo_masks
from run_agent_experiments import fingerprint, load_parcels, read_json, write_json


def inner_band(mask, width):
    padded = np.pad(np.asarray(mask, np.uint8), 1)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=width,
                        borderType=cv2.BORDER_CONSTANT, borderValue=0)[1:-1, 1:-1]
    return np.asarray(mask, bool) & ~eroded.astype(bool)


def matches(matrix, threshold, strict=False):
    valid = matrix > threshold if strict else matrix >= threshold
    weight = valid * (min(matrix.shape, default=0) + 1 + matrix)
    rows, cols = linear_sum_assignment(-weight)
    return [(r, c) for r, c in zip(rows, cols) if valid[r, c]]


def measure_boundary(gt, pred, width=2):
    iou, _, _ = pairwise_overlap(gt.masks, pred.masks, gt.classes, pred.classes)
    biou = np.zeros_like(iou)
    gt_bands, pred_bands = {}, {}
    # No pair with mask IoU < .5 can become a Boundary-PQ match; it also cannot
    # enter the mask-matched descriptive Boundary IoU below.
    for row, col in np.argwhere(iou >= .5):
        if row not in gt_bands:
            gt_bands[row] = inner_band(gt.masks[row], width)
        if col not in pred_bands:
            pred_bands[col] = inner_band(pred.masks[col], width)
        a, b = gt_bands[row], pred_bands[col]
        union = np.count_nonzero(a | b)
        biou[row, col] = np.count_nonzero(a & b) / union if union else 0.0
    mask_pairs = matches(iou, .5)
    combined = np.minimum(iou, biou)
    boundary_pairs = matches(combined, .5, strict=True)
    tp = len(boundary_pairs)
    fp, fn = len(pred.masks) - tp, len(gt.masks) - tp
    return {"boundary_tp": tp, "boundary_fp": fp, "boundary_fn": fn,
            "boundary_quality_sum": float(sum(combined[r, c] for r, c in boundary_pairs)),
            "mask_matched_pairs": len(mask_pairs),
            "mask_matched_boundary_iou_sum": float(sum(biou[r, c] for r, c in mask_pairs))}


def summarize(counts):
    denominator = counts["boundary_tp"] + .5 * (counts["boundary_fp"] + counts["boundary_fn"])
    matched = counts["mask_matched_pairs"]
    return dict(counts,
                boundary_pq_like_2px=counts["boundary_quality_sum"] / denominator if denominator else None,
                mean_boundary_iou_on_mask_matches_2px=counts["mask_matched_boundary_iou_sum"] / matched if matched else None)


def evaluate(root, dataset_root):
    if not (root / "completed.json").exists():
        raise ValueError("Run must be complete")
    manifest = read_json(root / "manifest.json")
    rows, summaries = [], []
    gt_cache, result_cache = {}, {}
    for method in manifest["primary_methods"]:
        total = None
        for entry in manifest["inputs"]:
            stem = entry["image_id"]
            path = root / method / "masks" / (stem + ".npz")
            key = (stem, fingerprint(path))
            if key not in result_cache:
                parcels, shape = load_parcels(path)
                if stem not in gt_cache:
                    gt_cache[stem] = read_yolo_masks(dataset_root / "labels" / "val" / (stem + ".txt"), *shape)
                result_cache[key] = measure_boundary(gt_cache[stem], instances(parcels))
            counts = result_cache[key]
            total = dict(counts) if total is None else {k: total[k] + counts[k] for k in total}
            rows.append(dict(method=method, image_id=stem, **summarize(counts)))
        summaries.append(dict(method=method, **summarize(total)))
        print(f"[boundary] {method}: {summaries[-1]['boundary_pq_like_2px']:.6f}", flush=True)
    output = root / "evaluation"
    output.mkdir(exist_ok=True)
    write_csv(output / "boundary_per_image.csv", rows)
    write_json(output / "boundary_summary.json", summaries)
    write_csv(output / "boundary_summary.csv", summaries)
    write_json(output / "boundary_protocol.json", {
        "evaluator_sha256": fingerprint(Path(__file__)), "band_width_pixels": 2,
        "band": "mask minus width iterations of padded 3x3 erosion; includes interior holes",
        "boundary_matching": "one-to-one maximum valid cardinality, then score; min(mask IoU, boundary IoU) > 0.5",
        "mask_matching_for_descriptive_mean": "IoU >= 0.5; conditional matched-pair mean, read with instance F1 and boundary PQ-like",
        "note": "Overlapping-instance adaptation, fixed pixel band; not official panoptic benchmark protocol",
        "reference": "https://arxiv.org/abs/2103.16562"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.runs, args.dataset_root)
