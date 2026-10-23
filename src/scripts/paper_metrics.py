"""Explicit fixed-grid instance/contour/area metrics, independent of the GPU stack.

All masks are binary in a square, aspect-preserving letterboxed evaluation grid.
Boundary matching uses square (Chebyshev) tolerance bands. Unmatched instances
contribute contour pixels to denominators, not to correct boundary counts.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


def ratio(a, b):
    return float(a / b) if b else 0.0


def boundary(mask):
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure, border_value=0) & ~binary_erosion(
        mask, structure=structure, border_value=1
    )


def expand(mask, radius):
    return binary_dilation(mask, structure=np.ones((2 * radius + 1,) * 2, dtype=bool)) if radius else mask.copy()


def intersections(predicted, targets):
    """Use bounding-box gating to avoid comparing every pixel in disjoint parcels."""
    def boxes(masks):
        result = []
        for mask in masks:
            y, x = np.nonzero(mask)
            result.append((x.min(), y.min(), x.max() + 1, y.max() + 1) if len(x) else (0, 0, 0, 0))
        return result
    result = np.zeros((len(predicted), len(targets)), dtype=np.int64)
    pb, gb = boxes(predicted), boxes(targets)
    for p, box in enumerate(pb):
        for g, other in enumerate(gb):
            x1, y1 = max(box[0], other[0]), max(box[1], other[1])
            x2, y2 = min(box[2], other[2]), min(box[3], other[3])
            if x1 < x2 and y1 < y2:
                result[p, g] = np.count_nonzero(predicted[p, y1:y2, x1:x2] & targets[g, y1:y2, x1:x2])
    return result


def greedy_pairs(iou, scores, same_class, threshold=0.5):
    """Score order, then highest available same-class mask IoU; no GT reuse."""
    used, pairs = set(), []
    for p in np.argsort(-scores, kind="stable"):
        candidates = [g for g in range(iou.shape[1]) if g not in used and same_class[p, g] and iou[p, g] >= threshold]
        if candidates:
            g = max(candidates, key=lambda candidate: iou[p, candidate])
            used.add(g)
            pairs.append((int(p), g))
    return pairs


def image_metrics(predicted, targets, scores, pred_classes, target_classes,
                  tolerance=2, small_area=1024, match_iou=0.5, sampling_metres=None):
    if predicted.ndim != 3 or targets.ndim != 3 or predicted.shape[1:] != targets.shape[1:]:
        raise ValueError("Masks must be N,H,W arrays on the same evaluation grid.")
    predicted, targets = predicted.astype(bool), targets.astype(bool)
    pa = predicted.sum((1, 2)).astype(float)
    ga = targets.sum((1, 2)).astype(float)
    intersection = intersections(predicted, targets)
    union = pa[:, None] + ga[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
    same = np.asarray(pred_classes)[:, None] == np.asarray(target_classes)[None, :]
    pairs = greedy_pairs(iou, scores, same, match_iou)
    covered_gt = np.divide(intersection, ga[None, :], out=np.zeros_like(union), where=ga[None, :] > 0)
    inside_gt = np.divide(intersection, pa[:, None], out=np.zeros_like(union), where=pa[:, None] > 0)
    pb, gb = [boundary(mask) for mask in predicted], [boundary(mask) for mask in targets]
    pband, gband = [expand(mask, tolerance) for mask in pb], [expand(mask, tolerance) for mask in gb]
    correct_p = correct_g = band_intersection = distance_n = 0
    distance_sum = physical_distance_sum = 0.0
    physical_distance_n = 0
    matched_area_absolute = matched_area_relative = 0.0
    for p, g in pairs:
        correct_p += np.count_nonzero(pb[p] & gband[g])
        correct_g += np.count_nonzero(gb[g] & pband[p])
        band_intersection += np.count_nonzero(pband[p] & gband[g])
        if pb[p].any() and gb[g].any():
            distance_sum += distance_transform_edt(~gb[g])[pb[p]].sum() + distance_transform_edt(~pb[p])[gb[g]].sum()
            count = int(pb[p].sum() + gb[g].sum())
            distance_n += count
            if sampling_metres is not None:
                physical_distance_sum += (
                    distance_transform_edt(~gb[g], sampling=sampling_metres)[pb[p]].sum()
                    + distance_transform_edt(~pb[p], sampling=sampling_metres)[gb[g]].sum()
                )
                physical_distance_n += count
        matched_area_absolute += abs(pa[p] - ga[g])
        matched_area_relative += ratio(abs(pa[p] - ga[g]), ga[g])
    best_iou = np.where(same, iou, 0).max(axis=1) if len(targets) else np.zeros(len(predicted))
    score_bins = np.minimum((np.asarray(scores) * 10).astype(int), 9)
    bins = [{"n": int((score_bins == b).sum()), "score_sum": float(np.asarray(scores)[score_bins == b].sum()),
             "iou_sum": float(best_iou[score_bins == b].sum())} for b in range(10)]
    pred_union_area = int(np.any(predicted, axis=0).sum())
    gt_union_area = int(np.any(targets, axis=0).sum())
    small = ga <= small_area
    return {
        "images": 1, "gt_instances": len(targets), "pred_instances": len(predicted),
        "tp": len(pairs), "fp": len(predicted) - len(pairs), "fn": len(targets) - len(pairs),
        "small_gt": int(small.sum()), "small_tp": sum(int(small[g]) for _, g in pairs),
        "merged_predictions": int(((covered_gt >= 0.5) & same).sum(axis=1).clip(0, 2).tolist().count(2)),
        "oversegmented_gt": int((((inside_gt >= 0.5) & (covered_gt >= 0.05) & same).sum(axis=0) >= 2).sum()),
        "boundary_pred_pixels": sum(int(mask.sum()) for mask in pb),
        "boundary_gt_pixels": sum(int(mask.sum()) for mask in gb),
        "boundary_correct_pred": int(correct_p), "boundary_correct_gt": int(correct_g),
        "band_intersection": int(band_intersection),
        "band_union": sum(int(mask.sum()) for mask in pband + gband) - int(band_intersection),
        "boundary_distance_sum_px": float(distance_sum), "boundary_distance_n": int(distance_n),
        "boundary_distance_sum_m": float(physical_distance_sum), "boundary_distance_m_n": physical_distance_n,
        "matched_area_absolute_error_sum_px": float(matched_area_absolute),
        "matched_area_relative_error_sum": float(matched_area_relative),
        "count_absolute_error": abs(len(predicted) - len(targets)),
        "foreground_area_absolute_error_px": abs(pred_union_area - gt_union_area),
        "foreground_area_bias_px": pred_union_area - gt_union_area,
        "foreground_gt_area_px": gt_union_area,
        "foreground_area_absolute_error_m2": abs(pred_union_area - gt_union_area) * np.prod(sampling_metres) if sampling_metres else 0.0,
        "georeferenced_images": int(sampling_metres is not None), "calibration_bins": bins,
    }


def aggregate(records):
    if not records:
        raise ValueError("Cannot aggregate an empty image group.")
    sums = {key: sum(row[key] for row in records) for key in records[0] if key != "calibration_bins"}
    bins = [{key: sum(row["calibration_bins"][index][key] for row in records) for key in ("n", "score_sum", "iou_sum")}
            for index in range(10)]
    p, r = ratio(sums["tp"], sums["pred_instances"]), ratio(sums["tp"], sums["gt_instances"])
    bp = ratio(sums["boundary_correct_pred"], sums["boundary_pred_pixels"])
    br = ratio(sums["boundary_correct_gt"], sums["boundary_gt_pixels"])
    return {
        "counts": sums, "precision": p, "recall": r, "f1": ratio(2 * p * r, p + r),
        "boundary_precision": bp, "boundary_recall": br, "boundary_f1": ratio(2 * bp * br, bp + br),
        "boundary_band_iou": ratio(sums["band_intersection"], sums["band_union"]),
        "merge_rate": ratio(sums["merged_predictions"], sums["pred_instances"]),
        "oversegmentation_rate": ratio(sums["oversegmented_gt"], sums["gt_instances"]),
        "small_parcel_recall": ratio(sums["small_tp"], sums["small_gt"]) if sums["small_gt"] else None,
        "score_iou_ece": sum(abs(b["score_sum"] - b["iou_sum"]) for b in bins) / sums["pred_instances"] if sums["pred_instances"] else None,
        "calibration_bins": bins,
        "matched_boundary_mean_distance_px": ratio(sums["boundary_distance_sum_px"], sums["boundary_distance_n"]) if sums["boundary_distance_n"] else None,
        "matched_boundary_mean_distance_m": ratio(sums["boundary_distance_sum_m"], sums["boundary_distance_m_n"]) if sums["boundary_distance_m_n"] else None,
        "matched_parcel_area_mae_px": ratio(sums["matched_area_absolute_error_sum_px"], sums["tp"]) if sums["tp"] else None,
        "matched_parcel_area_mape": ratio(sums["matched_area_relative_error_sum"], sums["tp"]) if sums["tp"] else None,
        "parcel_count_mae_per_image": ratio(sums["count_absolute_error"], sums["images"]),
        "foreground_area_mae_px_per_image": ratio(sums["foreground_area_absolute_error_px"], sums["images"]),
        "foreground_area_wape": ratio(sums["foreground_area_absolute_error_px"], sums["foreground_gt_area_px"]) if sums["foreground_gt_area_px"] else None,
        "foreground_area_bias_px": sums["foreground_area_bias_px"],
        "foreground_area_mae_m2_per_georeferenced_image": ratio(sums["foreground_area_absolute_error_m2"], sums["georeferenced_images"]) if sums["georeferenced_images"] else None,
    }
