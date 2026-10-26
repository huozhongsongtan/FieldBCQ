"""Fixed-grid metrics required by Tables 5--9; no distances or physical-area analysis."""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

SCOPES = {'boundary', 'structure', 'small', 'calibration', 'area'}


def ratio(a, b):
    return float(a / b) if b else None


def boundary(mask):
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure, border_value=0) & ~binary_erosion(
        mask, structure=structure, border_value=1)


def expand(mask, radius):
    return binary_dilation(mask, structure=np.ones((2*radius+1,)*2, dtype=bool)) if radius else mask


def intersections(predicted, targets):
    def boxes(masks):
        result = []
        for mask in masks:
            y, x = np.nonzero(mask)
            result.append((x.min(), y.min(), x.max()+1, y.max()+1) if len(x) else (0, 0, 0, 0))
        return result
    result = np.zeros((len(predicted), len(targets)), dtype=np.int64)
    gt_boxes = boxes(targets)
    for p, box in enumerate(boxes(predicted)):
        for g, other in enumerate(gt_boxes):
            x1, y1 = max(box[0], other[0]), max(box[1], other[1])
            x2, y2 = min(box[2], other[2]), min(box[3], other[3])
            if x1 < x2 and y1 < y2:
                result[p, g] = np.count_nonzero(predicted[p, y1:y2, x1:x2] & targets[g, y1:y2, x1:x2])
    return result


def greedy_pairs(iou, scores, same_class, threshold=0.5):
    used, pairs = set(), []
    for p in np.argsort(-scores, kind='stable'):
        candidates = [g for g in range(iou.shape[1]) if g not in used and same_class[p, g] and iou[p, g] >= threshold]
        if candidates:
            g = max(candidates, key=lambda candidate: iou[p, candidate])
            used.add(g)
            pairs.append((int(p), g))
    return pairs


def image_metrics(predicted, targets, scores, pred_classes, target_classes,
                  tolerance=2, small_area=1024, match_iou=0.5, scope=None):
    scope = SCOPES if scope is None else set(scope)
    if scope - SCOPES:
        raise ValueError(f'Unknown metric scopes: {scope - SCOPES}')
    predicted, targets = np.asarray(predicted, bool), np.asarray(targets, bool)
    scores, pred_classes, target_classes = map(np.asarray, (scores, pred_classes, target_classes))
    if predicted.ndim != 3 or targets.ndim != 3 or predicted.shape[1:] != targets.shape[1:]:
        raise ValueError('Masks must be N,H,W arrays on the same evaluation grid')
    if scores.shape != (len(predicted),) or pred_classes.shape != scores.shape or target_classes.shape != (len(targets),):
        raise ValueError('Mask, score and class counts differ')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Scores must be finite probabilities')
    pa, ga = predicted.sum((1, 2)).astype(float), targets.sum((1, 2)).astype(float)
    intersection = intersections(predicted, targets)
    union = pa[:, None] + ga[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
    same = pred_classes[:, None] == target_classes[None, :]
    pairs = greedy_pairs(iou, scores, same, match_iou)
    result = dict(images=1, gt_instances=len(targets), pred_instances=len(predicted), tp=len(pairs))
    if 'boundary' in scope:
        pb, gb = [boundary(m) for m in predicted], [boundary(m) for m in targets]
        result.update(boundary_pred_pixels=sum(int(m.sum()) for m in pb),
                      boundary_gt_pixels=sum(int(m.sum()) for m in gb),
                      boundary_correct_pred=sum(int(np.count_nonzero(pb[p] & expand(gb[g], tolerance))) for p, g in pairs),
                      boundary_correct_gt=sum(int(np.count_nonzero(gb[g] & expand(pb[p], tolerance))) for p, g in pairs))
    if 'structure' in scope:
        covered = np.divide(intersection, ga[None, :], out=np.zeros_like(union), where=ga[None, :] > 0)
        inside = np.divide(intersection, pa[:, None], out=np.zeros_like(union), where=pa[:, None] > 0)
        result.update(merged_predictions=int((((covered >= 0.5) & same).sum(axis=1) >= 2).sum()),
                      oversegmented_gt=int((((inside >= 0.5) & (covered >= 0.05) & same).sum(axis=0) >= 2).sum()))
    if 'small' in scope:
        small = ga <= small_area
        result.update(small_gt=int(small.sum()), small_tp=sum(int(small[g]) for _, g in pairs))
    if 'calibration' in scope:
        best = np.where(same, iou, 0).max(axis=1) if len(targets) else np.zeros(len(predicted))
        bins = np.minimum((scores * 10).astype(int), 9)
        result['calibration_bins'] = [dict(n=int((bins == b).sum()), score_sum=float(scores[bins == b].sum()),
                                           iou_sum=float(best[bins == b].sum())) for b in range(10)]
    if 'area' in scope:
        pred_area, gt_area = int(np.any(predicted, axis=0).sum()), int(np.any(targets, axis=0).sum())
        valid = [(p, g) for p, g in pairs if ga[g] > 0]
        result.update(matched_area_relative_error_sum=sum(float(abs(pa[p]-ga[g])/ga[g]) for p, g in valid),
                      matched_area_n=len(valid), count_absolute_error=abs(len(predicted)-len(targets)),
                      foreground_area_absolute_error_px=abs(pred_area-gt_area),
                      foreground_area_bias_px=pred_area-gt_area, foreground_gt_area_px=gt_area)
    return result


def aggregate(records):
    if not records:
        raise ValueError('Cannot aggregate an empty group')
    sums = {k: sum(r[k] for r in records) for k in records[0] if k != 'calibration_bins'}
    tp, npred, ngt = sums['tp'], sums['pred_instances'], sums['gt_instances']
    result = dict(counts=sums, precision=ratio(tp, npred), recall=ratio(tp, ngt), f1=ratio(2*tp, npred+ngt))
    if 'boundary_correct_pred' in sums:
        cp, cg = sums['boundary_correct_pred'], sums['boundary_correct_gt']
        bp, bg = sums['boundary_pred_pixels'], sums['boundary_gt_pixels']
        result['boundary_f1'] = (2*cp*cg/(cp*bg+cg*bp)) if cp and cg else (0.0 if bp+bg else None)
    if 'merged_predictions' in sums:
        result.update(merge_rate=ratio(sums['merged_predictions'], npred),
                      oversegmentation_rate=ratio(sums['oversegmented_gt'], ngt))
    if 'small_gt' in sums:
        result['small_parcel_recall'] = ratio(sums['small_tp'], sums['small_gt'])
    if 'calibration_bins' in records[0]:
        bins = [{k: sum(r['calibration_bins'][i][k] for r in records) for k in ('n', 'score_sum', 'iou_sum')}
                for i in range(10)]
        result['score_iou_ece'] = ratio(sum(abs(b['score_sum']-b['iou_sum']) for b in bins), npred)
    if 'matched_area_n' in sums:
        result.update(matched_parcel_area_mape=ratio(sums['matched_area_relative_error_sum'], sums['matched_area_n']),
                      parcel_count_mae_per_image=ratio(sums['count_absolute_error'], sums['images']),
                      foreground_area_wape=ratio(sums['foreground_area_absolute_error_px'], sums['foreground_gt_area_px']),
                      foreground_area_net_bias=ratio(sums['foreground_area_bias_px'], sums['foreground_gt_area_px']))
    return result
