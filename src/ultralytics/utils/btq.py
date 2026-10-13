"""Utilities for boundary-topology-quality (BTQ) instance segmentation."""

import torch


def topology_mask_nms(
    masks,
    scores,
    classes,
    boxes=None,
    iou_threshold=0.7,
    max_det=300,
    candidate_limit=1000,
    agnostic=False,
):
    """Suppress duplicate masks while retaining adjacent instances whose boxes overlap."""
    if masks.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (N,H,W), received {tuple(masks.shape)}")

    order = scores.argsort(descending=True)[: max(1, int(candidate_limit))]
    flat_masks = (masks[order] > 0.5).flatten(1).float()
    ordered_classes = classes[order]
    ordered_boxes = boxes[order] if boxes is not None else None
    areas = flat_masks.sum(1)
    suppressed = torch.zeros(order.numel(), dtype=torch.bool, device=order.device)
    kept_positions = []

    for position in range(order.numel()):
        if suppressed[position]:
            continue
        kept_positions.append(position)
        if len(kept_positions) >= max_det:
            break
        if position + 1 >= order.numel():
            continue
        remaining = torch.arange(position + 1, order.numel(), device=order.device)
        remaining = remaining[~suppressed[remaining]]
        if remaining.numel() == 0:
            continue
        same_class = torch.ones_like(remaining, dtype=torch.bool)
        if not agnostic:
            same_class = ordered_classes[remaining] == ordered_classes[position]
        comparable = same_class
        if ordered_boxes is not None:
            left_top = torch.maximum(ordered_boxes[remaining, :2], ordered_boxes[position, :2])
            right_bottom = torch.minimum(ordered_boxes[remaining, 2:4], ordered_boxes[position, 2:4])
            comparable = comparable & ((right_bottom - left_top) > 0).all(1)
        if not comparable.any():
            continue
        comparable_positions = remaining[comparable]
        intersection = (flat_masks[comparable_positions] * flat_masks[position]).sum(1)
        union = areas[comparable_positions] + areas[position] - intersection
        iou = intersection / union.clamp_min(1.0)
        suppressed[comparable_positions[iou > iou_threshold]] = True

    return order[torch.tensor(kept_positions, dtype=torch.long, device=order.device)]
