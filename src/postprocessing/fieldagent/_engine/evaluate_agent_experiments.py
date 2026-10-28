"""Offline, class-aware evaluation of cached FieldBQC repair experiments."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_mechanism_metrics import (InstanceSet, boundary_counts, pairwise_overlap,
                                        precision_recall_f1, read_yolo_masks)
from fieldbqc_agent import _component_count, _contour_vertices, _mask_iou
from run_agent_experiments import fingerprint, load_parcels, read_json, write_json


def instances(parcels):
    return InstanceSet([p.mask for p in parcels], np.asarray([p.cls for p in parcels], dtype=np.int64),
                       np.asarray([p.score for p in parcels]))


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def summary(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    bp, br, bf1 = precision_recall_f1(counts["boundary_pred_hits"], counts["boundary_pred_pixels"],
                                    counts["boundary_gt_hits"], counts["boundary_gt_pixels"])
    return dict(counts, instance_precision=divide(tp, tp + fp), instance_recall=divide(tp, tp + fn),
                instance_f1=divide(2 * tp, 2 * tp + fp + fn),
                pq=divide(counts["matched_iou_sum"], tp + 0.5 * fp + 0.5 * fn),
                union_iou=divide(counts["intersection_pixels"], counts["union_pixels"]),
                boundary_precision=bp, boundary_recall=br, boundary_f1=bf1,
                adhesion_rate=divide(counts["adhesion_count"], counts["pred_count"]),
                overseg_rate=divide(counts["overseg_count"], counts["gt_count"]),
                mean_vertices=divide(counts["vertices"], counts["pred_count"]),
                mean_components=divide(counts["components"], counts["pred_count"]))


def measure(gt, predicted, shape, iou_threshold=0.5, boundary_tolerance=2):
    iou, coverage, purity = pairwise_overlap(gt.masks, predicted.masks, gt.classes, predicted.classes)
    # Maximize valid-match cardinality first, then IoU; invalid edges cannot
    # consume a match that would otherwise pass the evaluation threshold.
    valid = iou >= iou_threshold
    weights = valid * (min(iou.shape, default=0) + 1 + iou)
    rows, columns = linear_sum_assignment(-weights)
    matched = [(r, c) for r, c in zip(rows, columns) if valid[r, c]]
    tp = len(matched)
    gt_union = np.logical_or.reduce(gt.masks) if gt.masks else np.zeros(shape, bool)
    pred_union = np.logical_or.reduce(predicted.masks) if predicted.masks else np.zeros(shape, bool)
    occupied = np.zeros(shape, dtype=np.int32)
    for mask in predicted.masks:
        occupied += mask
    boundaries = boundary_counts(gt.masks, predicted.masks, boundary_tolerance)
    counts = {
        "tp": tp, "fp": len(predicted.masks) - tp, "fn": len(gt.masks) - tp,
        "matched_iou_sum": float(sum(iou[r, c] for r, c in matched)),
        "pred_count": len(predicted.masks), "gt_count": len(gt.masks),
        "intersection_pixels": int(np.count_nonzero(gt_union & pred_union)),
        "union_pixels": int(np.count_nonzero(gt_union | pred_union)),
        "adhesion_count": int(np.count_nonzero((coverage >= 0.5).sum(axis=0) >= 2)),
        "overseg_count": int(np.count_nonzero(((purity >= 0.5) & (coverage >= 0.1)).sum(axis=1) >= 2)),
        "vertices": sum(_contour_vertices(mask) for mask in predicted.masks),
        "components": sum(_component_count(mask) for mask in predicted.masks),
        "overlap_pixels": int(np.count_nonzero(occupied > 1)),
        **{"boundary_" + key: value for key, value in boundaries.items()},
    }
    return summary(counts), counts


def select(instance_set, indices):
    indices = list(indices)
    return InstanceSet([instance_set.masks[i] for i in indices], instance_set.classes[indices], np.empty(len(indices)))


def event_outcome(gt, before, after, target_ids, action, shape, iou_threshold, boundary_tolerance):
    old_targets = [p.mask for p in before if p.parcel_id in target_ids]
    new_ids = {p.parcel_id for p in after} - {p.parcel_id for p in before}
    new_targets = [p.mask for p in after if p.parcel_id in set(target_ids) | new_ids]
    affected = old_targets + new_targets
    union = np.logical_or.reduce(affected) if affected else np.zeros(shape, bool)
    gt_indices = [i for i, mask in enumerate(gt.masks) if np.any(mask & union)]
    local_gt = select(gt, gt_indices)
    gt_union = np.logical_or.reduce(local_gt.masks) if local_gt.masks else np.zeros(shape, bool)
    def neighborhood(parcels, targets):
        return instances([p for p in parcels if p.parcel_id in targets or np.any(p.mask & gt_union)])
    old = neighborhood(before, set(target_ids))
    new = neighborhood(after, set(target_ids) | new_ids)
    a, _ = measure(local_gt, old, shape, iou_threshold, boundary_tolerance)
    b, _ = measure(local_gt, new, shape, iou_threshold, boundary_tolerance)
    delta_pq = (b["pq"] or 0.0) - (a["pq"] or 0.0)
    delta_bf1 = b["boundary_f1"] - a["boundary_f1"]
    same = len(old.masks) == len(new.masks) and all(
        np.array_equal(left, right) for left, right in zip(old.masks, new.masks))
    if same:
        outcome = "unchanged"
    elif delta_pq < -0.005 or delta_bf1 < -0.005 or b["overlap_pixels"] > a["overlap_pixels"]:
        outcome = "harmful"
    elif delta_pq > 0.005 or delta_bf1 > 0.005:
        outcome = "beneficial"
    elif (action == "contour_simplify" and b["vertices"] < a["vertices"] and old_targets and new_targets and
          _mask_iou(np.logical_or.reduce(old_targets), np.logical_or.reduce(new_targets)) >= 0.95):
        outcome = "beneficial_geometry"
    else:
        outcome = "neutral"
    return {"outcome": outcome, "delta_pq": delta_pq, "delta_boundary_f1": delta_bf1,
            "vertices_before": a["vertices"], "vertices_after": b["vertices"],
            "affected_gt": len(local_gt.masks)}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(rows, key, repetitions, seed):
    values = np.asarray([row["delta_" + key] for row in rows if row["delta_" + key] is not None])
    if not len(values):
        return {"mean": None, "ci95": None, "images": 0}
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(repetitions)]
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist() if means else None,
            "images": len(values), "unit": "image"}


def evaluate_runs(args):
    if not 0 < args.iou_threshold <= 1 or args.boundary_tolerance < 0 or args.bootstrap < 0:
        raise ValueError("Invalid evaluation thresholds or bootstrap count")
    root = args.dataset_root
    image_dir, label_dir = root / "images" / args.split, root / "labels" / args.split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected images/{args.split} and labels/{args.split} under {root}")
    image_paths = {}
    from predict_fieldbqc_agent import resolve_sources
    for path in resolve_sources(str(image_dir)):
        image_paths[path.stem] = path
    methods = sorted(p for p in args.runs.iterdir() if p.is_dir() and (p / "method_config.json").exists())
    if not methods:
        raise ValueError("No experiment method directories found")
    metadata = [read_json(p / "method_config.json") for p in methods]
    identity = lambda c: (c["image_ids"], c.get("checkpoint_sha256"), c.get("cache_signature"))
    if any(identity(c) != identity(metadata[0]) for c in metadata[1:]):
        raise ValueError("Methods must use identical initial checkpoints and image sets")
    signature = hashlib.sha256("".join(fingerprint(args.cache / "original_masks" / (stem + ".npz"))
                                      for stem in metadata[0]["image_ids"]).encode("ascii")).hexdigest()
    if metadata[0].get("cache_signature") != signature:
        raise ValueError("Evaluation cache differs from the cache used by the methods")
    for method in methods:
        if not (method / "completed.json").exists():
            raise ValueError(f"Incomplete method {method.name}; do not evaluate partial runs")
    image_rows, event_rows, totals, intervals = [], [], [], {}
    for method in methods:
        method_rows, counts_list, trace_list = [], [], []
        groups = {key: [] for key in ("all", "adhesion", "fragmentation", "low_quality", "contour_complexity", "normal")}
        for stem in metadata[0]["image_ids"]:
            before, shape = load_parcels(args.cache / "original_masks" / (stem + ".npz"))
            after, final_shape = load_parcels(method / "masks" / (stem + ".npz"))
            if shape != final_shape or stem not in image_paths:
                raise ValueError(f"Invalid prediction shape or missing dataset image: {stem}")
            label = (label_dir / image_paths[stem].relative_to(image_dir)).with_suffix(".txt")
            if not label.exists() and not args.allow_missing_labels:
                raise FileNotFoundError(f"Missing label: {label}; background images need empty label files")
            gt = read_yolo_masks(label, *shape)
            initial, _ = measure(gt, instances(before), shape, args.iou_threshold, args.boundary_tolerance)
            final, counts = measure(gt, instances(after), shape, args.iou_threshold, args.boundary_tolerance)
            trace = read_json(method / "traces" / (stem + ".json"))
            trace_list.append(trace)
            row = {"method": method.name, "image_id": stem, **final,
                   "delta_pq": final["pq"] - initial["pq"] if final["pq"] is not None and initial["pq"] is not None else None,
                   "delta_boundary_f1": final["boundary_f1"] - initial["boundary_f1"],
                   "initial_pq": initial["pq"], "initial_boundary_f1": initial["boundary_f1"],
                   "vertex_delta": final["vertices"] - initial["vertices"],
                   "postprocess_seconds": trace["postprocess_seconds"], "needs_review": trace["needs_review"]}
            before_ids = {p.parcel_id: p.mask for p in before}
            after_ids = {p.parcel_id: p.mask for p in after}
            row["geometry_changed"] = (before_ids.keys() != after_ids.keys() or any(
                not np.array_equal(mask, after_ids[key]) for key, mask in before_ids.items() if key in after_ids))
            row["harmful_image"] = ((row["delta_pq"] is not None and row["delta_pq"] < -0.005) or
                                     row["delta_boundary_f1"] < -0.005 or
                                     final["overlap_pixels"] > initial["overlap_pixels"])
            method_rows.append(row)
            counts_list.append(counts)
            groups["all"].append(len(counts_list) - 1)
            flags = []
            if initial["adhesion_count"]:
                flags.append("adhesion")
            if initial["overseg_count"]:
                flags.append("fragmentation")
            iou, _, _ = pairwise_overlap(gt.masks, instances(before).masks, gt.classes, instances(before).classes)
            best = iou.max(axis=0) if len(gt.masks) else np.zeros(len(before))
            if np.any(best < args.iou_threshold):
                flags.append("low_quality")
            contour_threshold = metadata[0]["agent_config"]["max_contour_vertices"]
            if any(_contour_vertices(p.mask) > contour_threshold for p in before):
                flags.append("contour_complexity")
            if not flags and initial["fp"] == 0 and initial["fn"] == 0:
                flags.append("normal")
            for flag in flags:
                groups[flag].append(len(counts_list) - 1)
            for record in sorted((method / "events" / stem).glob("*.json")):
                event = read_json(record)
                old, _ = load_parcels(record.with_name(record.stem + "_before.npz"))
                candidate, _ = load_parcels(record.with_name(record.stem + "_candidate.npz"))
                outcome = event_outcome(gt, old, candidate, event["target_ids"], event["action"],
                                        shape, args.iou_threshold, args.boundary_tolerance)
                event_rows.append({"method": method.name, "image_id": stem, "iteration": event["iteration"],
                                   "action": event["action"], "accepted": event["accepted"], **outcome})
        image_rows.extend(method_rows)
        committed = [e for e in event_rows if e["method"] == method.name and e["accepted"]]
        attempts = [t for trace in trace_list for t in trace["action_trace"] if "target_ids" in t and t["action"] != "plan"]
        plans = [t for trace in trace_list for t in trace["action_trace"] if t["action"] == "plan"]
        times = [t["postprocess_seconds"] for t in trace_list]
        for group, indices in groups.items():
            if not indices:
                continue
            summed = {key: sum(counts_list[i][key] for i in indices) for key in counts_list[0]}
            totals.append({"method": method.name, "subset": group, "images": len(indices), **summary(summed)})
            totals[-1].update(changed_image_rate=float(np.mean([method_rows[i]["geometry_changed"] for i in indices])),
                              harmful_image_rate=float(np.mean([method_rows[i]["harmful_image"] for i in indices])))
        all_row = next(row for row in totals if row["method"] == method.name and row["subset"] == "all")
        all_row.update(
            committed_events=len(committed), tool_attempts=len(attempts),
            success_rate=divide(sum(e["outcome"].startswith("beneficial") for e in committed), len(committed)),
            harmful_rate=divide(sum(e["outcome"] == "harmful" for e in committed), len(committed)),
            rollback_rate=divide(sum(bool(t.get("rolled_back")) for t in attempts), len(attempts)),
            review_image_rate=float(np.mean([t["needs_review"] for t in trace_list])),
            postprocess_mean_seconds=float(np.mean(times)), postprocess_p95_seconds=float(np.percentile(times, 95)),
            planner_calls=len(plans), fallback_rate=divide(sum("fallback" in t for t in plans), len(plans)),
            cache_hits=sum(bool(t.get("cache_hit")) for t in plans),
            billed_input_tokens=sum(t.get("billed_usage", {}).get("input_tokens",
                                    t.get("billed_usage", {}).get("prompt_tokens", 0)) for t in plans),
            billed_output_tokens=sum(t.get("billed_usage", {}).get("output_tokens",
                                     t.get("billed_usage", {}).get("completion_tokens", 0)) for t in plans))
        intervals[method.name] = {key: paired_interval(method_rows, key, args.bootstrap, args.seed)
                                  for key in ("pq", "boundary_f1")}
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "summary.csv", totals)
    write_csv(args.output / "per_image.csv", image_rows)
    write_csv(args.output / "events.csv", event_rows)
    write_json(args.output / "paired_intervals.json", intervals)
    write_json(args.output / "evaluation_config.json", {
        "dataset_root": str(root), "split": args.split, "iou_threshold": args.iou_threshold,
        "boundary_tolerance": args.boundary_tolerance, "adhesion_gt_coverage": 0.5,
        "fragment_pred_purity": 0.5, "fragment_min_gt_coverage": 0.1, "event_delta_threshold": 0.005,
        "bootstrap_repetitions": args.bootstrap, "seed": args.seed, "subset_unit": "image",
        "success_denominator": "committed candidate events", "mAP_computed": False,
        "warning": "Image bootstrap assumes independent images; use scene bootstrap for adjacent tiles."})
    lines = ["# Agent Experiment Results", "", "| Method | PQ | Instance F1 | Boundary F1 | Harmful rate | Seconds/image |",
             "|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "NA" if value is None else f"{value:.4f}"
    for row in totals:
        if row["subset"] == "all":
            lines.append("| " + row["method"] + " | " + " | ".join(fmt(row[key]) for key in
                         ("pq", "instance_f1", "boundary_f1", "harmful_rate", "postprocess_mean_seconds")) + " |")
    lines.extend(["", "PQ is a class-aware instance metric at the configured IoU threshold.",
                  "Boundary F1 is the micro instance-boundary-union score, including internal parcel boundaries.",
                  "Success/harmful rates use committed candidate events; rejected candidates are in events.csv.",
                  "Subsets are defined from initial predictions and GT, and may overlap. They contain entire images.",
                  "Timing includes diagnosis, planning, tools and event recording; excludes shared initial prediction.",
                  "Cached planner results are replay timing, not live API latency. Report cache hits separately.",
                  "mAP is deliberately omitted because geometry edits do not produce fresh ranking confidences.",
                  "Review flags describe unresolved evidence, not measured segmentation errors."])
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved evaluation to {args.output.resolve()}")
