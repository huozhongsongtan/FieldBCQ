"""Local adaptive minimum-structure agent for parcel boundary post-processing.

This is a reproducible no-network controller. It observes the current parcel,
classifies its boundary structure, tries several local simplifiers plus a few
lower-point direct probes, applies the shared typed verifier, and commits the
lowest-count accepted candidate. It is intended as an agent-policy control,
not as a replacement for the external GPT run.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from boundary_executor import BoundaryConfig, BoundaryExecutor
from boundary_tools import REGISTRY
from cvr_agent import dump, read_items, save_items, sha, scene_metrics, vertices
from evaluate_experiments import get_metrics, preview
from evaluate_goal_repair import frozen_ownership
from evaluate_mechanism_metrics import read_yolo_masks
from fieldbqc_geometry import polygons
from run_boundary_tools import load_image


def parcel_profile(box, item, items):
    g = item.geometry
    long = 0.0
    total = 0.0
    for part in polygons(g):
        p = np.asarray(part.exterior.simplify(1.0).coords)
        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        total += float(lengths.sum())
        long += float(lengths[lengths >= 12.0].sum())
    shared = sum(
        g.boundary.intersection(other.geometry.boundary).length
        for other in items if other.identifier != item.identifier
    )
    return {
        "vertices": vertices(g),
        "perimeter": float(g.length),
        "long_straight_fraction": long / max(total, 1e-9),
        "shared_length": float(shared),
        "holes": sum(len(p.interiors) for p in polygons(g)),
        "components": len(polygons(g)),
        "image_support": float(box.evidence(g)),
    }


def candidate_metrics(box, old, new):
    a, b = old.geometry, new.geometry
    return {
        "vertices": vertices(b),
        "reduction": 1.0 - vertices(b) / max(vertices(a), 1),
        "shift": float(a.boundary.hausdorff_distance(b.boundary)),
        "area_change": abs(b.area - a.area) / max(a.area, 1.0),
        "support_gain": float(box.evidence(b) - box.evidence(a)),
    }


def specs(profile):
    """Select candidate families from observable geometry, not fixed ranking."""
    n = profile["vertices"]
    shared = profile["shared_length"] > 1e-7
    result = []
    # Always expose both ordinary and shared-edge simplifiers. A boundary can
    # be shared only after a previous target has been committed, and relying on
    # a precomputed boolean would hide that scene-level state from the policy.
    result += [("contour_dp", {"tolerance": t}) for t in (0.5, 1.0, 2.0)]
    result += [("contour_shared_dp", {"tolerance": t}) for t in (0.5, 1.0, 2.0)]
    # Long, nearly linear parcels benefit from explicit collinearity/chord tests.
    if profile["long_straight_fraction"] >= 0.35 or n >= 80:
        result += [("contour_collinear", {"tolerance": t, "angle": 20.0})
                   for t in (0.5, 1.0, 2.0)]
        result += [("contour_corridor", {"tolerance": t}) for t in (1.0, 2.0)]
    # Curved/complex parcels retain a lower-tolerance alternative.
    if profile["long_straight_fraction"] < 0.35 and n < 80:
        result += [("contour_vw", {"tolerance": 1.0})]
    return result


def evaluate_target(executor, uid, prof, direct_probe_tolerances=(3.0, 4.0, 6.0)):
    old = next(p for p in executor.current if p.identifier == uid)
    attempts = []
    accepted = []
    for tool, params in specs(prof):
        try:
            candidate, info = executor.box.propose(tool, executor.current, [uid], params)
            verdict = executor.verify(candidate, [uid], REGISTRY[tool].task)
            target = next(p for p in candidate if p.identifier == uid)
            metrics = candidate_metrics(executor.box, old, target)
            row = {"kind": "tool", "tool": tool, "parameters": params,
                   "accept": verdict["accept"], "reasons": verdict["reasons"],
                   "metrics": metrics}
            attempts.append(row)
            if verdict["accept"] and metrics["vertices"] < prof["vertices"]:
                accepted.append((metrics["vertices"], metrics["shift"],
                                 metrics["area_change"], -metrics["support_gain"],
                                 row, candidate))
        except Exception as error:
            attempts.append({"kind": "tool", "tool": tool, "parameters": params,
                             "accept": False, "reasons": ["tool_error"],
                             "error": f"{type(error).__name__}: {error}"})

    # Direct structural probes are the local analogue of the agent's own
    # geometry proposal. They are deliberately allowed to be more aggressive;
    # the same verifier decides whether they can be committed.
    if prof["vertices"] >= 80:
        for tolerance in direct_probe_tolerances:
            try:
                g = old.geometry.simplify(tolerance, preserve_topology=True)
                candidate = [p.with_geometry(g) if p.identifier == uid else p
                             for p in executor.current]
                verdict = executor.verify(candidate, [uid], "simplify")
                target = next(p for p in candidate if p.identifier == uid)
                metrics = candidate_metrics(executor.box, old, target)
                row = {"kind": "direct_structure_probe", "tolerance": tolerance,
                       "accept": verdict["accept"], "reasons": verdict["reasons"],
                       "metrics": metrics}
                attempts.append(row)
                if verdict["accept"] and metrics["vertices"] < prof["vertices"]:
                    accepted.append((metrics["vertices"], metrics["shift"],
                                     metrics["area_change"], -metrics["support_gain"],
                                     row, candidate))
            except Exception as error:
                attempts.append({"kind": "direct_structure_probe", "tolerance": tolerance,
                                 "accept": False, "reasons": ["direct_probe_error"],
                                 "error": f"{type(error).__name__}: {error}"})
    if not accepted:
        return None, attempts
    accepted.sort(key=lambda x: x[:4])
    return accepted[0], attempts


def load_gt(root, scene, image_path, raw):
    pairs = json.loads((root / "paired_inputs.json").read_text(encoding="utf-8"))["pairs"]
    pair = next(p for p in pairs if p["image_id"] == scene)
    image = Image.open(image_path).convert("RGB")
    shape = image.size[::-1]
    label = root / pair["ground_truth_original_label"]
    gt = read_yolo_masks(label, *shape)
    return gt, shape, frozen_ownership(raw, gt, shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scene", default="FBIS_SK_622595_PL_3m_512")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("strict", "low_vertex"), default="strict",
                        help="strict preserves pixel boundary quality; low_vertex is an explicit generalized/LOD mode")
    parser.add_argument("--max-shift", type=float, default=None,
                        help="override the mode's cumulative boundary-shift gate")
    parser.add_argument("--min-iou", type=float, default=None,
                        help="override the mode's raster-IoU gate")
    parser.add_argument("--max-area-change", type=float, default=None,
                        help="override the mode's area-change gate")
    parser.add_argument("--no-direct", action="store_true",
                        help="disable direct structural proposals; tool-selection ablation")
    args = parser.parse_args()
    root = args.data.resolve()
    manifest = json.loads((root / "postprocess_inputs.json").read_text(encoding="utf-8"))
    entry = next(row for row in manifest["inputs"] if Path(row["image"]).stem == args.scene)
    image_path = root / entry["image"]
    vector_path = root / entry["prediction_vector"]
    image = load_image(image_path)
    raw = read_items(vector_path)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.mode == "low_vertex":
        # This is a separate cartographic/LOD operating point. It makes the
        # trade-off explicit instead of silently loosening the precision run.
        config = BoundaryConfig(budget=768, seconds=900., simplify_max_shift=6.0,
                                simplify_max_area_change=.10, simplify_min_iou=.90,
                                simplify_edge_drop=.10, realign_max_shift=8.0,
                                realign_max_area_change=.15)
        direct_tolerances = (3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
    else:
        config = BoundaryConfig(budget=512, seconds=900.)
        direct_tolerances = (3.0, 4.0, 6.0)
    overrides = {}
    if args.max_shift is not None:
        overrides["simplify_max_shift"] = args.max_shift
        overrides["realign_max_shift"] = max(config.realign_max_shift, args.max_shift)
    if args.min_iou is not None:
        overrides["simplify_min_iou"] = args.min_iou
    if args.max_area_change is not None:
        overrides["simplify_max_area_change"] = args.max_area_change
        overrides["realign_max_area_change"] = max(config.realign_max_area_change, args.max_area_change)
    if overrides:
        config = replace(config, **overrides)
    executor = BoundaryExecutor(image, raw, out / "run", config)
    decisions = []
    agent_calls = []
    for item in list(raw):
        uid = item.identifier
        current = next(p for p in executor.current if p.identifier == uid)
        prof = parcel_profile(executor.box, current, executor.current)
        best, attempts = evaluate_target(executor, uid, prof,
                                         () if args.no_direct else direct_tolerances)
        for attempt in attempts:
            agent_calls.append({"target_id": uid, "profile": prof, "attempt": attempt,
                                "selected": bool(best is not None and attempt is best[4])})
        committed = False
        if best is not None:
            row = best[4]
            if row["kind"] == "tool":
                result = executor.attempt(row["tool"], [uid], row["parameters"],
                                          expected_state=executor.state_id,
                                          source="adaptive_structure_agent")
                committed = bool(result["committed"])
            else:
                # A direct proposal is committed literally only after the
                # same verifier has accepted it; no hidden finishing step.
                candidate = best[5]
                tid = f"direct-{len(executor.history):05d}"
                folder = out / "run" / "direct_proposals" / tid
                before_scene = scene_metrics(executor.current)
                candidate_file = save_items(folder / "candidate.pixel.geojson", candidate)
                executor.current = list(candidate)
                executor.history.append({"trial_id": tid, "source": "adaptive_structure_agent",
                    "tool": "direct_structural_simplification", "target_ids": [uid],
                    "parameters": {"tolerance": row["tolerance"]}, "committed": True,
                    "verdict": {"accept": True, "reasons": [], "metrics": row["metrics"]},
                    "candidate_file": candidate_file, "before": before_scene,
                    "after": scene_metrics(executor.current), "output_state": executor.state_id})
                committed = True
        after = next(p for p in executor.current if p.identifier == uid)
        decisions.append({"target_id": uid, "profile": prof,
                          "attempts": attempts, "committed": committed,
                          "vertices_before": prof["vertices"],
                          "vertices_after": vertices(after.geometry),
                          "selected": best[4] if best else None})
    trace = executor.finish("all_targets_decided")
    dump(out / "decisions.json", decisions)
    with (out / "agent_calls.jsonl").open("w", encoding="utf-8") as f:
        for call in agent_calls:
            f.write(json.dumps(call, ensure_ascii=False) + "\n")
    dump(out / "manifest.json", {"scene": args.scene, "method": "adaptive_structure_agent",
        "mode": args.mode, "goal": "adaptive minimum structural vertices", "runtime_gt_access": False,
        "input_sha256": {str(vector_path): sha(vector_path), str(image_path): sha(image_path)},
        "quality_gate": {"max_shift": config.simplify_max_shift, "min_raster_iou": config.simplify_min_iou,
                          "max_area_change": config.simplify_max_area_change},
        "direct_probe_policy": f"tolerance {list(direct_tolerances)} when current target has >=80 vertices; every proposal is verifier-gated"})
    gt, shape, frozen = load_gt(root, args.scene, image_path, raw)
    final = read_items(out / "run" / "final" / "prediction.repaired.pixel.geojson")
    metrics = get_metrics(final, gt, shape, raw, frozen)
    summary = {"scene": args.scene, "method": "adaptive_structure_agent", "mode": args.mode,
               "direct_enabled": not args.no_direct,
               "vertices_before": sum(vertices(p.geometry) for p in raw),
               "vertices_after": sum(vertices(p.geometry) for p in final),
               "reduction": 1 - sum(vertices(p.geometry) for p in final) / max(1, sum(vertices(p.geometry) for p in raw)),
               **{k: metrics[k] for k in ("boundary_f1", "pixel_boundary_chamfer", "instance_f1_mean", "overlap_excess_area", "invalid_geometries")},
               "commits": sum(d["committed"] for d in decisions),
               "direct_probe_attempts": sum(sum(a["kind"] == "direct_structure_probe" for a in d["attempts"]) for d in decisions),
               "direct_probe_accepts": sum(sum(a["kind"] == "direct_structure_probe" and a["accept"] for a in d["attempts"]) for d in decisions)}
    dump(out / "summary.json", summary)
    with (out / "per_parcel.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["id", "before", "after", "committed"])
        for d in decisions: writer.writerow([d["target_id"], d["vertices_before"], d["vertices_after"], d["committed"]])
    source = Image.open(image_path).convert("RGB")
    panels = [preview(source, raw, "raw"), preview(source, final, "adaptive structure agent")]
    canvas = Image.new("RGB", (300 * len(panels), 345), "white")
    for i, panel in enumerate(panels): canvas.paste(panel, (300 * i, 0))
    canvas.save(out / "comparison.png")
    mode_text = "严格精度模式" if args.mode == "strict" else "低点数概化模式（单独报告几何误差）"
    lines = ["# 自适应结构点智能体实验", "", f"运行模式：{mode_text}。该实验为本地无网络控制器：根据当前地块复杂度、公共边和长直边比例选择候选；同时尝试更激进的直接结构化简化，并由同一几何门控决定是否提交。", "", "|指标|结果|", "|---|---:|", f"|原始顶点|{summary['vertices_before']}|", f"|处理后顶点|{summary['vertices_after']}|", f"|减点率|{summary['reduction']:.2%}|", f"|BF1|{summary['boundary_f1']:.5f}|", f"|平均边界距离|{summary['pixel_boundary_chamfer']:.4f}|", f"|实例F1|{summary['instance_f1_mean']:.5f}|", f"|重叠超额面积|{summary['overlap_excess_area']:.2f}|", f"|非法几何|{summary['invalid_geometries']}|", f"|直接结构探针|{summary['direct_probe_attempts']}（通过{summary['direct_probe_accepts']}）|", "", "![comparison](comparison.png)", "", "每个候选都经过共同质量门控；被拒的低点数直接方案保留在 `decisions.json`，完整候选调用逐行记录在 `agent_calls.jsonl`，不会冒充最终结果。"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
