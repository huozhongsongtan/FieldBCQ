"""Frozen test-set experiment: prediction, GT-blind repair, offline evaluation.

Local adaptive search is an algorithmic controller, NOT an external GPT run.
All stages resume only the same frozen configuration. Errors remain explicit.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_CONFIG_DIR", str(__import__("pathlib").Path(__file__).resolve().parents[2] / ".fieldbqc-config"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image

from boundary_executor import BoundaryConfig, BoundaryExecutor
from boundary_tools import REGISTRY
from cvr_agent import HOST, dump, read_items, save_items, sha, scene_metrics, vertices
from fieldbqc_geometry import from_parcels, polygons, geometry_mask, to_parcels
from run_adaptive_structure_agent import parcel_profile, evaluate_target
from run_boundary_tools import load_image

VERSION = "testset-boundary-v1"
METHODS = ("public_edge", "local_adaptive_strict", "local_adaptive_balanced")
CONFIGS = {
    "public_edge": BoundaryConfig(budget=10000, seconds=3600.),
    "local_adaptive_strict": BoundaryConfig(budget=10000, seconds=3600.),
    "local_adaptive_balanced": BoundaryConfig(budget=10000, seconds=3600.,
        simplify_max_shift=4., simplify_max_area_change=.06, simplify_min_iou=.95,
        simplify_edge_drop=.10, realign_max_shift=8., realign_max_area_change=.15),
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dump(temporary, value)
    os.replace(temporary, path)


def domain(stem):
    if stem.startswith("DDFHAPD_"):
        return "DDFHAPD"
    if stem.startswith("FBIS_"):
        return "FBIS"
    if stem.startswith("FTW_"):
        return "FTW"
    return stem.split("_")[0]


def prepare(args):
    root, data = args.output.resolve(), args.dataset.resolve()
    source = data / "images" / "test"
    images = sorted([p for p in source.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}])
    if not images or len({p.stem.casefold() for p in images}) != len(images):
        raise ValueError("Missing images or duplicate stems")
    entries = []
    for image in images:
        label = data / "labels" / "test" / (image.stem + ".txt")
        if not label.is_file():
            raise FileNotFoundError(label)
        entries.append({"scene": image.stem, "domain": domain(image.stem),
            "image": str(image), "image_sha256": sha(image),
            "label": str(label), "label_sha256": sha(label)})
    code = {str(p.resolve()): sha(p) for p in Path(__file__).parent.glob("*.py")}
    for p in HOST.glob("*.py"):
        if p.name.startswith(("fieldbqc_", "evaluate_", "run_agent_", "run_geometry_", "predict_fieldbqc_")):
            code[str(p.resolve())] = sha(p)
    frozen = {"version": VERSION, "dataset": str(data), "split": "test", "weights": str(args.weights.resolve()),
        "weights_sha256": sha(args.weights), "inference": {"imgsz": 640, "conf": .25, "iou": .7,
            "max_det": 710, "retina_masks": True, "device": "cpu", "torch_threads": args.threads},
        "methods": {k: asdict(v) for k, v in CONFIGS.items()},
        "entries": entries, "source_sha256": code,
        "external_GPT_in_local_methods": False,
        "runtime_GT_access": False,
        "evaluation_note": "Labels are opened only after all repair arms of a scene have been frozen. Empty-label scenes retained. No test-quality threshold chooses deployed outputs.",
        "independence_note": "Original train/val/test split reused; parent-scene independence and previous use of this split for visual-model selection require separate disclosure."}
    root.mkdir(parents=True, exist_ok=True)
    mf = root / "manifest.json"
    if mf.exists():
        if load(mf) != frozen:
            raise ValueError("Frozen inputs/settings/source changed; choose a new output")
    else:
        for path in code:
            p = Path(path)
            target = root / "source_snapshot" / ("cvr" if p.parent == Path(__file__).parent else "host") / p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        atomic_json(mf, frozen)
    # GT-free manifest for the existing autonomous GPT runner.
    atomic_json(root / "postprocess_inputs.json", {"inputs": [
        {"image": e["image"], "prediction_vector": str(root / "predictions" / e["scene"] / "prediction.pixel.geojson")}
        for e in entries], "source": "Frozen test predictions; no GT paths"})
    return frozen


def predict(args, manifest):
    import torch
    torch.set_num_threads(args.threads)
    from ultralytics import YOLO
    from fieldbqc_agent import parcels_from_ultralytics
    from predict_fieldbqc_agent import save_parcels
    model = None
    root = args.output.resolve()
    entries = manifest["entries"][:args.limit or None]
    for i, entry in enumerate(entries, 1):
        folder = root / "predictions" / entry["scene"]
        done = folder / "completed.json"
        if done.exists():
            recorded = load(done)
            if recorded["image_sha256"] != entry["image_sha256"] or sha(folder / "prediction.pixel.geojson") != recorded["vector_sha256"]:
                raise ValueError("Changed prediction cache")
            continue
        if model is None:
            model = YOLO(manifest["weights"], task="segment")
        begin = time.monotonic()
        result = model.predict(source=entry["image"], device="cpu", imgsz=640,
            conf=.25, iou=.7, max_det=710, retina_masks=True, save=False, verbose=False)[0]
        parcels = parcels_from_ultralytics(result)
        items = from_parcels(parcels)
        folder.mkdir(parents=True, exist_ok=True)
        save_parcels(folder / "prediction.original_masks.npz", parcels, result.orig_shape)
        vector = save_items(folder / "prediction.pixel.geojson", items)
        # Conversion is checked against every original instance mask, including holes.
        if not all(np.array_equal(p.mask, geometry_mask(g.geometry, result.orig_shape)) for p, g in zip(parcels, items)):
            raise ValueError("Raw vector/mask round trip failed")
        atomic_json(done, {"scene": entry["scene"], "image_sha256": entry["image_sha256"],
            "vector_sha256": vector["sha256"], "shape": list(result.orig_shape),
            "instances": len(items), "vertices": sum(vertices(p.geometry) for p in items),
            "inference_seconds": time.monotonic() - begin, "mask_roundtrip": True})
        atomic_json(root / "prediction_progress.json", {"completed": sum((root / "predictions" / e["scene"] / "completed.json").exists() for e in manifest["entries"]),
            "expected": len(manifest["entries"]), "last_scene": entry["scene"], "updated_utc": datetime.now(timezone.utc).isoformat()})
        print(f"PREDICT {i}/{len(entries)} {entry['scene']} instances={len(items)} seconds={time.monotonic()-begin:.2f}", flush=True)
    if len(entries) == len(manifest["entries"]):
        atomic_json(root / "prediction_complete.json", {"images": len(entries)})


def run_local_method(image, raw, folder, method):
    """Same candidate policy and verifier, compact target-only audit artifacts."""
    begin = time.monotonic()
    folder.mkdir(parents=True, exist_ok=False)
    if not raw:
        save_items(folder / "final.pixel.geojson", [])
        atomic_json(folder / "trace.json", {"model_calls": 0, "commits": 0, "candidates": 0,
            "empty_prediction": True, "runtime_gt_access": False})
        return []
    executor = BoundaryExecutor(image, raw, folder / "executor", CONFIGS[method])
    decisions, candidate_count = [], 0
    with (folder / "candidate_calls.jsonl").open("w", encoding="utf-8") as log:
        for item in raw:
            uid = item.identifier
            current = next(p for p in executor.current if p.identifier == uid)
            prof = parcel_profile(executor.box, current, executor.current)
            state = executor.state_id
            if method == "public_edge":
                params = {"tolerance": 2.}
                try:
                    candidate, _ = executor.box.propose("contour_shared_dp", executor.current, [uid], params)
                    verdict = executor.verify(candidate, [uid], "simplify")
                    attempts = [{"kind": "tool", "tool": "contour_shared_dp", "parameters": params,
                        "accept": verdict["accept"], "reasons": verdict["reasons"]}]
                    selected = (attempts[0], candidate) if verdict["accept"] else None
                except Exception as error:
                    attempts = [{"kind": "tool", "tool": "contour_shared_dp", "parameters": params,
                        "accept": False, "reasons": ["tool_error"], "error": f"{type(error).__name__}: {error}"}]
                    selected = None
            else:
                tolerances = (3., 4., 6.) if method.endswith("strict") else (3., 4., 6., 8., 10., 12., 16., 20.)
                best, attempts = evaluate_target(executor, uid, prof, tolerances)
                selected = (best[4], best[5]) if best else None
            candidate_count += len(attempts)
            for attempt in attempts:
                log.write(json.dumps({"target_id": uid, "input_state": state, "profile": prof,
                    "attempt": attempt, "selected": bool(selected and attempt is selected[0]),
                    "controller": "deterministic_shared_edge" if method == "public_edge" else "local_adaptive_search_not_GPT"}, ensure_ascii=False) + "\n")
            committed = False
            if selected:
                row, candidate = selected
                # A common final check applies to tool and direct local candidates alike.
                verdict = executor.verify(candidate, [uid], "simplify")
                if verdict["accept"]:
                    target = next(p for p in candidate if p.identifier == uid)
                    ref = save_items(folder / "commits" / f"target-{uid}.geojson", [target])
                    with executor.audit.call("commit_verified_local_candidate", {"target_id": uid,
                            "selected": row, "input_state": state}, ["common_typed_verification", "atomic_target_replacement"]) as event:
                        executor.current = list(candidate)
                        event.update(candidate=ref, output_state=executor.state_id)
                    committed = True
            after = next(p for p in executor.current if p.identifier == uid)
            decisions.append({"target_id": uid, "profile": prof, "selected": selected[0] if selected else None,
                "committed": committed, "vertices_before": vertices(current.geometry), "vertices_after": vertices(after.geometry)})
    final = list(executor.current)
    save_items(folder / "final.pixel.geojson", final)
    atomic_json(folder / "decisions.json", decisions)
    atomic_json(folder / "trace.json", {"config": asdict(CONFIGS[method]), "model_calls": 0,
        "controller": "local_algorithmic_controller_not_GPT", "candidates": candidate_count,
        "commits": sum(d["committed"] for d in decisions), "runtime_gt_access": False,
        "baseline": scene_metrics(raw), "final": scene_metrics(final),
        "seconds": time.monotonic() - begin,
        "log_scope": "Every generated tool/local simplification candidate plus common verifier verdict. Local direct probes are shapely.simplify, not model-authored geometry."})
    return final


def offline_evaluate(items, raw, label, shape):
    """Only this stage reads labels; undefined distances remain null."""
    from evaluate_agent_experiments import instances, measure
    from evaluate_mechanism_metrics import read_yolo_masks, instance_boundary_union
    from scipy.ndimage import distance_transform_edt
    gt = read_yolo_masks(Path(label), *shape)
    predicted = instances(to_parcels(items, shape))
    results, counts = {}, {}
    for threshold in (.5, .75, .9):
        measured, ct = measure(gt, predicted, shape, iou_threshold=threshold)
        counts[str(threshold)] = ct
        results[f"instance_f1_{threshold}"] = measured["instance_f1"]
        if threshold == .5:
            results.update(boundary_f1=measured["boundary_f1"], union_iou=measured["union_iou"],
                pq=measured["pq"], gt_instances=len(gt.masks), predicted_instances=len(items),
                empty_gt=len(gt.masks) == 0, empty_prediction=len(items) == 0)
    valid = [v for k, v in results.items() if k.startswith("instance_f1_") and v is not None]
    results["instance_f1_mean"] = float(np.mean(valid)) if valid else None
    pb, gb = instance_boundary_union(predicted.masks), instance_boundary_union(gt.masks)
    if pb.size == 0:
        pb = np.zeros(shape, bool)
    if gb.size == 0:
        gb = np.zeros(shape, bool)
    if pb.any() and gb.any():
        pg, gp = distance_transform_edt(~gb)[pb], distance_transform_edt(~pb)[gb]
        results.update(pixel_boundary_chamfer=float((pg.mean() + gp.mean()) / 2),
            pixel_boundary_p95=float(np.percentile(np.r_[pg, gp], 95)))
    else:
        results.update(pixel_boundary_chamfer=None, pixel_boundary_p95=None)
    before, after = {p.identifier: p for p in raw}, {p.identifier: p for p in items}
    results.update(scene_metrics(items), id_preservation=list(before) == list(after),
        max_boundary_shift=max((p.geometry.boundary.hausdorff_distance(after[uid].geometry.boundary) for uid, p in before.items()), default=0.),
        low_4_8_count=sum(4 <= vertices(p.geometry) <= 8 for p in items),
        low_3_10_count=sum(3 <= vertices(p.geometry) <= 10 for p in items))
    return {"metrics": results, "counts": counts}


def process_scene(root_s, scene, image_path, label_path, image_hash, label_hash):
    import cv2
    cv2.setNumThreads(1)
    root = Path(root_s)
    scene_out = root / "scenes" / scene
    done = scene_out / "completed.json"
    if done.exists():
        return scene
    raw = read_items(root / "predictions" / scene / "prediction.pixel.geojson")
    image = load_image(image_path)
    finals = {"raw": raw}
    for method in METHODS:
        folder = scene_out / method
        if (folder / "trace.json").exists() and (folder / "final.pixel.geojson").exists():
            finals[method] = read_items(folder / "final.pixel.geojson")
        else:
            # Preserve interrupted attempts rather than mixing their logs on resume.
            if folder.exists():
                folder.rename(folder.with_name(method + f".interrupted-{time.time_ns()}"))
            finals[method] = run_local_method(image, raw, folder, method)
    # Repair arms are all persisted before accessing reference labels.
    if sha(image_path) != image_hash or sha(label_path) != label_hash:
        raise ValueError("Frozen source changed")
    evaluated = {method: offline_evaluate(items, raw, label_path, image.shape[:2]) for method, items in finals.items()}
    with (scene_out / "per_parcel.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["instance_id", *finals])
        for index, p in enumerate(raw):
            w.writerow([p.identifier, *[vertices(items[index].geometry) for items in finals.values()]])
    atomic_json(done, {"scene": scene, "domain": domain(scene), "evaluation": evaluated,
        "input_hashes": {"image": image_hash, "label_offline_only": label_hash},
        "external_model_calls": 0, "runtime_gt_access": False})
    return scene


def repair(args, manifest):
    root = args.output.resolve()
    entries = manifest["entries"][:args.limit or None]
    pending, failures = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        unseen = {e["scene"]: e for e in entries if not (root / "scenes" / e["scene"] / "completed.json").exists()}
        while unseen or pending:
            for scene, entry in list(unseen.items()):
                if len(pending) >= args.workers * 2:
                    break
                if not (root / "predictions" / scene / "completed.json").exists():
                    continue
                future = pool.submit(process_scene, str(root), scene, entry["image"], entry["label"], entry["image_sha256"], entry["label_sha256"])
                pending[future] = scene
                del unseen[scene]
            if pending:
                ready, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in ready:
                    scene = pending.pop(future)
                    try:
                        future.result()
                        print(f"REPAIRED {scene}", flush=True)
                    except Exception:
                        failure = {"scene": scene, "error": traceback.format_exc()}
                        failures.append(failure)
                        atomic_json(root / "errors" / (scene + ".json"), failure)
                        print(f"FAILED {scene} {failure['error']}", flush=True)
                completed = sum((root / "scenes" / e["scene"] / "completed.json").exists() for e in entries)
                atomic_json(root / "repair_progress.json", {"completed": completed, "expected": len(entries),
                    "running": list(pending.values()), "errors": len(failures), "updated_utc": datetime.now(timezone.utc).isoformat()})
            elif unseen:
                if not args.watch:
                    print(f"{len(unseen)} scenes waiting for prediction; use --watch", flush=True)
                    break
                time.sleep(2)
    report(args, manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} scenes failed; see errors/")


def mean(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def report(args, manifest):
    from evaluate_agent_experiments import summary
    root = args.output.resolve()
    completed = [load(root / "scenes" / e["scene"] / "completed.json") for e in manifest["entries"]
        if (root / "scenes" / e["scene"] / "completed.json").exists()]
    if not completed:
        return
    rows, aggregate = [], {}
    for scene in completed:
        for method, data in scene["evaluation"].items():
            raw = scene["evaluation"]["raw"]["metrics"]
            m = data["metrics"]
            checks = [m["boundary_f1"] >= raw["boundary_f1"] - .005,
                m["invalid_geometries"] == 0, m["id_preservation"],
                m["overlap_excess_area"] <= raw["overlap_excess_area"] + 1e-7]
            if raw["instance_f1_mean"] is not None:
                checks.append(m["instance_f1_mean"] is not None and m["instance_f1_mean"] >= raw["instance_f1_mean"] - .01)
            if raw["pixel_boundary_chamfer"] is not None:
                checks.append(m["pixel_boundary_chamfer"] is not None and m["pixel_boundary_chamfer"] <= raw["pixel_boundary_chamfer"] + .25)
            rows.append({"scene": scene["scene"], "domain": scene["domain"], "method": method, **m,
                "vertices_before": raw["vertices"], "quality_pass": all(checks)})
    with (root / "per_scene.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for group in ["all", *sorted({s["domain"] for s in completed})]:
        for method in ["raw", *METHODS]:
            subset = [s for s in completed if group == "all" or s["domain"] == group]
            chosen = [r for r in rows if r["method"] == method and (group == "all" or r["domain"] == group)]
            counts = {}
            for threshold in ("0.5", "0.75", "0.9"):
                total = {k: sum(s["evaluation"][method]["counts"][threshold][k] for s in subset)
                    for k in subset[0]["evaluation"][method]["counts"][threshold]}
                counts[threshold] = summary(total)
            n0, n1 = sum(r["vertices_before"] for r in chosen), sum(r["vertices"] for r in chosen)
            predn = sum(r["predicted_instances"] for r in chosen)
            aggregate[f"{group}/{method}"] = {"scenes": len(chosen), "vertices_before": n0,
                "vertices_after": n1, "reduction": 1 - n1 / max(n0, 1),
                "predicted_instances": predn, "mean_vertices": n1 / max(predn, 1),
                "boundary_f1_macro": mean([r["boundary_f1"] for r in chosen]),
                "boundary_f1_micro": counts["0.5"]["boundary_f1"],
                "instance_f1_macro": mean([r["instance_f1_mean"] for r in chosen]),
                "instance_f1_micro": mean([counts[t]["instance_f1"] for t in counts]),
                "chamfer_macro": mean([r["pixel_boundary_chamfer"] for r in chosen]),
                "chamfer_defined_scenes": sum(r["pixel_boundary_chamfer"] is not None for r in chosen),
                "overlap_area": sum(r["overlap_excess_area"] for r in chosen),
                "invalid_geometries": sum(r["invalid_geometries"] for r in chosen),
                "quality_pass_scenes": sum(r["quality_pass"] for r in chosen),
                "low_4_8_ratio": sum(r["low_4_8_count"] for r in chosen) / max(predn, 1),
                "empty_gt_scenes": sum(r["empty_gt"] for r in chosen),
                "false_predictions_on_empty_gt": sum(r["predicted_instances"] for r in chosen if r["empty_gt"])}
    complete = len(completed) == len(manifest["entries"])
    atomic_json(root / "summary.json", {"complete": complete, "expected": len(manifest["entries"]),
        "completed": len(completed), "aggregate": aggregate, "external_GPT_calls_in_these_results": 0})
    lines = ["# 2000张测试集边界后处理实验", "", f"状态：{'全部完成' if complete else '部分完成'}；已完成 {len(completed)}/{len(manifest['entries'])} 张。", "",
        "本表是公共边规则与本地自适应搜索控制器的对照，外部 GPT 调用数为 0；不能作为 GPT 独立贡献的证明。所有组从相同冻结模型预测开始，GT 只在修复输出保存后用于评价。", "",
        "|方法|顶点|减点率|宏平均BF1|微平均BF1|宏平均实例F1|Chamfer|质量通过场景|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ["raw", *METHODS]:
        a = aggregate[f"all/{method}"]
        def fmt(v):
            return '—' if v is None else f'{v:.5f}'
        lines.append(f"|{method}|{a['vertices_after']}|{a['reduction']:.2%}|{fmt(a['boundary_f1_macro'])}|{fmt(a['boundary_f1_micro'])}|{fmt(a['instance_f1_macro'])}|{fmt(a['chamfer_macro'])}|{a['quality_pass_scenes']}/{a['scenes']}|")
    lines += ["", "质量通过：相对同图原始预测 BF1 下降≤0.005、实例F1下降≤0.01、Chamfer增加≤0.25px、无新增重叠、ID不变、无非法几何。该标签只用于离线统计，不选择部署输出。", "",
        "空标注图保留；两侧都无实例时实例F1未定义，不填零或填一；单侧无边界时Chamfer未定义，单独报告有效场景数。详情见 summary.json 的分数据源统计。", "",
        "这些图像是切片，不等于2000个独立地理场景。最终论文需披露原分割模型测试集是否曾用于选模型，以及相邻切片的空间相关性。"]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if complete:
        atomic_json(root / "completed.json", {"images": len(completed), "methods": ["raw", *METHODS]})
    print(f"REPORT {len(completed)}/{len(manifest['entries'])}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("data"))
    ap.add_argument("--weights", type=Path, default=Path("../FieldBCQ/model/abc/best.pt"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage", choices=("prepare", "predict", "repair", "report", "all"), default="all")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()
    if min(args.threads, args.workers) < 1 or args.limit < 0:
        ap.error("Invalid worker/thread/limit values")
    if args.stage in ("prepare", "all", "predict"):
        manifest = prepare(args)
    else:
        manifest = load(args.output / "manifest.json")
        for path, digest in manifest["source_sha256"].items():
            if sha(path) != digest:
                raise ValueError("Source changed after freeze: " + path)
    if args.stage in ("all", "predict"):
        predict(args, manifest)
    if args.stage in ("all", "repair"):
        repair(args, manifest)
    if args.stage == "report":
        report(args, manifest)


if __name__ == "__main__":
    main()
