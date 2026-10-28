"""Local geometry experiments using frozen predictions; separate offline evaluation.

Examples are in docs/geometry_agent_local.md. No model inference is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys

import cv2
import numpy as np
from shapely.geometry import shape as geometry_shape

from fieldbqc_geometry import (GeometryAgent, GeometryConfig, GeometryInstance, GeometryTools,
                               feature_collection, from_parcels, to_parcels)
from predict_fieldbqc_agent import save_parcels as _save_parcels
from run_agent_experiments import fingerprint, load_parcels, read_json, write_json


def save_parcels(path, parcels, shape):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _save_parcels(Path(path), parcels, shape)


def credentials_from_environment():
    if os.environ.get("OPENAI_API_KEY"):
        return
    if os.name == "nt":
        import winreg
        for hive, key_path in ((winreg.HKEY_CURRENT_USER, "Environment"),
                               (winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    value, _ = winreg.QueryValueEx(key, "OPENAI_API_KEY")
                if isinstance(value, str) and value.strip():
                    os.environ["OPENAI_API_KEY"] = value.strip()
                    return
            except OSError:
                pass


def vector_input(path):
    data = read_json(path)
    if data.get("type") != "FeatureCollection":
        raise ValueError("Vector input must be a pixel-coordinate FeatureCollection")
    items = []
    for index, feature in enumerate(data["features"]):
        props = feature.get("properties", {})
        geom = geometry_shape(feature["geometry"])
        import shapely
        if not np.isfinite(shapely.get_coordinates(geom)).all():
            raise ValueError("Non-finite coordinates require source-data correction")
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Expected polygon or multipolygon input")
        uid = str(props.get("instance_id", index))
        items.append(GeometryInstance(uid, geom, float(props.get("score", 1)),
                                      int(props.get("class_id", 0)), tuple(props.get("source_ids", [uid]))))
    return items


def sources(args):
    if args.tiles:
        from fieldbqc_geometry_tiles import load_tile_scene
        image, items, pairs, metadata = load_tile_scene(args.tiles)
        yield str(metadata["scene_id"]), image, items, pairs, metadata, None
    elif args.geojson:
        if not args.image:
            raise ValueError("--geojson needs --image and pixel coordinates")
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot read vector reference image")
        yield args.geojson.stem, image, vector_input(args.geojson), [], {"source": str(args.image.resolve())}, None
    else:
        cache_config = read_json(args.cache / "run_config.json")
        paths = {Path(p).stem: Path(p) for p in cache_config["sources"]}
        files = sorted((args.cache / "original_masks").glob("*.npz"))
        if args.limit:
            files = files[:args.limit]
        if not files:
            raise ValueError("No cached masks found")
        for cached in files:
            parcels, shape = load_parcels(cached)
            image = cv2.imread(str(paths[cached.stem]), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != shape:
                raise ValueError(f"Image shape mismatch: {cached.stem}")
            yield cached.stem, image, from_parcels(parcels), [], {"source": str(paths[cached.stem])}, cached


def write_vectors(path, items, metadata):
    data = feature_collection(items)
    data["coordinate_space"] = {"units": "pixels", "origin": metadata.get("origin", [0, 0]),
                                "note": "Local pixel-cell geometry, not longitude/latitude"}
    write_json(path, data)


def run(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory; completed outputs are never overwritten")
    config = GeometryConfig(**(read_json(args.config) if args.config else {}))
    if args.target_vertices is not None:
        config = replace(config, target_vertices=args.target_vertices, min_vertices=args.target_vertices)
    if "gpt" in args.methods and not args.planner_config:
        raise ValueError("--planner-config is required for gpt")
    planner_settings = read_json(args.planner_config) if "gpt" in args.methods else None
    if planner_settings:
        credentials_from_environment()
        if not os.environ.get(planner_settings.get("api_key_env", "OPENAI_API_KEY")):
            raise ValueError("Configure the planner API key environment variable first")
        if args.direct:
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                os.environ.pop(name, None)
    args.output.mkdir(parents=True)
    code_names = ("fieldbqc_geometry.py", "fieldbqc_vertex_optimization.py", "fieldbqc_geometry_planner.py", "fieldbqc_geometry_tiles.py",
                  "fieldbqc_planner.py", "run_geometry_experiments.py")
    manifest = {"version": "field-geometry-target-v3", "methods": args.methods, "config": asdict(config),
                "thresholds_calibrated": False, "planner": planner_settings,
                "code_sha256": {n: fingerprint(Path(__file__).parent / n) for n in code_names},
                "inputs": [], "notes": "Development pilot; pixel units; no labels used by repair"}
    write_json(args.output / "manifest.json", manifest)
    for stem, image, items, seam_pairs, metadata, cached in sources(args):
        if Path(stem).name != stem or stem in {".", ".."}:
            raise ValueError("Unsafe image or scene ID")
        shape = image.shape[:2]
        protected = None
        if args.protected_dir and (args.protected_dir / (stem + ".png")).exists():
            protected = cv2.imread(str(args.protected_dir / (stem + ".png")), cv2.IMREAD_GRAYSCALE)
            if protected is None or protected.shape != shape:
                raise ValueError("Invalid protected-region mask")
            protected = protected > 0
        input_dir = args.output / "initial"
        save_parcels(input_dir / (stem + ".npz"), to_parcels(items, shape), shape)
        write_vectors(input_dir / (stem + ".pixel.geojson"), items, metadata)
        for method in args.methods:
            root = args.output / method
            tools = GeometryTools(image, config, protected, seam_pairs)
            planner = None
            if method == "gpt":
                from fieldbqc_geometry_planner import GeometryPlanner
                planner = GeometryPlanner(planner_settings, args.output / "planner_cache")
            state_files = {tuple(items): input_dir / (stem + ".npz")}
            def save_event_state(values, destination):
                key = tuple(values)
                if key not in state_files:
                    state = root / "event_states" / stem / f"{len(state_files):04d}.npz"
                    save_parcels(state, to_parcels(values, shape, tools.mask), shape)
                    state_files[key] = state
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(state_files[key], destination)
                except OSError:
                    shutil.copyfile(state_files[key], destination)
            def observe(iteration, before, candidate, action, ids, decision):
                base = root / "events" / stem
                save_event_state(before, base / f"{iteration:03d}_before.npz")
                save_event_state(candidate, base / f"{iteration:03d}_candidate.npz")
                write_json(base / f"{iteration:03d}.json", {"iteration": iteration, "action": action,
                                                           "target_ids": ids, "verification": decision})
            agent = GeometryAgent(tools, planner, "rule" if method == "gpt" else method, observe)
            result, trace = agent.run(items)
            save_parcels(root / "masks" / (stem + ".npz"), to_parcels(result, shape, tools.mask), shape)
            write_vectors(root / "vectors" / (stem + ".pixel.geojson"), result, metadata)
            vectors_path = root / "vectors" / (stem + ".pixel.geojson")
            vectors = read_json(vectors_path)
            for feature in vectors["features"]:
                props = feature["properties"]
                props.update(target_vertices=config.target_vertices,
                             vertex_target_met=props["vertex_count"] <= config.target_vertices)
            write_json(vectors_path, vectors)
            from evaluate_agent_experiments import write_csv
            (root / "vertex_reports").mkdir(parents=True, exist_ok=True)
            write_csv(root / "vertex_reports" / (stem + ".csv"), trace["vertex_report"]["parcels"])
            trace.update(method=method, metadata=metadata, image_shape=list(shape))
            write_json(root / "traces" / (stem + ".json"), trace)
            print(f"[{method}] {stem}: vertices {trace['initial']['vertices']}->{trace['final']['vertices']}; "
                  f"overlap {trace['initial']['overlap_excess_area']:.1f}->{trace['final']['overlap_excess_area']:.1f}; "
                  f"{trace['postprocess_seconds']:.2f}s", flush=True)
        manifest["inputs"].append({"image_id": stem, "source": metadata,
                                   "cached_sha256": fingerprint(cached) if cached else None})
        write_json(args.output / "manifest.json", manifest)
    if not manifest["inputs"]:
        raise ValueError("No inputs were processed")
    write_json(args.output / "completed.json", {"images": len(manifest["inputs"]), "methods": args.methods})


def evaluate(args):
    """Offline only: GT metrics never enter GeometryAgent or its planner."""
    from evaluate_agent_experiments import measure, instances, summary, write_csv, paired_interval
    from evaluate_mechanism_metrics import read_yolo_masks
    if not (args.runs / "completed.json").exists():
        raise ValueError("Cannot evaluate incomplete runs")
    manifest = read_json(args.runs / "manifest.json")
    rows, summaries, event_rows = [], [], []
    for method in manifest["methods"]:
        total = None
        method_rows = []
        accepted, attempts, failures, review_count = 0, 0, 0, 0
        vertex_rows = []
        for item in manifest["inputs"]:
            stem = item["image_id"]
            label = args.dataset_root / "labels" / args.split / (stem + ".txt")
            if not label.exists():
                raise FileNotFoundError(f"No label for {stem}; use a scene-level evaluator for tile mosaics")
            before, shape = load_parcels(args.runs / "initial" / (stem + ".npz"))
            after, final_shape = load_parcels(args.runs / method / "masks" / (stem + ".npz"))
            if final_shape != shape:
                raise ValueError("Prediction shape mismatch")
            gt = read_yolo_masks(label, *shape)
            a, _ = measure(gt, instances(before), shape)
            b, counts = measure(gt, instances(after), shape)
            # Audit files share immutable states. Repeated before/unchanged
            # candidates have identical bytes; memoization changes no metric.
            state_results = {fingerprint(args.runs / "initial" / (stem + ".npz")): a,
                             fingerprint(args.runs / method / "masks" / (stem + ".npz")): b}
            def measured_state(path):
                digest = fingerprint(path)
                if digest not in state_results:
                    values, event_shape = load_parcels(path)
                    if event_shape != shape:
                        raise ValueError("Audit state shape mismatch")
                    state_results[digest] = measure(gt, instances(values), shape)[0]
                return state_results[digest]
            total = counts if total is None else {k: total[k] + counts[k] for k in total}
            trace = read_json(args.runs / method / "traces" / (stem + ".json"))
            vertex_rows.extend(dict(r, method=method, image_id=stem)
                               for r in trace.get("vertex_report", {}).get("parcels", []))
            action_rows = [e for e in trace["action_trace"] if e["action"] in __import__("fieldbqc_geometry").ACTIONS]
            attempts += len(action_rows)
            accepted += sum(e.get("accepted", False) for e in action_rows)
            failures += sum("tool_error" in e.get("reasons", []) for e in action_rows)
            review_count += bool(trace["needs_review"])
            row = {"method": method, "image_id": stem, "pq": b["pq"], "boundary_f1": b["boundary_f1"],
                   "delta_pq": None if a["pq"] is None or b["pq"] is None else b["pq"] - a["pq"],
                   "delta_boundary_f1": b["boundary_f1"] - a["boundary_f1"],
                   "vertices_before": trace["initial"]["vertices"], "vertices_after": trace["final"]["vertices"],
                   "overlap_before": trace["initial"]["overlap_excess_area"],
                   "overlap_after": trace["final"]["overlap_excess_area"],
                   "seconds": trace["postprocess_seconds"]}
            row["harmful_image"] = ((row["delta_pq"] is not None and row["delta_pq"] < -0.005)
                                    or row["delta_boundary_f1"] < -0.005
                                    or row["overlap_after"] > row["overlap_before"] + 1e-7)
            rows.append(row)
            method_rows.append(row)
            for event in sorted((args.runs / method / "events" / stem).glob("*.json")):
                record = read_json(event)
                x = measured_state(event.with_name(event.stem + "_before.npz"))
                y = measured_state(event.with_name(event.stem + "_candidate.npz"))
                event_rows.append({"method": method, "image_id": stem, "iteration": record["iteration"],
                                   "action": record["action"], "accepted": record["verification"]["accept"],
                                   "reasons": ";".join(record["verification"]["reasons"]),
                                   "delta_pq": None if x["pq"] is None or y["pq"] is None else y["pq"] - x["pq"],
                                   "delta_boundary_f1": y["boundary_f1"] - x["boundary_f1"]})
            print(f"[evaluate {method}] {stem}: PQ={b['pq']}; BoundaryF1={b['boundary_f1']:.6f}", flush=True)
        final = summary(total)
        vertex_counts = [r["vertices_after"] for r in vertex_rows]
        v0 = sum(r["vertices_before"] for r in method_rows)
        v1 = sum(r["vertices_after"] for r in method_rows)
        o0 = sum(r["overlap_before"] for r in method_rows)
        o1 = sum(r["overlap_after"] for r in method_rows)
        summaries.append({"method": method, "images": len(method_rows), "pq": final["pq"],
                          "boundary_f1": final["boundary_f1"], "instance_f1": final["instance_f1"],
                          "vertices_before": v0, "vertices_after": v1, "vertex_reduction": (v0-v1)/max(v0, 1),
                          "overlap_before": o0, "overlap_after": o1,
                          "mean_seconds": float(np.mean([r["seconds"] for r in method_rows])),
                          "attempts": attempts, "accepted": accepted, "tool_errors": failures,
                          "harmful_images": sum(r["harmful_image"] for r in method_rows),
                          "review_images": review_count,
                          "vertex_target": manifest["config"].get("target_vertices"),
                          "parcels": len(vertex_rows),
                          "parcels_at_target": sum(r["target_met"] for r in vertex_rows),
                          "target_met_fraction": sum(r["target_met"] for r in vertex_rows)/max(1, len(vertex_rows)),
                          "mean_vertices": float(np.mean(vertex_counts)) if vertex_counts else None,
                          "median_vertices": float(np.median(vertex_counts)) if vertex_counts else None,
                          "p90_vertices": float(np.percentile(vertex_counts, 90)) if vertex_counts else None,
                          "paired_pq": paired_interval(method_rows, "pq", 2000, 20261019),
                          "paired_boundary_f1": paired_interval(method_rows, "boundary_f1", 2000, 20261019)})
    root = args.runs / "evaluation"
    root.mkdir(exist_ok=True)
    write_json(root / "provenance.json", {
        "evaluator_sha256": fingerprint(Path(__file__)),
        "metrics_sha256": fingerprint(Path(__file__).parent / "evaluate_agent_experiments.py"),
        "dataset_root": str(args.dataset_root.resolve()), "split": args.split,
        "event_audit": True, "memoized_by": "immutable_npz_sha256"})
    write_csv(root / "per_image.csv", rows)
    write_csv(root / "events.csv", event_rows)
    write_json(root / "summary.json", summaries)
    write_csv(root / "summary.csv", [{k: v for k, v in s.items() if not isinstance(v, dict)} for s in summaries])
    lines = ["# Geometry development pilot", "", "Thresholds are uncalibrated. GT used only in offline evaluation.",
             "", "| Method | PQ | Boundary F1 | Vertex reduction | Overlap excess area | Harmful images | Seconds/image |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['method']} | {s['pq']:.6f} | {s['boundary_f1']:.6f} | {s['vertex_reduction']:.1%} | "
                     f"{s['overlap_before']:.1f} → {s['overlap_after']:.1f} | {s['harmful_images']} | {s['mean_seconds']:.2f} |")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--cache", type=Path)
    group.add_argument("--geojson", type=Path)
    group.add_argument("--tiles", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=["baseline", "compact", "fixed", "rule", "gpt"], default=["baseline", "compact", "rule"])
    p.add_argument("--planner-config", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--target-vertices", type=int, help="Desired maximum per parcel; constraints remain hard (default 10)")
    p.add_argument("--protected-dir", type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--direct", action="store_true", help="Disable proxy variables in this process only")
    p = commands.add_parser("evaluate")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split", default="val")
    args = parser.parse_args()
    if args.command == "run" and (args.limit < 0 or len(set(args.methods)) != len(args.methods)):
        parser.error("Limit must be nonnegative and methods unique")
    (run if args.command == "run" else evaluate)(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
