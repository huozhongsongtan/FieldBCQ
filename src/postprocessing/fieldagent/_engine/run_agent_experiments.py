"""Prepare predictions, run paired repair experiments, or evaluate them offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from fieldbqc_agent import AgentConfig, FieldBqcAgent, LocalReinfer, Parcel, parcels_from_ultralytics
from fieldbqc_planner import APIPlanner
from fieldbqc_policy import DomainRouter, image_metadata
from fieldbqc_recipes import RepairRecipeLibrary
from predict_fieldbqc_agent import _build_quality_btq_predictor, _serializable_trace, resolve_sources, save_parcels

METHODS = ("baseline", "fixed", "fixed-blind", "rule", "gpt", "qwen", "rule-no-verifier", "gpt-no-verifier", "qwen-no-verifier")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parcels(path):
    with np.load(path, allow_pickle=False) as arrays:
        masks = arrays["masks"].astype(bool)
        if masks.ndim != 3:
            raise ValueError(f"Invalid masks in {path}")
        count = len(masks)
        scores, classes = arrays["scores"], arrays["classes"]
        ids = arrays["parcel_ids"] if "parcel_ids" in arrays else np.arange(count).astype(str)
        if any(len(values) != count for values in (scores, classes, ids)):
            raise ValueError(f"Inconsistent prediction arrays in {path}")
        quality = arrays["qualities"] if "qualities" in arrays else np.full(count, np.nan)
        valid = arrays["quality_valid"] if "quality_valid" in arrays else np.isfinite(quality)
        mask_quality = arrays["mask_qualities"] if "mask_qualities" in arrays else np.full(count, np.nan)
        boundary = arrays["boundary_qualities"] if "boundary_qualities" in arrays else np.full(count, np.nan)
        sources = arrays["quality_sources"] if "quality_sources" in arrays else np.full(count, "unavailable")
        def score(values, index):
            return float(values[index]) if valid[index] and np.isfinite(values[index]) else None
        return [Parcel(mask.copy(), float(scores[i]), int(classes[i]), score(quality, i),
                       score(mask_quality, i), score(boundary, i), str(ids[i]),
                       bool(valid[i]), str(sources[i])) for i, mask in enumerate(masks)], tuple(masks.shape[1:])


def prepare(args):
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    command = [sys.executable, str(Path(__file__).with_name("predict_fieldbqc_agent.py")),
               "--weights", str(args.weights.resolve()), "--source", args.source,
               "--output", str(args.output), "--imgsz", str(args.imgsz), "--conf", str(args.conf),
               "--iou", str(args.iou), "--max-det", str(args.max_det), "--limit", str(args.limit),
               "--max-iterations", "0"]
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.manifest:
        command.extend(["--manifest", str(args.manifest)])
    subprocess.run(command, check=True)
    config = read_json(args.output / "run_config.json")
    config["checkpoint_sha256"] = fingerprint(args.weights)
    config["cache_version"] = 1
    write_json(args.output / "run_config.json", config)


def image_index(cache_config, override):
    paths = resolve_sources(str(override)) if override else [Path(p) for p in cache_config["sources"]]
    index = {path.stem: path for path in paths}
    if len(index) != len(paths):
        raise ValueError("Image stems must be unique")
    return index


def local_predictor(args, cache_config, config, expected_hash):
    if args.disable_local or config.max_local_calls == 0:
        return None
    weights = args.weights or Path(cache_config["arguments"]["weights"])
    if not weights.is_file():
        raise FileNotFoundError("Provide --weights with the server checkpoint path, or use --disable-local")
    if expected_hash and fingerprint(weights) != expected_hash:
        raise ValueError("Local inference checkpoint differs from the initial-prediction checkpoint")
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor

    model = YOLO(str(weights), task="segment")
    overrides = {"predictor": _build_quality_btq_predictor(SegmentationPredictor, use_topology_nms=False),
                 "stream": False, "retina_masks": True, "save": False, "verbose": False,
                 "iou": cache_config["arguments"].get("iou", 0.7),
                 "max_det": cache_config["arguments"].get("max_det", 710)}
    if args.device is not None:
        overrides["device"] = args.device
    def predict(crop, imgsz, conf):
        return parcels_from_ultralytics(model.predict(source=crop, imgsz=imgsz, conf=conf, **overrides)[0])
    # Exclude one-time model setup and kernel warmup from individual method timing.
    predict(np.zeros((64, 64, 3), dtype=np.uint8), 64, 0.25)
    return predict


def run(args):
    cache_config = read_json(args.cache / "run_config.json")
    cache_files = sorted((args.cache / "original_masks").glob("*.npz"))
    if args.limit:
        cache_files = cache_files[:args.limit]
    if not cache_files:
        raise ValueError("No original_masks/*.npz in cache")
    paths = image_index(cache_config, args.images)
    for path in cache_files:
        if path.stem not in paths or not paths[path.stem].is_file():
            raise FileNotFoundError(f"Missing image {path.stem}; use --images with the server image directory")
    values = read_json(args.agent_config) if args.agent_config else {}
    config = AgentConfig(**dict({"enable_boundary_tool": False, "max_iterations": 8, "max_seconds": 120}, **values))
    if args.disable_local:
        config = replace(config, max_local_calls=0)
    policies = read_json(args.policy_config) if args.policy_config else {
        name: {k: v for k, v in value.items() if k != "domain"}
        for name, value in cache_config.get("policies", {}).items()}
    router = DomainRouter(policies)
    recipe_config = getattr(args, "recipe_config", None)
    recipes = RepairRecipeLibrary.from_json(recipe_config) if recipe_config else RepairRecipeLibrary()
    planners = {}
    for name in ("gpt", "qwen"):
        if any(method.startswith(name) for method in args.methods):
            setting = getattr(args, name + "_config")
            if setting is None:
                raise ValueError(f"--{name}-config is required when running {name}")
            planners[name] = APIPlanner(read_json(setting), args.output / "planner_cache" / name)
            if not os.environ.get(planners[name].config.get("api_key_env", "OPENAI_API_KEY")):
                raise ValueError(f"Missing API key environment variable for {name}; configure it before this experiment")
    for method in args.methods:
        if (args.output / method).exists() and not getattr(args, "resume", False):
            raise FileExistsError(f"{method} already exists; use a new output directory or select other methods")
    has_tools = any(method != "baseline" for method in args.methods)
    predict = local_predictor(args, cache_config, config, cache_config.get("checkpoint_sha256")) if has_tools else None
    cache_signature = hashlib.sha256("".join(fingerprint(p) for p in cache_files).encode("ascii")).hexdigest()
    experiment_config = {"cache": str(args.cache.resolve()), "cache_signature": cache_signature,
                         "checkpoint_sha256": cache_config.get("checkpoint_sha256"),
                         "agent_config": asdict(config), "policies": policies, "seed": 0,
                         "thresholds_calibrated": False, "image_ids": [p.stem for p in cache_files],
                         "initial_inference_arguments": cache_config["arguments"],
                         "planner_configs": {name: planner.config for name, planner in planners.items()},
                         "recipes": recipes.to_dict()}
    args.output.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        root = args.output / method
        method_config = dict(experiment_config, method=method,
                             verifier_enabled=not method.endswith("-no-verifier"))
        if root.exists():
            if not (root / "method_config.json").exists() or read_json(root / "method_config.json") != method_config:
                raise ValueError(f"Cannot resume {method}: configuration or cache changed")
            if (root / "completed.json").exists():
                print(f"[{method}] already completed", flush=True)
                continue
        for folder in ("masks", "traces", "events"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        write_json(root / "method_config.json", method_config)
        for index, cached in enumerate(cache_files, start=1):
            mask_path = root / "masks" / cached.name
            trace_path = root / "traces" / (cached.stem + ".json")
            if getattr(args, "resume", False) and mask_path.exists() and trace_path.exists():
                load_parcels(mask_path)
                read_json(trace_path)
                print(f"[{method} {index}/{len(cache_files)}] {cached.stem} resumed", flush=True)
                continue
            parcels, shape = load_parcels(cached)
            image = cv2.imread(str(paths[cached.stem]), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != shape:
                raise ValueError(f"Image/mask shape mismatch: {paths[cached.stem]}")
            prior_trace = args.cache / "traces" / (cached.stem + ".json")
            metadata = read_json(prior_trace).get("metadata", {}) if prior_trace.exists() else image_metadata(
                str(paths[cached.stem]), shape[1], shape[0])
            event_dir = root / "events" / cached.stem
            if getattr(args, "resume", False) and event_dir.exists():
                # Only remove this runner's unfinished event files for this image.
                for pattern in ("[0-9][0-9][0-9]*.json", "[0-9][0-9][0-9]*.npz", "[0-9][0-9][0-9]*.json.tmp"):
                    for incomplete in event_dir.glob(pattern):
                        incomplete.unlink()
            def observe(iteration, before, proposal, decision):
                event_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"{iteration:03d}"
                save_parcels(event_dir / (prefix + "_before.npz"), before, shape)
                save_parcels(event_dir / (prefix + "_candidate.npz"), proposal.parcels, shape)
                write_json(event_dir / (prefix + ".json"), {
                    "iteration": iteration, "action": proposal.action, "target_ids": proposal.target_ids,
                    "accepted": decision.accept, "verification": asdict(decision)})
            routing_mode = "fixed" if method == "fixed" else "blind" if method == "fixed-blind" else "rule"
            agent = FieldBqcAgent(config, local_tool=LocalReinfer(predict, config) if predict else None,
                                  router=router, planner=planners.get(method.split("-")[0]),
                                  routing_mode=routing_mode,
                                  verify=not method.endswith("-no-verifier"), proposal_observer=observe,
                                  recipes=recipes)
            started = time.perf_counter()
            initial_state = agent.inspect(image, parcels, metadata)
            if method == "baseline":
                output = FieldBqcAgent(replace(config, max_iterations=0), router=router).run(image, parcels, metadata)
            else:
                output = agent.run(image, parcels, metadata)
            elapsed = time.perf_counter() - started
            save_parcels(mask_path, output.parcels, shape)
            trace = _serializable_trace(output, str(paths[cached.stem]), metadata)
            trace.update(method=method, postprocess_seconds=elapsed,
                         initial_state=asdict(initial_state), initial_cache=str(cached.resolve()))
            write_json(trace_path, trace)
            print(f"[{method} {index}/{len(cache_files)}] {cached.stem} "
                  f"{len(parcels)}->{len(output.parcels)} review={output.needs_review} {elapsed:.2f}s", flush=True)
        write_json(root / "completed.json", {"images": len(cache_files), "method": method})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Cache initial predictions once; no labels are loaded")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=710)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--manifest", type=Path)
    p.set_defaults(function=prepare)
    p = commands.add_parser("run", help="Apply methods to identical cached predictions; no labels are loaded")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", choices=METHODS, nargs="+", default=["baseline", "fixed", "rule"])
    p.add_argument("--images", type=Path, help="Override source paths when moving a cache between machines")
    p.add_argument("--weights", type=Path, help="Same checkpoint, required for local inference if cache path moved")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--agent-config", type=Path)
    p.add_argument("--policy-config", type=Path)
    p.add_argument("--recipe-config", type=Path)
    p.add_argument("--gpt-config", type=Path)
    p.add_argument("--qwen-config", type=Path)
    p.add_argument("--disable-local", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume completed images only if cache and config match")
    p.set_defaults(function=run)
    p = commands.add_parser("evaluate", help="Offline evaluation; ground truth never enters planning")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--boundary-tolerance", type=int, default=2)
    p.add_argument("--allow-missing-labels", action="store_true")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    def evaluate(args):
        from evaluate_agent_experiments import evaluate_runs
        evaluate_runs(args)
    p.set_defaults(function=evaluate)
    args = parser.parse_args()
    if getattr(args, "limit", 0) < 0:
        parser.error("--limit must be nonnegative")
    if hasattr(args, "methods") and len(set(args.methods)) != len(args.methods):
        parser.error("--methods must be unique")
    args.function(args)


if __name__ == "__main__":
    main()
