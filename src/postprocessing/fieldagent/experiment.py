"""Four-arm experiment with frozen inputs and explicit GPT-error continuation."""
from contextlib import contextmanager
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from . import activate_engine, PACKAGE_ROOT, PROJECT_ROOT, __version__
from .io import digest, load, save, now, read_inputs, planner_config

LOCAL_METHODS = ("public_edge", "local_adaptive_strict", "local_adaptive_balanced")
CORE_METHODS = ("raw", "public_edge", "local_adaptive_strict", "gpt_autonomous")


def source_hashes():
    paths = list(PACKAGE_ROOT.rglob("*.py")) + list(PACKAGE_ROOT.rglob("*.md"))
    return {p.relative_to(PACKAGE_ROOT).as_posix(): digest(p) for p in sorted(paths)}


def freeze(inputs, out, methods, calls_per_target, planner):
    if not methods or len(set(methods)) != len(methods) or set(methods) - {*LOCAL_METHODS, "gpt_autonomous"}:
        raise ValueError("Distinct known methods are required; raw is always included")
    if type(calls_per_target) is not int or calls_per_target < 1:
        raise ValueError("calls_per_target must be positive")
    entries = read_inputs(inputs)
    activate_engine()
    from run_testset_experiment import CONFIGS
    record = {"version": __version__, "inputs_sha256": digest(inputs), "entries": entries,
              "methods": ["raw", *methods], "calls_per_target": calls_per_target,
              "planner": planner if "gpt_autonomous" in methods else None,
              "local_configs": {m: asdict(CONFIGS[m]) for m in methods if m in CONFIGS},
              "gpt_config": asdict(CONFIGS["local_adaptive_strict"]) if "gpt_autonomous" in methods else None,
              "source_sha256": source_hashes(), "runtime_GT_access": False,
              "scope": "boundary_vertex_reduction_only", "automatic_API_retries": 0}
    out = Path(out)
    manifest = out / "manifest.json"
    if manifest.exists():
        prior = load(manifest)
        if {k: v for k, v in prior.items() if k != "created_utc"} != record:
            raise ValueError("Frozen inputs, methods, code or settings changed; use a new output folder")
        return prior
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("A new experiment requires an empty output folder")
    for relative in record["source_sha256"]:
        destination = out / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PACKAGE_ROOT / relative, destination)
    record["created_utc"] = now()
    save(manifest, record)
    return record


@contextmanager
def stage_lock(out, stage):
    path = Path(out) / (stage + ".lock")
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_utc": now()}, stream)
    except FileExistsError:
        raise RuntimeError("Stage already active or interrupted; inspect lock before restarting: " + str(path)) from None
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def check_inputs(entry):
    for name in ("image", "prediction_vector"):
        if digest(entry[name]) != entry[name + "_sha256"]:
            raise ValueError("Frozen input changed: " + entry[name])


def read_scene(entry):
    """Only image and prediction enter the repair code, never the label path."""
    check_inputs(entry)
    from cvr_agent import read_items
    from run_boundary_tools import load_image
    import shapely
    import numpy as np
    raw = read_items(entry["prediction_vector"])
    image = load_image(entry["image"])
    if len({p.identifier for p in raw}) != len(raw):
        raise ValueError("Duplicate instance IDs")
    h, w = image.shape[:2]
    for parcel in raw:
        geometry = parcel.geometry
        if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid:
            raise ValueError("Input polygon must be valid: " + parcel.identifier)
        if not np.isfinite(shapely.get_coordinates(geometry)).all():
            raise ValueError("Non-finite input coordinates")
        x0, y0, x1, y1 = geometry.bounds
        if x0 < -1e-7 or y0 < -1e-7 or x1 > w + 1e-7 or y1 > h + 1e-7:
            raise ValueError("Expected original-image pixel coordinates: " + parcel.identifier)
    return image, raw


def persist_method(out, entry, method, items, raw, image, trace=None):
    """Persist/fingerprint geometry first; read GT only in the following step."""
    from cvr_agent import save_items, vertices
    from run_testset_experiment import offline_evaluate
    folder = Path(out) / "scenes" / entry["scene"] / method
    folder.mkdir(parents=True, exist_ok=True)
    vector = folder / "final.pixel.geojson"
    if not vector.exists():
        save_items(vector, items)
    record = {"scene": entry["scene"], "domain": entry["domain"], "method": method,
              "final_sha256": digest(vector), "runtime_GT_access": False,
              "parcel_vertices": [{"instance_id": p.identifier, "before": vertices(a.geometry),
                                    "after": vertices(p.geometry),
                                    "boundary_shift_px": a.geometry.boundary.hausdorff_distance(p.geometry.boundary),
                                    "area_change_fraction": abs(p.geometry.area-a.geometry.area)/max(a.geometry.area, 1.)}
                                   for a, p in zip(raw, items)],
              "seconds": (trace or {}).get("seconds", 0.),
              "model_calls": (trace or {}).get("model_calls", 0),
              "direct_committed": (trace or {}).get("direct_committed", 0),
              "commits": (trace or {}).get("commits", 0),
              "unfinished_target_ids": (trace or {}).get("unfinished_target_ids", []),
              "processed_targets": (trace or {}).get("processed_targets"),
              "prediction_instances": len(items), "vertices": sum(vertices(p.geometry) for p in items)}
    if [p.identifier for p in raw] != [p.identifier for p in items]:
        raise ValueError("Instance order or identity changed")
    if entry.get("label"):
        if digest(entry["label"]) != entry["label_sha256"]:
            raise ValueError("Offline reference changed")
        record["evaluation"] = offline_evaluate(items, raw, entry["label"], image.shape[:2])
    save(folder / "completed.json", record)
    return record


def completed_method(out, entry, method):
    folder = Path(out) / "scenes" / entry["scene"] / method
    if not (folder / "completed.json").exists():
        return False
    record = load(folder / "completed.json")
    if digest(folder / "final.pixel.geojson") != record["final_sha256"]:
        raise ValueError("Completed geometry changed")
    return True


def run_local(manifest, out):
    from run_testset_experiment import run_local_method
    from cvr_agent import read_items
    methods = [m for m in manifest["methods"] if m in LOCAL_METHODS]
    if not methods:
        raise ValueError("No local methods selected")
    with stage_lock(out, "local"):
        for entry in manifest["entries"]:
            if all(completed_method(out, entry, m) for m in ["raw", *methods]):
                continue
            image, raw = read_scene(entry)
            if not completed_method(out, entry, "raw"):
                persist_method(out, entry, "raw", raw, raw, image)
            for method in methods:
                if completed_method(out, entry, method):
                    continue
                folder = Path(out) / "scenes" / entry["scene"] / method
                if folder.exists():
                    if (folder / "trace.json").exists() and (folder / "final.pixel.geojson").exists():
                        items = read_items(folder / "final.pixel.geojson")
                        persist_method(out, entry, method, items, raw, image, load(folder / "trace.json"))
                        continue
                    folder.rename(folder.with_name(method + f".interrupted-{time.time_ns()}"))
                items = run_local_method(image, raw, folder, method)
                persist_method(out, entry, method, items, raw, image, load(folder / "trace.json"))
            write_progress(manifest, out, "running_local", entry["scene"])
            print("LOCAL", entry["scene"], flush=True)
        write_progress(manifest, out, "local_complete")


def resume_no_budget(run, prior):
    """A failed last request is still charged; retain it without an extra request."""
    snapshot = run / "recovery_snapshots" / str(time.time_ns())
    snapshot.mkdir(parents=True)
    shutil.copyfile(run / "trace.json", snapshot / "trace.json")
    trace = dict(prior, gpt_error=None, stop_reason="original_attempt_budget_exhausted",
                 recovered_external_errors=[*prior.get("recovered_external_errors", []),
                    {"error": prior["gpt_error"], "retained_snapshot": str(snapshot / "trace.json")}])
    save(run / "trace.json", trace)
    return trace


def run_gpt(manifest, out, *, resume_after_error=False):
    if "gpt_autonomous" not in manifest["methods"]:
        raise ValueError("GPT is not selected in this frozen experiment")
    if (Path(out) / "runtime_failure.json").exists():
        raise RuntimeError("An unreconciled runtime failure exists; inspect it before any paid replay")
    settings = manifest["planner"]
    if not os.environ.get(settings.get("api_key_env", "OPENAI_API_KEY")):
        raise RuntimeError("Set the configured API key in this terminal environment; never put it in a file")
    from .gpt import boundary_only_capabilities, empty_prediction_call
    from .continuation import continue_scene
    from .archive import archive_call
    from boundary_executor import BoundaryConfig
    from cvr_agent import read_items
    ba, _ = boundary_only_capabilities()
    failure = Path(out) / "external_failure.json"
    if failure.exists():
        if not resume_after_error:
            raise RuntimeError("External service previously failed; use --resume-after-error for one explicit continuation")
        session = Path(out) / "recovery_sessions" / str(time.time_ns())
        session.mkdir(parents=True)
        failure.rename(session / "prior_external_failure.json")
    with stage_lock(out, "gpt"):
        for entry in manifest["entries"]:
            if completed_method(out, entry, "gpt_autonomous"):
                continue
            image, raw = read_scene(entry)
            if not completed_method(out, entry, "raw"):
                persist_method(out, entry, "raw", raw, raw, image)
            folder = Path(out) / "scenes" / entry["scene"] / "gpt_autonomous"
            run = folder / "run"
            folder.mkdir(parents=True, exist_ok=True)
            calls = manifest["calls_per_target"]
            config_values = dict(manifest["gpt_config"])
            # Match the inherited full-test protocol, including its scene deadline.
            config_values.update(budget=10000, seconds=max(7200., len(raw)*calls*90.+300.))
            config = BoundaryConfig(**config_values)
            try:
                if not raw:
                    if (run / "empty_scene" / "error.json").exists():
                        if not resume_after_error:
                            raise RuntimeError("Explicit continuation required for the interrupted empty-scene inspection")
                        run.rename(run.with_name("run.interrupted-" + str(time.time_ns())))
                    trace = empty_prediction_call(image, run / "empty_scene", settings)
                    items = []
                elif (run / "trace.json").exists():
                    prior = load(run / "trace.json")
                    if prior.get("gpt_error"):
                        if not resume_after_error:
                            raise RuntimeError("Explicit continuation required for the interrupted GPT scene")
                        remaining = [uid for uid in prior["unfinished_target_ids"]
                                     if prior["per_target_calls"].get(uid, 0) < calls]
                        trace = (continue_scene(ba, image, raw, run, config, settings, calls)
                                 if remaining else resume_no_budget(run, prior))
                    else:
                        trace = prior
                    items = read_items(run / "final/prediction.repaired.pixel.geojson")
                else:
                    if run.exists() and any(run.iterdir()):
                        raise ValueError("Interrupted paid run has no reconciled trace; inspect before any replay")
                    trace = ba.run_autonomous(image, raw, run, config, settings,
                        max_calls=max(1, len(raw)*calls), calls_per_target=calls,
                        allow_direct=True, goal_mode="minimize_vertices")
                    items = read_items(run / "final/prediction.repaired.pixel.geojson")
                for request in run.glob("gpt_calls/*/request.json"):
                    archive_call(request.parent, Path(out))
                if trace.get("gpt_error"):
                    save(failure, {"scene": entry["scene"], "error": trace["gpt_error"], "updated_utc": now(),
                                   "automatic_retry": False, "attempts_still_charged": True})
                    write_progress(manifest, out, "stopped_external_error", entry["scene"])
                    raise RuntimeError(trace["gpt_error"])
                persist_method(out, entry, "gpt_autonomous", items, raw, image, trace)
                write_progress(manifest, out, "running_gpt", entry["scene"])
                print("GPT", entry["scene"], flush=True)
            except Exception as error:
                if not failure.exists():
                    if (run / "empty_scene/error.json").exists():
                        save(failure, {"scene": entry["scene"], "error": str(error), "updated_utc": now(),
                                       "automatic_retry": False, "empty_prediction": True})
                        write_progress(manifest, out, "stopped_external_error", entry["scene"])
                    else:
                        save(Path(out) / "runtime_failure.json", {"scene": entry["scene"], "error": str(error),
                             "traceback": traceback.format_exc(), "updated_utc": now(), "no_automatic_replay": True})
                raise
        write_progress(manifest, out, "gpt_budgeted_pass_complete")


def write_progress(manifest, out, status, scene=None):
    counts = {m: sum((Path(out)/"scenes"/e["scene"]/m/"completed.json").exists()
                    for e in manifest["entries"]) for m in manifest["methods"]}
    save(Path(out) / "progress.json", {"status": status, "expected_images": len(manifest["entries"]),
         "completed_by_method": counts, "scene": scene, "updated_utc": now()})


def csv_file(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def usage_records(out):
    """Include failed attempts in call counts and any usage returned by the service."""
    rows = []
    for scene in (Path(out) / "scenes").glob("*"):
        folders = set(scene.glob("gpt_autonomous/run*/**/gpt_calls/*"))
        folders.update(scene.glob("gpt_autonomous/run*/empty_scene"))
        for call in sorted(folders):
            source = call / "metadata.json"
            if not source.exists():
                source = call / "error.json"
            if not source.exists():
                continue
            meta = load(source)
            usage = meta.get("usage", {})
            rows.append({"scene": scene.name, "call": str(call.relative_to(out)),
                         "successful_response": source.name == "metadata.json",
                         "requested_model": meta.get("requested_model"), "response_model": meta.get("response_model"),
                         "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                         "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                         "seconds": meta.get("seconds"), "error": meta.get("error"),
                         "request_sha256": meta.get("request_sha256")})
    return rows


def aggregate(records):
    import numpy as np
    from evaluate_agent_experiments import summary
    values = [p["after"] for record in records for p in record["parcel_vertices"]]
    before = sum(p["before"] for record in records for p in record["parcel_vertices"])
    total = sum(values)
    result = {"images": len(records), "instances": len(values), "vertices_before": before,
              "vertices_after": total, "reduction": 1-total/before if before else None,
              "mean_vertices": total/len(values) if values else None,
              "median_vertices": float(np.median(values)) if values else None,
              "p90_vertices": float(np.percentile(values, 90)) if values else None,
              "postprocess_seconds": sum(r["seconds"] for r in records),
              "attempted_model_calls": sum(r["model_calls"] for r in records),
              "direct_committed": sum(r["direct_committed"] for r in records),
              "commits": sum(r["commits"] for r in records),
              "budget_unfinished_targets": sum(len(r["unfinished_target_ids"]) for r in records)}
    evaluated = [r["evaluation"] for r in records if "evaluation" in r]
    result["evaluated_images"] = len(evaluated)
    for key, metric in (("boundary_f1_macro", "boundary_f1"), ("chamfer_macro_px", "pixel_boundary_chamfer"),
                        ("boundary_p95_macro_px", "pixel_boundary_p95"), ("instance_f1_macro", "instance_f1_mean")):
        defined = [e["metrics"][metric] for e in evaluated if e["metrics"][metric] is not None]
        result[key] = float(np.mean(defined)) if defined else None
        result[key + "_defined_images"] = len(defined)
    if evaluated:
        count_keys = evaluated[0]["counts"]["0.5"].keys()
        counts = {k: sum(e["counts"]["0.5"][k] for e in evaluated) for k in count_keys}
        result["boundary_f1_micro"] = summary(counts)["boundary_f1"]
    else:
        result["boundary_f1_micro"] = None
    return result


def build_report(manifest, out):
    """No missing GPT image can enter the four-arm paired table."""
    out = Path(out)
    methods = manifest["methods"]
    records = {}
    for entry in manifest["entries"]:
        for method in methods:
            if completed_method(out, entry, method):
                records[(entry["scene"], method)] = load(out/"scenes"/entry["scene"]/method/"completed.json")
    local = [m for m in methods if m != "gpt_autonomous"]
    groups = {"local_paired": local}
    if "gpt_autonomous" in methods:
        groups["gpt_paired"] = methods
    result = {"expected_images": len(manifest["entries"]), "scope": "boundary_vertex_reduction_only",
              "complete_by_method": {m: sum(key[1] == m for key in records) for m in methods},
              "paired_groups": {}, "updated_utc": now()}
    rows, parcels = [], []
    for name, required in groups.items():
        scenes = [e["scene"] for e in manifest["entries"] if all((e["scene"], m) in records for m in required)]
        domains = ["all", *sorted({records[(s, required[0])]["domain"] for s in scenes})]
        grouped = {}
        for domain in domains:
            chosen = [s for s in scenes if domain == "all" or records[(s, required[0])]["domain"] == domain]
            grouped[domain] = {m: aggregate([records[(s, m)] for s in chosen]) for m in required} if chosen else {}
        result["paired_groups"][name] = {"images": len(scenes), "scenes": scenes, "aggregate": grouped}
        for domain, data in grouped.items():
            rows.extend({"comparison": name, "domain": domain, "method": m, **metrics} for m, metrics in data.items())
    for (scene, method), record in records.items():
        parcels.extend({"scene": scene, "domain": record["domain"], "method": method, **p} for p in record["parcel_vertices"])
    calls = usage_records(out)
    result["API"] = {"attempts_with_terminal_metadata": len(calls),
                     "successful_responses": sum(r["successful_response"] for r in calls),
                     "input_tokens_reported": sum(r["input_tokens"] or 0 for r in calls),
                     "output_tokens_reported": sum(r["output_tokens"] or 0 for r in calls),
                     "cost": None, "cost_note": "Provider billing is not inferred from the model name"}
    result["complete"] = all(result["complete_by_method"][m] == len(manifest["entries"]) for m in methods)
    save(out / "summary.json", result)
    csv_file(out / "comparison.csv", rows)
    csv_file(out / "per_parcel.csv", parcels)
    csv_file(out / "API_calls.csv", calls)
    lines = ["# FieldAgent 边界减点对照", "", f"预期图像：{len(manifest['entries'])}；全组完成：{result['complete']}。", "",
             "本地自适应组不调用GPT。GT只用于已保存输出的离线评价；未完成GPT图像不计入GPT配对表。", ""]
    for name, group in result["paired_groups"].items():
        lines += [f"## {name}：{group['images']}张共同完成图像", "",
                  "|方法|总点数|减点率|平均点数|中位数|P90|BF1宏平均|Chamfer/px|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        def fmt(value):
            return "—" if value is None else f"{value:.4f}"
        for method, metrics in group["aggregate"].get("all", {}).items():
            reduction = "—" if metrics["reduction"] is None else f"{metrics['reduction']:.2%}"
            lines.append(f"|{method}|{metrics['vertices_after']}|{reduction}|{fmt(metrics['mean_vertices'])}|{fmt(metrics['median_vertices'])}|{fmt(metrics['p90_vertices'])}|{fmt(metrics['boundary_f1_macro'])}|{fmt(metrics['chamfer_macro_px'])}|")
        lines.append("")
    lines += ["4～8个点仅为简单形状的尝试目标。门限内减点不等于提高真实边界精度。", "",
              "GPT完成计数表示该图的既定调用预算执行并导出；未决目标数另见summary.json。失败调用也消耗原预算，不自动重试。", "",
              "Chamfer/P95在任一侧没有边界时记为null，报告有效图数；图像切片不等于独立地理场景。", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return result
