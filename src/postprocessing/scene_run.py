"""New-version whole-image experiment CLI; never reuses old GPT outputs."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from fieldagent import activate_engine, PROJECT_ROOT, PACKAGE_ROOT
from fieldagent.io import digest, load, save, now, read_inputs, planner_config
import scene_batch


def sources():
    from fieldagent.experiment import source_hashes
    result = {"fieldagent/" + k: (PACKAGE_ROOT / k) for k in source_hashes()}
    for name in ("scene_batch.py", "scene_run.py", "SCENE_BATCH_PROMPT.md", "fast_gpt.py", "scripts/run_gpt_scene.py"):
        result["scene_mode/" + name] = Path(__file__).resolve().parent / name
    return result


def freeze(inputs, out, settings, calls, controls_cache=None):
    from run_testset_experiment import CONFIGS
    paths = sources()
    entries = read_inputs(inputs)
    record = {"version": scene_batch.VERSION, "scope": "boundary_vertex_reduction_only",
        "protocol": "whole_image_batch", "entries": entries, "inputs_sha256": digest(inputs),
        "methods": ["raw", "public_edge", "local_adaptive_strict", "gpt_autonomous"],
        "max_calls_per_scene": calls, "planner": settings, "runtime_GT_access": False,
        "local_configs": {m: asdict(CONFIGS[m]) for m in ("public_edge", "local_adaptive_strict")},
        "gpt_config": asdict(CONFIGS["local_adaptive_strict"]),
        "source_sha256": {name: digest(path) for name, path in paths.items()}, "automatic_API_retries": 0,
        "controls_cache": ({"path": str(controls_cache.resolve()),
                            "manifest_sha256": digest(controls_cache / "manifest.json")} if controls_cache else None)}
    if (out / "manifest.json").exists():
        prior = load(out / "manifest.json")
        if {k: v for k, v in prior.items() if k != "created_utc"} != record:
            raise ValueError("Frozen whole-image inputs/code/settings changed; use a new output directory")
        return prior
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Whole-image mode needs a new empty output folder; do not use the old GPT folder")
    for name, path in paths.items():
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    record["created_utc"] = now()
    save(out / "manifest.json", record)
    return record


def import_controls(manifest, out):
    """Reuse already-completed identical local controls, with input/config hashes."""
    from fieldagent.experiment import read_scene, completed_method, persist_method
    from cvr_agent import read_items
    cache_record = manifest["controls_cache"]
    if not cache_record:
        return 0
    cache = Path(cache_record["path"])
    if digest(cache / "manifest.json") != cache_record["manifest_sha256"]:
        raise ValueError("Local control cache manifest changed")
    parent = load(cache / "manifest.json")
    for method in ("public_edge", "local_adaptive_strict"):
        if parent["methods"][method] != manifest["local_configs"][method]:
            raise ValueError("Cached local controller has different quality settings")
    originals = {e["scene"]: e for e in parent["entries"]}
    count = 0
    for entry in manifest["entries"]:
        folder = cache / "scenes" / entry["scene"]
        if not (folder / "completed.json").exists():
            continue
        if all(completed_method(out, entry, m) for m in ("raw", "public_edge", "local_adaptive_strict")):
            continue
        original = originals[entry["scene"]]
        pred = cache / "predictions" / entry["scene"]
        if (entry["image_sha256"] != original["image_sha256"]
                or entry["prediction_vector_sha256"] != load(pred / "completed.json")["vector_sha256"]
                or digest(pred / "prediction.pixel.geojson") != entry["prediction_vector_sha256"]
                or (entry.get("label_sha256") and entry["label_sha256"] != original["label_sha256"])):
            raise ValueError("Cached control inputs differ: " + entry["scene"])
        image, raw = read_scene(entry)
        source = load(folder / "completed.json")
        for method in ("raw", "public_edge", "local_adaptive_strict"):
            if completed_method(out, entry, method):
                continue
            if method == "raw":
                items, trace = raw, {}
                vector_source = Path(entry["prediction_vector"])
            else:
                vector_source = folder / method / "final.pixel.geojson"
                if not vector_source.exists():
                    raise ValueError("Missing completed local control vector")
                items = read_items(vector_source)
                trace = load(folder / method / "trace.json")
            target = out / "scenes" / entry["scene"] / method
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(vector_source, target / "final.pixel.geojson")
            # Cached GT evaluation is reused only after saving the corresponding geometry.
            repair_entry = {k: v for k, v in entry.items() if k not in {"label", "label_sha256"}}
            record = persist_method(out, repair_entry, method, items, raw, image, trace)
            if entry.get("label"):
                if digest(entry["label"]) != entry["label_sha256"]:
                    raise ValueError("Offline reference changed")
                record["evaluation"] = source["evaluation"][method]
            record["imported_control"] = {"source_vector": str(vector_source), "source_vector_sha256": digest(vector_source),
                "source_completed": str(folder / "completed.json"), "source_completed_sha256": digest(folder / "completed.json")}
            save(target / "completed.json", record)
            save(target / "trace.json", trace)
        count += 1
    return count


def initialize_worker():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    activate_engine()
    import cv2
    cv2.setNumThreads(1)
    from run_geometry_experiments import credentials_from_environment
    credentials_from_environment()


def build_report(manifest, out):
    from fieldagent.experiment import build_report as base_report, csv_file
    result = base_report(manifest, out)
    rows = []
    for entry in manifest["entries"]:
        folder = out / "scenes" / entry["scene"] / "gpt_autonomous"
        if not (folder / "run/trace.json").exists():
            continue
        trace = load(folder / "run/trace.json")
        rows.append({"scene": entry["scene"], "domain": entry["domain"], "completed": (folder / "completed.json").exists(),
            "model_calls": trace["model_calls"], "batch_commits": trace.get("batch_commits", 0),
            "parcel_commits": trace["commits"], "unfinished_targets": len(trace["unfinished_target_ids"]),
            "empty_inspection_confirmed": trace.get("empty_inspection_confirmed")})
    completed = [r for r in rows if r["completed"]]
    result.update(protocol="whole_image_batch", max_calls_per_scene=manifest["max_calls_per_scene"],
        whole_image_efficiency={"completed_images": len(completed),
            "calls_on_completed_images": sum(r["model_calls"] for r in completed),
            "mean_calls_per_completed_image": sum(r["model_calls"] for r in completed) / len(completed) if completed else None,
            "batch_commits": sum(r["batch_commits"] for r in completed),
            "API_speedup_vs_old_protocol_measured": False,
            "note": "Call counts alone do not measure API latency or total wall time speedup"})
    save(out / "summary.json", result)
    csv_file(out / "SCENE_API.csv", rows)
    report = out / "RESULTS.md"
    report.write_text("# 整图批量GPT实验（独立版本）\n\n"
        + f"每张图最多 {manifest['max_calls_per_scene']} 次请求；一次观察覆盖全部预测田块，联合候选需后续显式确认。\n\n"
        + "本结果不与原逐田块GPT混合。整图请求次数和联合提交次数见 SCENE_API.csv；真实API耗时加速比尚未测量。\n\n"
        + report.read_text(encoding="utf-8"), encoding="utf-8")
    return result


def process_image(job):
    payload, output, entry = job
    manifest, resume = payload["config"], payload["resume_after_error"]
    out = Path(output)
    from fieldagent.experiment import completed_method, read_scene, persist_method
    from boundary_executor import BoundaryConfig
    from cvr_agent import read_items
    scene = entry["scene"]
    try:
        if completed_method(out, entry, "gpt_autonomous"):
            return {"scene": scene, "status": "completed", "skipped": True}
        image, raw = read_scene(entry)
        folder = out / "scenes" / scene / "gpt_autonomous"
        cfg = dict(manifest["gpt_config"])
        cfg.update(seconds=max(7200., manifest["max_calls_per_scene"] * manifest["planner"].get("timeout_seconds", 90) + 300.))
        trace = scene_batch.run_scene(image, raw, folder / "run", BoundaryConfig(**cfg), manifest["planner"],
            max_calls=manifest["max_calls_per_scene"], resume_after_error=resume)
        if trace.get("gpt_error"):
            return {"scene": scene, "status": "external_error", "error": trace["gpt_error"], "model_calls": trace["model_calls"]}
        items = read_items(folder / "run/final/prediction.repaired.pixel.geojson") if raw else []
        record = persist_method(out, entry, "gpt_autonomous", items, raw, image, trace)
        record.update(protocol="whole_image_batch", batch_commits=trace.get("batch_commits", 0),
                      empty_inspection_confirmed=trace.get("empty_inspection_confirmed"))
        save(folder / "completed.json", record)
        # Offline reference access occurs only after the GPT output is persisted.
        if not completed_method(out, entry, "raw"):
            persist_method(out, entry, "raw", raw, raw, image)
        return {"scene": scene, "status": "completed", "model_calls": trace["model_calls"],
                "commits": trace["commits"], "batch_commits": trace.get("batch_commits", 0)}
    except Exception:
        return {"scene": scene, "status": "runtime_error", "error": traceback.format_exc()}


def run_gpt(manifest, out, workers, resume_after_error):
    from fast_gpt import schedule, batch_lock
    from fieldagent.experiment import completed_method
    from fieldagent.archive import archive_call
    if (out / "runtime_failure.json").exists():
        raise RuntimeError("Inspect runtime_failure.json before any paid continuation")
    with batch_lock(out / "scene_gpt.lock"):
        failure = out / "external_failure.json"
        if failure.exists():
            if not resume_after_error:
                raise RuntimeError("Explicit --resume-after-error required")
            previous = out / "recovery_sessions" / str(time.time_ns())
            previous.mkdir(parents=True)
            failure.rename(previous / "prior_external_failure.json")
        settings = manifest["planner"]
        if not os.environ.get(settings.get("api_key_env", "OPENAI_API_KEY")):
            raise RuntimeError("Set the configured API key in this terminal")
        worker_config = {k: manifest[k] for k in ("gpt_config", "planner", "max_calls_per_scene")}
        payload = {"config": worker_config, "resume_after_error": resume_after_error}
        jobs = [(payload, str(out), e) for e in manifest["entries"]
                if not completed_method(out, e, "gpt_autonomous")]
        complete = len(manifest["entries"]) - len(jobs)
        session = out / "execution_sessions" / str(time.time_ns())
        session.mkdir(parents=True)
        save(session / "settings.json", {"workers": workers, "created_utc": now(),
             "protocol": "whole_image_batch", "max_calls_per_scene": manifest["max_calls_per_scene"],
             "automatic_API_retries": 0, "resume_after_error": resume_after_error})
        failures = []

        def progress(result):
            nonlocal complete
            if result["status"] == "completed":
                complete += 1
            else:
                failures.append(result)
                filename = "runtime_failure.json" if any(r["status"] == "runtime_error" for r in failures) else "external_failure.json"
                save(out / filename, {"failures": failures, "updated_utc": now()})
            for request in (out / "scenes" / result["scene"] / "gpt_autonomous/run/gpt_calls").glob("*/request.json"):
                archive_call(request.parent, out)
            save(session / (result["scene"] + ".json"), result)
            build_report(manifest, out)
            save(out / "progress.json", {"status": "draining_after_failure" if failures else "running_whole_image_GPT",
                 "images_completed": complete, "expected_images": len(manifest["entries"]), "workers": workers,
                 "last_scene": result["scene"], "updated_utc": now()})
            print(f"SCENE GPT {complete}/{len(manifest['entries'])} {result['scene']} {result['status']} calls={result.get('model_calls', '-')}", flush=True)

        try:
            results, stopped = schedule(jobs, workers, worker=process_image, initializer=initialize_worker,
                on_result=progress, should_stop=(out / "scene_stop.request").exists)
        except Exception:
            save(out / "runtime_failure.json", {"error": traceback.format_exc(), "updated_utc": now()})
            raise
        build_report(manifest, out)
        status = "stopped_after_error" if failures else "stopped_by_user" if stopped else "gpt_image_budgeted_pass_complete"
        save(out / "progress.json", {"status": status, "images_completed": complete, "expected_images": len(manifest["entries"]), "updated_utc": now()})
        return 2 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=PROJECT_ROOT / "data/inputs.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner", type=Path, default=PROJECT_ROOT / "data/planner.example.json")
    parser.add_argument("--stage", choices=("local", "gpt", "all", "report", "import-controls"), default="gpt")
    parser.add_argument("--max-calls-per-scene", type=int, default=4)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--controls-cache", type=Path, help="Reuse identical completed local controls from the original E-drive run")
    parser.add_argument("--resume-after-error", action="store_true")
    args = parser.parse_args(argv)
    if not 2 <= args.max_calls_per_scene <= 16:
        parser.error("--max-calls-per-scene must be 2..16; changing it requires a new output directory")
    if args.stage == "import-controls" and not args.controls_cache:
        parser.error("--stage import-controls requires --controls-cache")
    activate_engine()
    import cv2
    cv2.setNumThreads(1)
    from fieldagent.experiment import run_local
    out = args.output.resolve()
    if args.stage == "report":
        manifest = load(out / "manifest.json")
        if manifest.get("protocol") != "whole_image_batch":
            raise ValueError("This report entry accepts whole-image outputs only")
        result = build_report(manifest, out)
        print(json.dumps(result["complete_by_method"]))
        return 0
    settings = planner_config(args.planner)
    manifest = freeze(args.inputs, out, settings, args.max_calls_per_scene, args.controls_cache)
    # One coordinator owns all mutations; even control import/report cannot race.
    from fast_gpt import batch_lock
    with batch_lock(out / "whole_scene_run.lock"):
        if args.controls_cache:
            print("Imported local controls:", import_controls(manifest, out), flush=True)
        if args.stage in {"local", "all"}:
            run_local(manifest, out)
        code = 0
        if args.stage in {"gpt", "all"}:
            from run_geometry_experiments import credentials_from_environment
            credentials_from_environment()
            code = run_gpt(manifest, out, args.workers, args.resume_after_error)
        result = build_report(manifest, out)
    print(json.dumps({"complete_by_method": result["complete_by_method"], "output": str(out)}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
