"""Bounded image-level parallel continuation of the frozen legacy GPT experiment.

Workers own disjoint scenes; only the coordinator writes reports/failure flags or
deduplicates shared image blobs. No retry, prompt, gate or target budget changes.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def batch_lock(path):
    """An OS lock releases even after a crash; never blindly delete stale files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another fast runner owns this output folder") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def active_legacy_pids(output):
    import psutil
    found = []
    for process in psutil.process_iter(["pid", "name"]):
        if process.pid == os.getpid() or not (process.info["name"] or "").lower().startswith("python"):
            continue
        try:
            args = process.cmdline()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            raise RuntimeError("Cannot inspect a Python process; cannot exclude duplicate paid work") from None
        if not any(Path(arg).name in {"resume_gpt_testset.py", "run_full_testset.py"} for arg in args):
            continue
        for index, arg in enumerate(args[:-1]):
            if arg == "--output" and Path(args[index + 1]).resolve() == Path(output).resolve():
                found.append(process.pid)
    return found


def validate(data, out, legacy):
    parent, frozen = load(data / "manifest.json"), load(out / "manifest.json")
    entries = parent["entries"]
    if (len(entries) != 2000 or frozen["expected_images"] != 2000
            or frozen["parent_manifest_sha256"] != digest(data / "manifest.json")
            or frozen["scenes"] != [entry["scene"] for entry in entries]):
        raise ValueError("Frozen full-test manifest mismatch")
    for path, expected in {**parent["source_sha256"], **frozen["source_sha256"]}.items():
        if digest(path) != expected:
            raise ValueError("Frozen source/prompt changed: " + path)
    if digest(legacy / "run_full_testset.py") != frozen["source_sha256"][str(legacy / "run_full_testset.py")]:
        raise ValueError("Legacy runner differs from frozen sources")
    settings_path = legacy.parent.parent / "ultralytics-yolo11-main/experiments/geometry_direct_edit_planner.json"
    if load(settings_path) != frozen["planner"]:
        raise ValueError("Frozen provider/model settings changed")
    if (out / "runtime_failure.json").exists():
        raise RuntimeError("Unreconciled runtime_failure.json; inspect before paid continuation")
    return entries, frozen


def remaining_targets(prior, calls):
    counts = prior["per_target_calls"]
    if any(type(count) is not int or not 0 <= count <= calls for count in counts.values()):
        raise ValueError("Invalid saved per-target attempt budget")
    return [uid for uid in prior["unfinished_target_ids"] if counts.get(uid, 0) < calls]


def preflight_checkpoints(entries, out):
    for entry in entries:
        folder = out / "scenes" / entry["scene"]
        if (folder / "completed.json").exists():
            continue
        run = folder / "run"
        if run.exists() and any(run.iterdir()) and not (run / "trace.json").exists():
            raise RuntimeError("Paid scene lacks a reconciled trace; refusing replay: " + entry["scene"])


def initialize(legacy, settings, calls):
    """Run once per process: monkeypatched executors cannot cross image workers."""
    global _CTX
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.path.insert(0, str(legacy))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("YOLO_CONFIG_DIR", "E:/field/.fieldbqc-config")
    from native_runtime import preflight
    native = preflight()
    import run_full_testset as full
    from resume_gpt_testset import continue_scene
    import cvr_agent  # Activates the frozen legacy host dependency directory.
    from run_geometry_experiments import credentials_from_environment
    import cv2
    cv2.setNumThreads(1)
    credentials_from_environment()
    if not os.environ.get(settings.get("api_key_env", "OPENAI_API_KEY")):
        raise RuntimeError("Configured API key is unavailable")
    ba, allowed = full.boundary_only_capabilities()
    _CTX = full, ba, allowed, continue_scene, settings, calls, native


def process_scene(job):
    data, out, entry = job
    data, out = Path(data), Path(out)
    scene = entry["scene"]
    folder, run = out / "scenes" / scene, out / "scenes" / scene / "run"
    started = time.monotonic()
    full, ba, allowed, continue_scene, settings, calls, native = _CTX
    try:
        from boundary_executor import BoundaryConfig
        from cvr_agent import read_items, vertices
        from run_boundary_tools import load_image
        from run_testset_experiment import offline_evaluate
        if (folder / "completed.json").exists():
            return {"scene": scene, "status": "completed", "skipped": True}
        save(folder / "parallel_worker.json", {"pid": os.getpid(), "started_utc": now(), "native_runtime": native})
        pred = data / "predictions" / scene
        if (digest(entry["image"]) != entry["image_sha256"]
                or digest(pred / "prediction.pixel.geojson") != load(pred / "completed.json")["vector_sha256"]):
            raise ValueError("Frozen image or prediction changed")
        image = load_image(entry["image"])
        items = read_items(pred / "prediction.pixel.geojson")
        config = BoundaryConfig(budget=max(10000, len(items) * calls),
            seconds=max(7200., len(items) * calls * settings.get("timeout_seconds", 90) + 300.))
        if (run / "trace.json").exists():
            trace = load(run / "trace.json")
            if trace.get("gpt_error"):
                if remaining_targets(trace, calls):
                    trace = continue_scene(ba, image, items, run, config, settings, calls)
                else:
                    snapshot = run / "recovery_snapshots" / str(time.time_ns())
                    snapshot.mkdir(parents=True)
                    shutil.copyfile(run / "trace.json", snapshot / "trace.json")
                    trace = dict(trace, gpt_error=None, stop_reason="original_attempt_budget_exhausted",
                        recovered_external_errors=[*trace.get("recovered_external_errors", []),
                            {"error": trace["gpt_error"], "retained_snapshot": str(snapshot / "trace.json")}])
                    save(run / "trace.json", trace)
        elif run.exists() and any(run.iterdir()):
            raise RuntimeError("Interrupted paid scene lacks a reconciled trace")
        elif items:
            trace = ba.run_autonomous(image, items, run, config, settings,
                max_calls=calls * len(items), calls_per_target=calls,
                allow_direct=True, goal_mode="minimize_vertices")
        else:
            try:
                trace = full.empty_prediction_call(image, run / "gpt_calls/001", settings)
            except Exception as error:
                return {"scene": scene, "status": "external_error", "error": str(error)}
            save(run / "trace.json", trace)
        if trace.get("gpt_error"):
            return {"scene": scene, "status": "external_error", "error": trace["gpt_error"]}
        if items and any(trace["per_target_calls"].get(parcel.identifier, 0) < 1 for parcel in items):
            raise ValueError("Some frozen targets were never visited")
        final = read_items(run / "final/prediction.repaired.pixel.geojson") if items else []
        # First GT access is AFTER the complete committed output has been saved.
        if digest(entry["label"]) != entry["label_sha256"]:
            raise ValueError("Offline reference changed")
        raw_eval = offline_evaluate(items, items, entry["label"], image.shape[:2])
        final_eval = offline_evaluate(final, items, entry["label"], image.shape[:2])
        lookup = {parcel.identifier: parcel for parcel in final}
        with (folder / "per_parcel.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["instance_id", "raw_vertices", "gpt_vertices"])
            for parcel in items:
                writer.writerow([parcel.identifier, vertices(parcel.geometry), vertices(lookup[parcel.identifier].geometry)])
        if (run / "tool_catalog.json").exists():
            catalog = load(run / "tool_catalog.json")
            catalog["tools"] = {key: value for key, value in catalog["tools"].items() if key in allowed}
            save(run / "tool_catalog.json", catalog)
        save(folder / "completed.json", {"scene": scene, "domain": entry["domain"],
            "raw": raw_eval, "gpt_autonomous": final_eval, "runtime_gt_access": False,
            **{key: trace[key] for key in ("model_calls", "commits", "direct_proposals", "direct_committed", "unfinished_target_ids")}})
        return {"scene": scene, "status": "completed", "model_calls": trace["model_calls"],
                "commits": trace["commits"], "seconds": time.monotonic() - started}
    except Exception:
        error = traceback.format_exc()
        save(folder / "parallel_runtime_failure.json", {"error": error, "updated_utc": now()})
        return {"scene": scene, "status": "runtime_error", "error": error}


def schedule(jobs, workers, *, worker=process_scene, initializer=None, initargs=(), on_result=None, should_stop=None):
    """At most workers paid scenes in flight; stop dispatch on first failure.

    SIGINT requests a drain rather than killing a paid request mid-checkpoint.
    A whole completed wave is inspected before submitting any more work.
    """
    pending, results, stop = {}, [], False
    callback_error = None
    iterator = iter(jobs)
    previous = signal.getsignal(signal.SIGINT)

    def interrupt(signum, frame):
        nonlocal stop
        stop = True
        print("Stop requested: saving in-flight scenes; please wait.", flush=True)

    signal.signal(signal.SIGINT, interrupt)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initializer, initargs=initargs) as pool:
            def fill():
                nonlocal stop
                if should_stop and should_stop():
                    stop = True
                while not stop and len(pending) < workers:
                    job = next(iterator, None)
                    if job is None:
                        break
                    pending[pool.submit(worker, job)] = job

            fill()
            while pending:
                done, _ = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
                if should_stop and should_stop():
                    stop = True
                wave = []
                for future in done:
                    job = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception:
                        result = {"status": "runtime_error", "error": traceback.format_exc(),
                                  "scene": job[2]["scene"] if isinstance(job, tuple) and len(job) == 3 else str(job)}
                    if result["status"] != "completed":
                        stop = True
                    wave.append(result)
                for result in wave:
                    results.append(result)
                    if on_result:
                        try:
                            on_result(result)
                        except Exception as error:
                            # A disk/report failure must also stop dispatch while
                            # allowing every already-paid scene to checkpoint.
                            stop = True
                            callback_error = error
                fill()
        if callback_error:
            raise RuntimeError("Coordinator failed after draining active scenes") from callback_error
        return results, stop
    finally:
        signal.signal(signal.SIGINT, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-code", type=Path, default=Path("E:/field/agent-go/vertex_only"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--resume-after-external-error", action="store_true")
    parser.add_argument("--wait-for-existing", action="store_true", help="Wait for the old runner to exit; never kill it")
    parser.add_argument("--check-only", action="store_true", help="Read-only validation; zero API requests")
    args = parser.parse_args(argv)
    data, out, legacy = args.data.resolve(), args.output.resolve(), args.legacy_code.resolve()
    entries, frozen = validate(data, out, legacy)
    existing = active_legacy_pids(out)
    complete = sum((out / "scenes" / entry["scene"] / "completed.json").exists() for entry in entries)
    if args.check_only:
        if not existing:
            preflight_checkpoints(entries, out)
        print(json.dumps({"frozen_protocol_verified": True, "completed": complete, "expected": len(entries),
            "workers": args.workers, "active_legacy_pids": existing, "API_requests": 0,
            "checkpoint_scan": "passed" if not existing else "deferred_until_old_runner_exits",
            "model": frozen["planner"]["model"], "calls_per_target": frozen["calls_per_target"]}, indent=2))
        return 0
    with batch_lock(out / "parallel_control/batch.lock"):
        while existing:
            if not args.wait_for_existing:
                raise RuntimeError(f"Old runner still active (PIDs {existing}); use --wait-for-existing")
            print(f"Waiting for old runner {existing}; no duplicate requests will be sent.", flush=True)
            time.sleep(10)
            existing = active_legacy_pids(out)
        # Recheck after waiting: another runner may have produced failure/checkpoints.
        entries, frozen = validate(data, out, legacy)
        preflight_checkpoints(entries, out)
        failure = out / "external_failure.json"
        if failure.exists() and not args.resume_after_external_error:
            raise RuntimeError("Use --resume-after-external-error for explicit continuation")
        stop_file = out / "parallel_stop.request"
        if stop_file.exists():
            raise RuntimeError("parallel_stop.request exists; remove it to resume")
        jobs = [(str(data), str(out), entry) for entry in entries
                if not (out / "scenes" / entry["scene"] / "completed.json").exists()]
        # Native runtime and credentials checked before changing active failure flags.
        sys.path.insert(0, str(legacy))
        from native_runtime import configure
        configure()
        import run_full_testset as full
        import cvr_agent  # Same import order as the original continuation entry.
        from run_geometry_experiments import credentials_from_environment
        credentials_from_environment()
        if jobs and not os.environ.get(frozen["planner"].get("api_key_env", "OPENAI_API_KEY")):
            raise RuntimeError("Configured API key unavailable; set it in the terminal")
        session = out / "parallel_sessions" / str(time.time_ns())
        session.mkdir(parents=True)
        shutil.copyfile(__file__, session / "fast_gpt.py")
        save(session / "manifest.json", {"created_utc": now(), "pid": os.getpid(), "workers": args.workers,
            "base_experiment_manifest_sha256": digest(out / "manifest.json"), "harness_sha256": digest(__file__),
            "continuation_helper_sha256": digest(legacy / "resume_gpt_testset.py"),
            "archiver_sha256": digest(legacy / "archive_call_artifacts.py"),
            "scientific_protocol_changes": False, "automatic_API_retries": 0,
            "execution_change": "parallel independent scenes; within-scene order unchanged",
            "failure_policy": "stop dispatch, drain active scenes, then preserve all failures",
            "wall_time_comparable_to_serial": False})
        for name in ("resume_gpt_testset.py", "archive_call_artifacts.py", "native_runtime.py"):
            shutil.copyfile(legacy / name, session / name)
        if failure.exists():
            failure.replace(session / "prior_external_failure.json")
        failures = []
        completed = len(entries) - len(jobs)
        from archive_call_artifacts import archive_call

        def progress(result):
            nonlocal completed
            scene = result["scene"]
            # Only this coordinator touches shared content-addressed blobs/reports.
            for request in (out / "scenes" / scene / "run/gpt_calls").glob("*/request.json"):
                archive_call(request.parent, out)
            save(session / (scene + ".json"), result)
            if result["status"] == "completed":
                completed += 1
            else:
                failures.append(result)
                marker = "runtime_failure.json" if any(r["status"] == "runtime_error" for r in failures) else "external_failure.json"
                save(out / marker, {**failures[0], "failures": failures, "updated_utc": now(), "recovery_session": str(session)})
            full.report(data, out, entries)
            save(out / "progress.json", {"status": "draining_after_failure" if failures else "running_actual_GPT_parallel",
                "images_completed": completed, "expected_images": len(entries), "workers": args.workers,
                "last_scene": scene, "updated_utc": now(), "recovery_session": str(session)})
            print(f"GPT {completed}/{len(entries)} {scene} {result['status']} calls={result.get('model_calls', '-')}", flush=True)

        print(f"GPT parallel start: completed={completed}/{len(entries)}, workers={args.workers}", flush=True)
        try:
            results, stopped = schedule(jobs, args.workers, initializer=initialize,
                initargs=(str(legacy), frozen["planner"], frozen["calls_per_target"]),
                on_result=progress, should_stop=stop_file.exists)
        except Exception:
            save(out / "runtime_failure.json", {"scene": "parallel_coordinator", "error": traceback.format_exc(),
                "updated_utc": now(), "recovery_session": str(session)})
            save(out / "progress.json", {"status": "stopped_runtime_error", "images_completed": completed,
                "expected_images": len(entries), "updated_utc": now()})
            raise
        full.report(data, out, entries)
        if completed == len(entries) and not failures:
            summary = load(out / "summary.json")
            save(out / "completed.json", {"images": len(entries), "every_image_actual_GPT": True,
                "all_targets_decided": summary["images_with_unfinished_targets"] == 0, "updated_utc": now()})
        status = ("stopped_runtime_error" if any(r["status"] == "runtime_error" for r in failures)
                  else "stopped_external_error" if failures else "stopped_by_user" if stopped else "all_images_processed")
        save(out / "progress.json", {"status": status, "images_completed": completed,
            "expected_images": len(entries), "workers": args.workers, "updated_utc": now()})
        return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
