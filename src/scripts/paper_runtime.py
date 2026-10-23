"""Run provenance and explicitly scoped inference benchmarks for paper reports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from project_paths import PROJECT_ROOT


EVALUATOR_VERSION = "fieldbcq-paper-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def code_provenance():
    files = {}
    for folder in ("src/scripts", "src/ultralytics"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml"}:
                files[path.relative_to(PROJECT_ROOT).as_posix()] = sha256(path)
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"git_commit": revision, "git_dirty": dirty, "source_sha256": fingerprint, "files": files}


def environment():
    import torch
    import ultralytics
    versions = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "scipy", "ultralytics-thop"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info().get("brand_raw")
    except ImportError:
        cpu = None
    devices = []
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": prop.name, "total_memory_bytes": prop.total_memory})
    return {
        "utc_time": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "cpu": cpu or platform.processor(), "logical_cpu_count": os.cpu_count(),
        "packages": versions, "ultralytics": ultralytics.__version__,
        "ultralytics_file": ultralytics.__file__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpus": devices,
        "torch_threads": torch.get_num_threads(),
    }


def configuration_of(model):
    head = model.model[-1]
    name = type(head).__name__
    args = getattr(model, "args", {})
    mode = args.get("boundary_mode", "dice_bce") if isinstance(args, dict) else getattr(args, "boundary_mode", "dice_bce")
    if name == "BASMQCSegment":
        return "fieldbcq"
    if name == "Segment_BAS":
        return "baps" if mode == "bce" else "baps_pcsc"
    if name == "BoundaryDiceSegment":
        return "pcsc"
    return "baseline" if name == "Segment" else name


def attach_training_metadata(model, stage, cli_args, train_args):
    """Save actual resolved trainer args, seed, code, optimizer and checkpoint hashes."""
    report = {}

    def start(trainer):
        import torch
        from ultralytics.utils.torch_utils import de_parallel
        if getattr(trainer, "rank", -1) not in (-1, 0) or int(os.environ.get("RANK", -1)) not in (-1, 0):
            return
        resolved = vars(trainer.args)
        actual_stage = configuration_of(de_parallel(trainer.model))
        if actual_stage != stage:
            raise ValueError(f"Requested stage {stage} but actual model is {actual_stage}; check resume checkpoint.")
        report.update({
            "configuration": stage, "cli_args": vars(cli_args), "resolved_args": resolved,
            "requested_train_args": train_args, "seed": trainer.args.seed,
            "resume": trainer.args.resume, "environment": environment(), "code": code_provenance(),
            "optimizer_class": type(trainer.optimizer).__name__,
            "optimizer_groups": [{k: v for k, v in group.items() if k != "params"} for group in trainer.optimizer.param_groups],
            "amp_actually_enabled": bool(trainer.amp),
        })
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        write_json(Path(trainer.save_dir) / "run_metadata.json", report)

    def end(trainer):
        import torch
        if not report:
            return
        report["checkpoints"] = {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in (("best", trainer.best), ("last", trainer.last)) if Path(path).is_file()
        }
        if trainer.device.type == "cuda":
            report["training_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(trainer.device)
            report["training_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(trainer.device)
        report["completed_epochs"] = trainer.epoch + 1
        write_json(Path(trainer.save_dir) / "run_metadata.json", report)

    model.add_callback("on_train_start", start)
    model.add_callback("on_train_end", end)


def benchmark(weights, imgsz=640, batch=1, device="cpu", half=False, warmup=20, iterations=100,
              conf=0.001, iou=0.70, max_det=710, images=None, quality_blend=None):
    """Measure raw forward and decoded-image prediction separately; no disk I/O in timing."""
    import copy
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device

    selected = select_device(device, verbose=False)
    if half and selected.type != "cuda":
        raise ValueError("FP16 benchmark requires a CUDA device; use fp32 for CPU.")
    wrapper = YOLO(str(weights))
    network = wrapper.model.to(selected).eval()
    if quality_blend is not None:
        head = network.model[-1]
        if not hasattr(head, "quality_blend"):
            raise ValueError("--quality-blend is only valid for a quality-head checkpoint.")
        head.quality_blend = quality_blend
    parameters = sum(p.numel() for p in network.parameters())
    profile_error, gflops = None, None
    profiled = sample = None
    try:
        from thop import profile
        profiled = copy.deepcopy(network).float()
        sample = torch.zeros(1, 3, imgsz, imgsz, device=selected)
        with torch.inference_mode():
            macs, _ = profile(profiled, inputs=(sample,), verbose=False)
        gflops = 2 * macs / 1e9
    except (ImportError, RuntimeError, TypeError, AttributeError, ValueError, KeyError) as exc:
        profile_error = str(exc)
    finally:
        del profiled, sample
        if selected.type == "cuda":
            torch.cuda.empty_cache()
    network.half() if half else network.float()
    tensor = torch.zeros(batch, 3, imgsz, imgsz, device=selected, dtype=torch.float16 if half else torch.float32)

    def sync():
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)

    def measure(operation):
        with torch.inference_mode():
            for _ in range(warmup):
                operation()
            sync()
            if selected.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected)
            durations = []
            for _ in range(iterations):
                sync()
                start = time.perf_counter()
                result = operation()
                sync()
                durations.append(time.perf_counter() - start)
                del result
        mean = float(np.mean(durations))
        return {
            "batch_latency_ms_mean": 1000 * mean,
            "batch_latency_ms_p50": 1000 * float(np.percentile(durations, 50)),
            "batch_latency_ms_p95": 1000 * float(np.percentile(durations, 95)),
            "amortized_ms_per_image": 1000 * mean / batch, "images_per_second": batch / mean,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(selected) if selected.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(selected) if selected.type == "cuda" else None,
        }

    raw = measure(lambda: network(tensor))
    del tensor
    if images is not None:
        decoded = [images[index % len(images)] for index in range(batch)]
        image_kind = "first decoded evaluation images (cycled if fewer than batch)"
    else:
        decoded = [np.zeros((imgsz, imgsz, 3), dtype=np.uint8) for _ in range(batch)]
        image_kind = "blank synthetic images; postprocessing cost is not representative"
    options = dict(source=decoded, imgsz=imgsz, batch=batch, device=str(selected), half=half,
                   rect=False, conf=conf, iou=iou, max_det=max_det, verbose=False, save=False)
    wrapper.predict(**options)  # initialize predictor outside timed scope
    pipeline = measure(lambda: wrapper.predict(**options))
    return {
        "parameters_total": parameters, "gflops_per_image": gflops, "flops_definition": "2 * THOP MACs",
        "flops_profile_error": profile_error, "batch": batch, "imgsz": imgsz,
        "precision": "fp16" if half else "fp32", "device": str(selected),
        "warmup_batches": warmup, "timed_batches": iterations, "cuda_synchronized": selected.type == "cuda",
        "raw_forward": raw, "decoded_image_pipeline": pipeline, "pipeline_inputs": image_kind,
        "raw_scope": "resident tensor -> unfused eval model forward including quality fusion; excludes preprocessing/NMS",
        "pipeline_scope": "decoded BGR images -> preprocessing, model (predictor may fuse layers), NMS, masks, Results; excludes disk I/O and metrics",
        "conf": conf, "nms_iou": iou, "max_det": max_det, "quality_blend": quality_blend,
        "cpu_peak_memory": "not measured; CUDA memory values are null on CPU",
    }
