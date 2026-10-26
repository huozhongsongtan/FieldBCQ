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
from quality_scoring import set_quality_scoring




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
        if mode not in {"none", "dice", "bce", "dice_bce"}:
            raise ValueError(f"Unsupported BAS-MQC boundary_mode: {mode}")
        gain = args.get("boundary", 0.2) if isinstance(args, dict) else getattr(args, "boundary", 0.2)
        if mode == "none" or float(gain) == 0:
            return "rbqs"
        return {"dice": "bds_rbqs", "bce": "baps_rbqs", "dice_bce": "fieldbcq"}[mode]
    if name == "BASSegment":
        return "baps"
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


def benchmark(weights, imgsz=640, device="0", image=None, quality_mode=None,
              expected=None, warmup=50, iterations=200):
    """Table 10 only: FP32, batch one, 2*THOP MACs, synchronized mean latencies."""
    import copy
    import numpy as np
    import torch
    from thop import profile
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Table 10 requires a decoded real BGR image")
    if warmup < 0 or iterations < 1:
        raise ValueError("Invalid benchmark iteration count")
    selected = select_device(device, verbose=False)
    wrapper = YOLO(str(weights))
    if expected is not None and configuration_of(wrapper.model) != expected:
        raise ValueError(f"Wrong benchmark checkpoint configuration: {weights}")
    network = wrapper.model.to(selected).float().eval()
    scoring = set_quality_scoring(network, quality_mode)
    parameters = sum(p.numel() for p in network.parameters())
    sample = torch.zeros(1, 3, imgsz, imgsz, device=selected, dtype=torch.float32)
    profiled = copy.deepcopy(network)
    with torch.inference_mode():
        macs, _ = profile(profiled, inputs=(sample,), verbose=False)
    gflops = 2 * macs / 1e9
    del profiled
    if not np.isfinite(gflops) or gflops <= 0:
        raise RuntimeError("THOP did not produce valid GFLOPs")

    def sync():
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)

    def measure(operation):
        with torch.inference_mode():
            for _ in range(warmup):
                operation()
            sync()
            durations = []
            for _ in range(iterations):
                sync()
                start = time.perf_counter()
                result = operation()
                sync()
                durations.append((time.perf_counter() - start) * 1000)
                del result
        return float(np.mean(durations))

    raw_ms = measure(lambda: network(sample))
    # Decode and initialize before timing; square input and batch one match the paper.
    options = dict(source=[image], imgsz=imgsz, batch=1, device=str(selected), half=False,
                   rect=False, conf=0.001, iou=0.70, max_det=710, verbose=False, save=False)
    wrapper.predict(**options)
    pipeline_ms = measure(lambda: wrapper.predict(**options))
    return dict(parameters_total=parameters, gflops_per_image=gflops, raw_forward_ms=raw_ms,
                prediction_pipeline_ms=pipeline_ms, device=str(selected), precision="fp32",
                batch=1, imgsz=imgsz, warmup=warmup, iterations=iterations,
                cuda_synchronized=selected.type == "cuda", flops_definition="2 * THOP MACs; unfused eval graph",
                raw_scope="resident FP32 tensor -> unfused eval forward, excludes preprocessing and NMS",
                pipeline_scope="decoded BGR image -> preprocessing, predictor forward, NMS, masks; excludes disk and metrics",
                **scoring)
