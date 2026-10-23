"""Single-pass mask evaluation, or a compact summary of existing paper reports."""

import argparse
import copy
import csv
import json
import logging
import math
import platform
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

from quality_scoring import QUALITY_MODES, set_quality_scoring
from project_paths import project_path
from paper_runtime import configuration_of, sha256, write_json


EVALUATOR_VERSION = "fieldbcq-main-v1"
FIELDS = ["model", "seed", "P", "R", "F1", "AP50", "AP75", "AP50_95"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--weights", action="append", metavar="[NAME=]CHECKPOINT",
                        help="Repeat to evaluate multiple models, once each.")
    inputs.add_argument("--report", action="append", help="Extract existing paper_report.json without inference.")
    parser.add_argument("--data", default="data/datasets.yaml")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--conf", type=float, default=0.001, help="Low prediction floor for AP, not a fixed F1 threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=710)
    parser.add_argument("--project", default="results/validation")
    parser.add_argument("--name", help="New output directory name; existing directories are never overwritten.")
    parser.add_argument("--plots", action="store_true", help="Opt in to evaluation plots; disabled by default.")
    scoring = parser.add_mutually_exclusive_group()
    scoring.add_argument("--quality-mode", choices=(*QUALITY_MODES, "all"),
                         help="Runtime scoring only; all evaluates the same checkpoint four times.")
    scoring.add_argument("--quality-blend", type=float, choices=(0.0, 1.0), help="Legacy C0/C2 switch.")
    args = parser.parse_args()
    if args.report and (args.quality_mode is not None or args.quality_blend is not None):
        parser.error("Quality overrides require --weights; existing reports cannot be rescored")
    if min(args.batch, args.imgsz, args.max_det) < 1 or args.imgsz % 32 or args.workers < 0:
        parser.error("batch/max-det must be positive; imgsz a positive multiple of 32; workers nonnegative")
    if not 0 <= args.conf <= 1 or not 0 < args.iou <= 1:
        parser.error("conf must be in [0,1] and iou in (0,1]")
    if args.name and (args.name in {".", ".."} or any(c in args.name for c in "/\\:")):
        parser.error("name must be a directory name without path separators")
    return args


def metric_row(name, seed, mask):
    p, r = float(mask["precision"]), float(mask["recall"])
    values = [p, r, 2 * p * r / (p + r) if p + r else 0.0,
              float(mask["map50"]), float(mask["map75"]), float(mask["map50_95"])]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Non-finite or out-of-range mask metrics: " + name)
    return dict(zip(FIELDS, [name, seed, *values]))


def models_to_run(values):
    models, names = [], set()
    for item in values:
        name, separator, raw = item.partition("=")
        weights = project_path(raw if separator else item)
        if not separator:
            name = weights.parent.parent.name if weights.parent.name == "weights" else weights.parent.name
        if not name or name in {".", ".."} or any(c in name for c in "/\\:") or name in names:
            raise ValueError("Use unique --weights NAME=CHECKPOINT names without path separators")
        if not weights.is_file():
            raise FileNotFoundError(weights)
        names.add(name)
        models.append((name, weights))
    return models


def main_validator_class():
    from ultralytics.models.yolo.segment import SegmentationValidator

    class MainTableValidator(SegmentationValidator):
        def get_desc(self):
            return "Validating masks (one pass)"

        def print_results(self):
            pass  # print the compact combined mask table after all checkpoints

    return MainTableValidator


@contextmanager
def concise_logs():
    from ultralytics.utils import LOGGER

    class WarningsOnly(logging.Filter):
        def filter(self, record):
            return record.levelno >= logging.WARNING

    log_filter = WarningsOnly()
    LOGGER.addFilter(log_filter)
    try:
        yield
    finally:
        LOGGER.removeFilter(log_filter)


def evaluate_one(args, name, weights, output):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.precision == "fp16" and (args.device == "cpu" or not torch.cuda.is_available()):
        raise ValueError("FP16 requires CUDA; use --precision fp32 for CPU")
    model = YOLO(str(weights))
    stored = dict((model.ckpt or {}).get("train_args", {}))
    seed = stored.get("seed")
    configuration = configuration_of(model.model)
    scoring = set_quality_scoring(model.model, args.quality_mode, args.quality_blend)
    checkpoint_hash = sha256(weights)
    options = dict(data=str(project_path(args.data)), split=args.split, imgsz=args.imgsz,
                   batch=args.batch, workers=args.workers, half=args.precision == "fp16",
                   conf=args.conf, iou=args.iou, max_det=args.max_det, rect=True,
                   augment=False, plots=args.plots, save=False, save_json=False,
                   save_txt=False, verbose=False, project=str(output), name=name, exist_ok=True)
    if args.device is not None:
        options["device"] = args.device
    with concise_logs():
        result = model.val(validator=main_validator_class(), **options)  # exactly one dataset pass
    mask = result.seg
    row = metric_row(name, seed, dict(precision=mask.mp, recall=mask.mr,
                                     map50=mask.map50, map75=mask.map75, map50_95=mask.map))
    parameter = next(model.model.parameters())
    metadata = dict(model=name, configuration=configuration, seed=seed,
                    **scoring,
                    checkpoint=str(weights), checkpoint_sha256=checkpoint_hash,
                    data=options["data"], data_yaml_sha256=sha256(options["data"]),
                    evaluation_options=options,
                    environment=dict(python=platform.python_version(), pytorch=torch.__version__,
                                     cuda=torch.version.cuda, ultralytics=ultralytics.__version__,
                                     ultralytics_file=ultralytics.__file__,
                                     opencv=cv2.__version__, cpu=platform.processor(),
                                     device=str(parameter.device), dtype=str(parameter.dtype),
                                     gpu=torch.cuda.get_device_name(parameter.device) if parameter.device.type == "cuda" else None))
    return row, metadata


def extract_report(path):
    path = project_path(path)
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    name = report.get("run_name", path.parent.name)
    mask = report.get("native", {}).get("mask", report.get("mask"))
    if mask is None:
        raise ValueError("No native mask metrics in " + str(path))
    row = metric_row(name, report.get("seed"), mask)
    metadata = {key: report.get(key) for key in ("configuration", "seed", "checkpoint",
                "checkpoint_sha256", "data", "data_yaml_sha256", "dataset_content_sha256",
                "evaluation_protocol", "environment", "quality_mode", "quality_blend", "quality_formula")}
    metadata.update(model=name, source_report=str(path))
    return row, metadata


def save_results(output, rows, metadata):
    with (output / "paper_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "validation_config.json", dict(
        evaluator_version=EVALUATOR_VERSION, evaluator_sha256=sha256(__file__),
        metric_scale="0-1", f1_definition="harmonic mean of native mask P/R at the framework operating point; not fixed-conf F1",
        mode="extract_existing_reports" if any("source_report" in entry for entry in metadata) else "single_pass_native",
        runs=metadata))


def print_table(rows, output):
    print("\nMask metrics (0-1):")
    print("\t".join(FIELDS))
    for row in rows:
        print("\t".join(str(row[key]) if key in {"model", "seed"} else "{:.4f}".format(row[key]) for key in FIELDS))
    print("Saved: " + str(output / "paper_metrics.csv"))


def main():
    args = parse_args()
    models = models_to_run(args.weights) if args.weights else []
    output = project_path(args.project) / (args.name or datetime.now().strftime("val_%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    rows, metadata = [], []
    if args.quality_mode == "all":
        models = [(name + "_q" + mode, weights, mode) for name, weights in models for mode in QUALITY_MODES]
    else:
        models = [(name, weights, args.quality_mode) for name, weights in models]
    inputs = models if models else args.report
    for index, item in enumerate(inputs, 1):
        print("\n[{}/{}] {}".format(index, len(inputs), item[0] if models else item), flush=True)
        if models:
            selected = copy.copy(args)
            selected.quality_mode = item[2]
            row, info = evaluate_one(selected, item[0], item[1], output)
        else:
            row, info = extract_report(item)
        rows.append(row)
        metadata.append(info)
        save_results(output, rows, metadata)  # preserve completed models if a later one fails
    print_table(rows, output)


if __name__ == "__main__":
    main()
