"""Evaluate the eight paper-stage checkpoints sequentially on one test split."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Keep Ultralytics settings in a writable server directory before importing it.
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    "/home/root1/projects/students/lzw/per_image_validation_ultralytics/_ultralytics_config",
)

from ultralytics import YOLO


DEFAULT_MODELS: List[Tuple[str, Path]] = [
    ("round1_exp3", Path("/home/root1/projects/students/lzw/实验结果/一轮/exp3/weights/best.pt")),
    (
        "round1_boundary_dice",
        Path("/home/root1/projects/students/lzw/实验结果/一轮/exp_BoundaryDice/weights/best.pt"),
    ),
    (
        "round1_bas_boundary_dice",
        Path("/home/root1/projects/students/lzw/实验结果/一轮/exp_BAS_BoundaryDice/weights/best.pt"),
    ),
    (
        "round1_bas_btq_fusion",
        Path("/home/root1/projects/students/lzw/实验结果/一轮/bas_btq_fusion/f1_bas_mqc_seed0/weights/best.pt"),
    ),
    ("round2_exp", Path("/home/root1/projects/students/lzw/实验结果/二轮/exp/weights/best.pt")),
    ("round2_a", Path("/home/root1/projects/students/lzw/实验结果/二轮/a_seed0/weights/best.pt")),
    ("round2_ab", Path("/home/root1/projects/students/lzw/实验结果/二轮/ab_seed0/weights/best.pt")),
    ("round2_abc", Path("/home/root1/projects/students/lzw/实验结果/二轮/abc_seed0/weights/best.pt")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        action="append",
        dest="weights",
        metavar="NAME=WEIGHTS",
        help="Override the defaults; repeat for multiple models. A bare path is also accepted.",
    )
    parser.add_argument("--data", default="dataset/datasets.yaml")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int, default=710)
    parser.add_argument("--project", default="runs/paper_test")
    parser.add_argument("--name", default=None, help="Optional prefix for each model's validation run name.")
    return parser.parse_args()


def parse_models(values: Optional[Sequence[str]]) -> List[Tuple[str, Path]]:
    if not values:
        return list(DEFAULT_MODELS)

    models = []
    names = set()
    for value in values:
        if "=" in value:
            name, raw_path = value.split("=", 1)
            name, raw_path = name.strip(), raw_path.strip()
        else:
            raw_path = value.strip()
            name = Path(raw_path).parent.parent.name
        if not name or not raw_path:
            raise ValueError("--weights must use NAME=WEIGHTS or a non-empty checkpoint path")
        if name in names:
            raise ValueError("Duplicate model name: {}".format(name))
        names.add(name)
        models.append((name, Path(raw_path).expanduser()))
    return models


def paper_metrics(results) -> dict:
    return {
        "box": {
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
        },
        "mask": {
            "precision": float(results.seg.mp),
            "recall": float(results.seg.mr),
            "map50": float(results.seg.map50),
            "map75": float(results.seg.map75),
            "map50_95": float(results.seg.map),
        },
    }


def main() -> None:
    args = parse_args()
    models = parse_models(args.weights)
    project = Path(args.project).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for index, (model_name, weights) in enumerate(models, start=1):
        if not weights.is_file():
            raise FileNotFoundError("Model weights not found: {}".format(weights))

        print("\n[{}/{}] {} -> {}".format(index, len(models), model_name, weights), flush=True)
        model = YOLO(str(weights))
        run_name = "{}_{}".format(args.name, model_name) if args.name else model_name
        val_args = {
            "data": args.data,
            "split": args.split,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "max_det": args.max_det,
            "project": str(project),
            "name": run_name,
            "plots": True,
            "exist_ok": True,
        }
        if args.device is not None:
            val_args["device"] = args.device

        results = model.val(**val_args)
        metrics = paper_metrics(results)
        record = {
            "model": model_name,
            "weights": str(weights),
            "run_name": run_name,
            "data": str(args.data),
            "split": args.split,
            "metrics": metrics,
        }
        all_metrics.append(record)

        output = Path(results.save_dir) / "paper_metrics.json"
        output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Paper metrics saved to: {}".format(output), flush=True)

        summary = project / "paper_metrics_all.json"
        summary.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        del model
        gc.collect()

    print("\nAll model metrics saved to: {}".format(project / "paper_metrics_all.json"), flush=True)


if __name__ == "__main__":
    main()
