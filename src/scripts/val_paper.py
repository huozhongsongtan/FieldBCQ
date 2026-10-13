"""Evaluate a paper-stage checkpoint on the fixed test split."""

import argparse
import json
from pathlib import Path

from project_paths import project_path, relative_path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/datasets.yaml")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int, default=710)
    parser.add_argument("--project", default="results/runs")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    weights = project_path(args.weights)
    model = YOLO(str(weights))
    val_args = {
        "data": str(project_path(args.data)),
        "split": args.split,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "max_det": args.max_det,
        "project": str(project_path(args.project)),
        "name": args.name or weights.parent.name,
        "plots": True,
    }
    if args.device is not None:
        val_args["device"] = args.device
    results = model.val(**val_args)
    metrics = {
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
            "map50_95": float(results.seg.map),
        },
    }
    output = Path(results.save_dir) / "paper_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Paper metrics saved to: {relative_path(output)}")


if __name__ == "__main__":
    main()
