"""Run a small train/validation/checkpoint smoke test for all five configurations."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import yaml

from project_paths import project_path
from paper_runtime import attach_training_metadata, configuration_of, environment, sha256, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/datasets.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-images", type=int, default=16)
    parser.add_argument("--val-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", choices=("x", "n"), default="x", help="x tests the paper models; n is a smaller pipeline check.")
    parser.add_argument("--weights", help="Optional local pretrained checkpoint; default is random initialization.")
    parser.add_argument("--amp", action="store_true", help="Also exercise AMP; default FP32 avoids AMP-check downloads.")
    parser.add_argument("--stages", nargs="+", default=["baseline", "bds", "baps", "bds_baps", "fieldbcq"],
                        choices=("baseline", "bds", "pcsc", "baps", "baps_only", "bds_baps", "baps_pcsc", "fieldbcq"))
    parser.add_argument("--project", default="results/smoke")
    parser.add_argument("--name", help="Optional fresh run directory name; default includes a timestamp.")
    args = parser.parse_args()
    if args.imgsz < 64 or args.imgsz % 32 or args.batch < 2:
        parser.error("imgsz must be >=64 and divisible by 32; batch must be >=2")
    if min(args.epochs, args.val_images) < 1 or args.train_images < args.batch or args.workers < 0:
        parser.error("Require epochs/val-images >=1, train-images >=batch and workers >=0")
    if args.name and (args.name in {".", ".."} or any(c in args.name for c in "/\\:")):
        parser.error("--name must be a directory name without path separators")
    return args


def prepare_fonts():
    """Use a bundled local font for smoke checks instead of waiting for a download."""
    import matplotlib
    from ultralytics.utils import USER_CONFIG_DIR
    source = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("Arial.ttf", "Arial.Unicode.ttf"):
        target = USER_CONFIG_DIR / name
        if not target.exists() or not target.stat().st_size:
            shutil.copyfile(source, target)


def prepare_subset(args, output):
    from ultralytics.data.utils import img2label_paths
    from val_by_source import resolve_images
    selected, config = {}, None
    for split, limit in (("train", args.train_images), ("val", args.val_images)):
        _, config, images = resolve_images(args.data, split)
        random.Random(args.seed).shuffle(images)
        chosen = images[:limit]
        if len(chosen) < (args.batch if split == "train" else 1):
            raise ValueError(f"Not enough {split} images for smoke: {len(chosen)}")
        labels = [Path(path) for path in img2label_paths([str(p) for p in chosen])]
        for label in labels:
            if not label.is_file():
                raise FileNotFoundError(f"Missing smoke label: {label}; negative images need an empty label file")
        if not any(label.read_text(encoding="utf-8-sig").strip() for label in labels):
            raise ValueError(f"Selected {split} images have no parcels; increase --{split}-images or change --seed")
        selected[split] = chosen
        (output / f"{split}.txt").write_text("".join(str(path) + "\n" for path in chosen), encoding="utf-8")
    names = config.get("names")
    if names is None:
        names = [f"class_{index}" for index in range(config["nc"])]
    subset = {"path": str(output), "train": "train.txt", "val": "val.txt", "nc": len(names), "names": names}
    dataset = output / "smoke.yaml"
    dataset.write_text(yaml.safe_dump(subset, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dataset, selected


def run_stage(args, public_name, canonical, stage, run, dataset, selected):
    import torch
    from ultralytics import YOLO
    definition = yaml.safe_load((project_path("src") / stage["model"]).read_text(encoding="utf-8"))
    definition["scale"] = args.scale
    model_yaml = run / "data" / f"yolo11{args.scale}-seg-{public_name}.yaml"
    model_yaml.write_text(yaml.safe_dump(definition, sort_keys=False), encoding="utf-8")
    options = dict(data=str(dataset), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                   workers=args.workers, device=args.device, seed=args.seed, deterministic=True,
                   project=str(run), name=public_name, exist_ok=False, pretrained=False,
                   amp=args.amp, plots=False, mosaic=0.0, close_mosaic=0,
                   optimizer="auto", save=True, val=True, **stage["overrides"])
    model = YOLO(str(model_yaml))
    if args.weights:
        model.load(str(project_path(args.weights)))
    attach_training_metadata(model, canonical, args, options)
    model.train(**options)
    output = Path(model.trainer.save_dir)
    checkpoint = output / "weights/best.pt"
    if not checkpoint.is_file() or not (output / "weights/last.pt").is_file():
        raise RuntimeError("Training did not create both best.pt and last.pt")
    with (output / "results.csv").open(encoding="utf-8", newline="") as handle:
        losses = [float(value) for row in csv.DictReader(handle) for key, value in row.items()
                  if key.strip().endswith("loss")]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise RuntimeError("Training/validation losses are missing or nonfinite")
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata["configuration"] != canonical or metadata["completed_epochs"] < 1:
        raise RuntimeError("Training metadata does not confirm the requested configuration")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    restored = YOLO(str(checkpoint))
    if configuration_of(restored.model) != canonical:
        raise RuntimeError("Configuration changed when loading the saved checkpoint")
    predictions = restored.predict(source=str(selected["val"][0]), imgsz=args.imgsz,
                                   device=args.device, half=False, rect=False, save=False, verbose=False)
    if len(predictions) != 1 or not torch.isfinite(predictions[0].boxes.data).all():
        raise RuntimeError("Reloaded checkpoint prediction failed or produced nonfinite values")
    result = {"status": "passed", "configuration": canonical, "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha256(checkpoint), "completed_epochs": metadata["completed_epochs"],
              "checked_losses": len(losses), "validation": "native per-epoch and final best-checkpoint validation",
              "reload_prediction": "passed"}
    del restored, predictions
    return result


def main():
    args = parse_args()
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import torch
    from train_paper import STAGES, STAGE_ALIASES
    stages = [(name, STAGE_ALIASES.get(name, name)) for name in args.stages]
    if len(set(canonical for _, canonical in stages)) != len(stages):
        raise ValueError("The same configuration was requested more than once through aliases")
    if args.weights and not project_path(args.weights).is_file():
        raise FileNotFoundError(project_path(args.weights))
    run = project_path(args.project) / (args.name or datetime.now().strftime("smoke_%Y%m%d_%H%M%S_%f"))
    run.mkdir(parents=True, exist_ok=False)
    (run / "data").mkdir()
    prepare_fonts()
    dataset, selected = prepare_subset(args, run / "data")
    summary = {"is_smoke_test": True, "scale": args.scale, "args": vars(args), "environment": environment(),
               "subset": {key: [str(path) for path in value] for key, value in selected.items()}, "stages": {}}
    report = run / "smoke_summary.json"
    write_json(report, summary)
    for public_name, canonical in stages:
        print(f"\nSMOKE START: {public_name}, scale={args.scale}, seed={args.seed}", flush=True)
        try:
            summary["stages"][public_name] = run_stage(args, public_name, canonical, STAGES[canonical], run, dataset, selected)
            print(f"SMOKE PASS: {public_name}", flush=True)
        except Exception as exc:
            summary["stages"][public_name] = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
            traceback.print_exc()
            print(f"SMOKE FAIL: {public_name}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_json(report, summary)
    passed = sum(row["status"] == "passed" for row in summary["stages"].values())
    print(f"\nSMOKE RESULT: {passed}/{len(stages)} passed; report: {report}", flush=True)
    raise SystemExit(0 if passed == len(stages) else 1)


if __name__ == "__main__":
    main()
