"""Train FieldBCQ and its boundary-supervision ablations."""

import argparse
import warnings

from project_paths import project_path

from ultralytics import YOLO


STAGES = {
    "baseline": {
        "model": "ultralytics/cfg/models/11/yolo11x-seg.yaml",
        "name": "baseline",
        "overrides": {},
    },
    "pcsc": {
        "model": "ultralytics/cfg/models/11/yolo11x-seg-boundary-dice.yaml",
        "name": "pcsc",
        "overrides": {"boundary": 0.20},
    },
    "baps_pcsc": {
        "model": "ultralytics/cfg/models/11/yolo11x-seg-bas.yaml",
        "name": "baps_pcsc",
        "overrides": {"boundary": 0.20},
    },
    "baps": {
        "model": "ultralytics/cfg/models/11/yolo11x-seg-bas.yaml",
        "name": "baps",
        "overrides": {"boundary": 0.20, "boundary_mode": "bce"},
    },
    "fieldbcq": {
        "model": "ultralytics/cfg/models/11/yolo11x-seg-bas-mqc.yaml",
        "name": "fieldbcq",
        "overrides": {
            "boundary": 0.20,
            "btq_quality": 0.25,
            "btq_boundary_width": 2,
        },
    },
}

STAGE_ALIASES = {"bds": "pcsc", "bds_baps": "baps_pcsc", "baps_only": "baps"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, *STAGE_ALIASES], default="baseline")
    parser.add_argument("--data", default="data/datasets.yaml")
    parser.add_argument("--weights", default="model/pretrained/yolo11x-seg.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0, help="Training-set fraction; use 0.01 for smoke tests.")
    parser.add_argument("--val-fraction", type=float, default=1.0, help="Validation-set fraction; use 0.05 for smoke tests.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project", default="results/train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", default=None, help="Resume from a stage-specific last.pt checkpoint.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load YOLO11x-seg pretrained weights.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    if not 0.0 < args.val_fraction <= 1.0:
        raise ValueError("--val-fraction must be in (0, 1]")
    stage_name = STAGE_ALIASES.get(args.stage, args.stage)
    stage = STAGES[stage_name]

    if args.resume:
        model = YOLO(str(project_path(args.resume)))
        train_args = {"resume": True}
    else:
        model = YOLO(str(project_path("src") / stage["model"]))
        if not args.no_pretrained:
            model.load(str(project_path(args.weights)))
        train_args = {
            "data": str(project_path(args.data)),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "fraction": args.fraction,
            "val_fraction": args.val_fraction,
            "optimizer": "auto",
            "close_mosaic": 10,
            "seed": args.seed,
            "deterministic": True,
            "project": str(project_path(args.project)),
            "name": args.name or f"{stage['name']}_seed{args.seed}",
            "amp": args.amp,
            **stage["overrides"],
        }

    if args.device is not None:
        train_args["device"] = args.device
    from paper_runtime import attach_training_metadata
    attach_training_metadata(model, stage_name, args, train_args)
    model.train(**train_args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
