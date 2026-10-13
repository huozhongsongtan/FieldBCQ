"""Evaluate one segmentation checkpoint on source-specific subsets of test.

The script reads the existing datasets.yaml, resolves its ``test`` entry, and
filters image paths by source tokens such as DDFHAPD, FBIS, and FTW. It writes
temporary test lists under the run directory, so the original dataset files
are never changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from project_paths import project_path, relative_path

import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed segmentation checkpoint on DDFHAPD/FBIS/FTW test subsets."
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/datasets.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--sources", default="DDFHAPD,FBIS,FTW")
    parser.add_argument(
        "--domestic-token",
        default="DDFHAPD",
        help="Path/name token used for the domestic (China) aggregate group.",
    )
    parser.add_argument(
        "--foreign-tokens",
        default="FBIS,FTW",
        help="Comma-separated path/name tokens used for the foreign aggregate group.",
    )
    parser.add_argument(
        "--source-token",
        action="append",
        default=[],
        metavar="NAME=TOKEN",
        help="Add or override a source token, for example --source-token china=CQ.",
    )
    parser.add_argument("--project", default="results/source_eval")
    parser.add_argument("--name", default="best_model")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--quality-blend",
        type=float,
        default=1.0,
        help="Set MQC quality blending when the checkpoint exposes quality_blend.",
    )
    parser.add_argument("--metadata-csv", default=None, help="Optional metadata CSV for image-count auditing.")
    return parser.parse_args()


def _resolve_root(data_path: Path, config: dict) -> Path:
    raw_root = os.path.expandvars(str(config.get("path") or "."))
    root = Path(raw_root).expanduser()
    return (data_path.parent / root).resolve() if not root.is_absolute() else root.resolve()


def _expand_entry(entry, root: Path) -> List[Path]:
    path = Path(os.path.expandvars(str(entry))).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    if path.is_file() and path.suffix.lower() == ".txt":
        paths = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            item = Path(os.path.expandvars(value)).expanduser()
            if not item.is_absolute():
                local = (path.parent / item).resolve()
                item = local if local.exists() else (root / item).resolve()
            paths.append(item.resolve())
        return paths
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    raise FileNotFoundError("Test entry does not exist: {}".format(path))


def resolve_images(data: str, split: str) -> tuple[Path, dict, List[Path]]:
    data_path = project_path(data)
    with data_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    root = _resolve_root(data_path, config)
    entries = config.get(split)
    if entries is None:
        raise KeyError("Dataset YAML has no '{}' entry".format(split))
    entries = entries if isinstance(entries, (list, tuple)) else [entries]
    images = []
    for entry in entries:
        images.extend(_expand_entry(entry, root))
    images = list(dict.fromkeys(path.resolve() for path in images))
    missing = [path for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError("{} image(s) from {} are missing; first: {}".format(len(missing), split, missing[0]))
    return data_path, config, images


def parse_source_tokens(args) -> Dict[str, str]:
    source_names = [value.strip() for value in args.sources.split(",") if value.strip()]
    tokens = {name: name for name in source_names}
    for value in args.source_token:
        if "=" not in value:
            raise ValueError("--source-token must use NAME=TOKEN, received '{}'".format(value))
        name, token = value.split("=", 1)
        if not name.strip() or not token.strip():
            raise ValueError("--source-token must use non-empty NAME=TOKEN")
        tokens[name.strip()] = token.strip()
    return tokens


def filter_sources(
    images: Sequence[Path],
    tokens: Dict[str, str],
    domestic_token: str = "DDFHAPD",
    foreign_tokens: Sequence[str] = ("FBIS", "FTW"),
) -> Dict[str, List[Path]]:
    normalized = [(path, str(path).replace("\\", "/").lower()) for path in images]
    groups = {"all": [path for path, _ in normalized]}
    domestic_token = domestic_token.strip().lower()
    foreign_tokens = tuple(token.strip().lower() for token in foreign_tokens if token.strip())
    if domestic_token:
        groups["domestic"] = [path for path, value in normalized if domestic_token in value]
    if foreign_tokens:
        groups["foreign"] = [
            path for path, value in normalized if any(token in value for token in foreign_tokens)
        ]
    for name, token in tokens.items():
        token = token.lower()
        groups[name] = [path for path, value in normalized if token in value]
    return groups


def write_group_data(
    base_config: dict, groups: Dict[str, List[Path]], output_dir: Path, dataset_root: Path
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_paths = {}
    for name, images in groups.items():
        list_path = output_dir / "{}_images.txt".format(name)
        # Ultralytics anchors entries beginning with ./ to the image-list file.
        list_path.write_text(
            "\n".join("./" + relative_path(path, output_dir) for path in images) + "\n",
            encoding="utf-8",
        )
        group_config = dict(base_config)
        # Omitting path makes the generated YAML's directory its dataset root.
        group_config.pop("path", None)
        for split in ("train", "val", "minival"):
            entries = base_config.get(split)
            if not entries:
                continue
            values = entries if isinstance(entries, list) else [entries]
            paths = [
                relative_path(dataset_root / Path(os.path.expandvars(str(entry))).expanduser(), output_dir)
                for entry in values
            ]
            group_config[split] = paths if isinstance(entries, list) else paths[0]
        group_config["test"] = list_path.name
        yaml_path = output_dir / "{}_data.yaml".format(name)
        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(group_config, handle, allow_unicode=True, sort_keys=False)
        data_paths[name] = yaml_path
    return data_paths


def load_metadata_counts(metadata_csv: str, split: str) -> Dict[str, int]:
    counts = {}
    if not metadata_csv:
        return counts
    with project_path(metadata_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            if row.get("split") and row["split"] != split:
                continue
            source = row.get("source", "unknown") or "unknown"
            counts[source] = counts.get(source, 0) + 1
    return counts


def metric_record(name: str, result, image_count: int) -> dict:
    metric = result.seg
    precision = float(metric.mp)
    recall = float(metric.mr)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "group": name,
        "images": int(image_count),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(metric.map50),
        "map75": float(metric.map75),
        "map50_95": float(metric.map),
        "box_map50_95": float(result.box.map),
    }


def main():
    args = parse_args()
    if not 0.0 <= args.quality_blend <= 1.0:
        raise ValueError("--quality-blend must be in [0, 1]")

    data_path, base_config, images = resolve_images(args.data, args.split)
    tokens = parse_source_tokens(args)
    domestic_token = args.domestic_token
    foreign_tokens = args.foreign_tokens.split(",")
    groups = filter_sources(images, tokens, domestic_token=domestic_token, foreign_tokens=foreign_tokens)
    empty = [name for name, values in groups.items() if not values]
    if empty:
        raise RuntimeError(
            "No images matched source group(s): {}. Source names must occur in the resolved test path. "
            "Use --source-token NAME=TOKEN when folder names differ.".format(", ".join(empty))
        )

    project = project_path(args.project)
    generated = project / args.name / "group_data"
    dataset_root = _resolve_root(data_path, base_config)
    group_data = write_group_data(base_config, groups, generated, dataset_root)
    metadata_counts = load_metadata_counts(args.metadata_csv, args.split)
    print("Resolved {} {} images from {}".format(len(images), args.split, relative_path(data_path)))
    for name, values in groups.items():
        print("  {:>16}: {:>6} images".format(name, len(values)))
    if metadata_counts:
        print("Metadata CSV counts: {}".format(metadata_counts))

    model = YOLO(str(project_path(args.weights)))
    head = getattr(model.model, "model", [None])[-1]
    if hasattr(head, "quality_blend"):
        head.quality_blend = args.quality_blend

    records = []
    for name, yaml_path in group_data.items():
        result = model.val(
            data=str(yaml_path),
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            split="test",
            conf=args.conf,
            max_det=args.max_det,
            augment=False,
            plots=False,
            project=str(project),
            name="{}_{}".format(args.name, name.lower()),
            exist_ok=True,
            **({"device": args.device} if args.device is not None else {}),
        )
        record = metric_record(name, result, len(groups[name]))
        records.append(record)
        print(
            "{:<16} images={:>5} P={:.4f} R={:.4f} F1={:.4f} mAP50={:.4f} mAP75={:.4f} mAP50-95={:.4f}".format(
                name,
                len(groups[name]),
                record["precision"],
                record["recall"],
                record["f1"],
                record["map50"],
                record["map75"],
                record["map50_95"],
            )
        )

    report = project / args.name / "source_test_metrics.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSource metrics saved to {}".format(relative_path(report)))


if __name__ == "__main__":
    main()
