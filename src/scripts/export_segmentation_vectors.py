"""Freeze FieldBCQ predictions for the subsequent five-arm experiment.

Each ORIGINAL instance mask is vectorized separately without simplification.
Pixel-cell polygonization preserves all components, holes and instance overlap.
"""
import argparse
from pathlib import Path
import sys
import time

from project_paths import PROJECT_ROOT, project_path

sys.path.insert(0, str(PROJECT_ROOT / "src/postprocessing"))
from fieldagent import activate_engine
from fieldagent.io import digest, load, save


def export_result(result, folder, image_sha256):
    """Lossless mask export, also used by the integration regression test."""
    import numpy as np
    from fieldbqc_agent import parcels_from_ultralytics
    from fieldbqc_geometry import from_parcels, geometry_mask, vertices
    from cvr_agent import save_items
    from predict_fieldbqc_agent import save_parcels
    parcels = parcels_from_ultralytics(result)
    items = from_parcels(parcels)
    if not all(np.array_equal(p.mask, geometry_mask(g.geometry, result.orig_shape))
               for p, g in zip(parcels, items)):
        raise ValueError("Instance mask/vector round-trip mismatch")
    folder.mkdir(parents=True, exist_ok=True)
    vector = folder / "prediction.pixel.geojson"
    save_parcels(folder / "prediction.original_masks.npz", parcels, result.orig_shape)
    saved = save_items(vector, items)
    record = dict(image_sha256=image_sha256, vector_sha256=saved["sha256"],
                  masks_sha256=digest(folder / "prediction.original_masks.npz"),
                  instances=len(items), vertices=sum(vertices(p.geometry) for p in items),
                  shape=list(result.orig_shape), mask_roundtrip=True)
    return vector, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="model/abc/best.pt")
    parser.add_argument("--data", default="data/datasets.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--include-labels", action="store_true",
                        help="Link labels for OFFLINE evaluation only; never sent to the agent")
    parser.add_argument("--limit", type=int, default=0, help="First N images for smoke checks; 0=all")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.limit < 0:
        parser.error("threads must be positive and limit nonnegative")
    activate_engine()
    from val_paper import check_weight_file
    from paper_data import resolve_images
    weights = project_path(args.weights)
    check_weight_file(weights)
    data, _, _, images = resolve_images(args.data, args.split)
    images = images[:args.limit or None]
    if len({p.stem.casefold() for p in images}) != len(images):
        raise ValueError("Duplicate image stems; provide a dataset with unique scene IDs")
    from ultralytics.data.utils import img2label_paths
    entries = []
    for image in images:
        entry = dict(scene=image.stem, domain=image.stem.split("_")[0],
                     image=str(image), image_sha256=digest(image))
        if args.include_labels:
            label = Path(img2label_paths([str(image)])[0])
            if not label.is_file():
                raise FileNotFoundError("Missing label; negatives require an empty file: " + str(label))
            entry.update(label=str(label), label_sha256=digest(label))
        entries.append(entry)
    if args.check_only:
        print(f"Checked checkpoint and {len(entries)} images; inference/API calls=0")
        return 0
    output = project_path(args.output)
    inference = dict(imgsz=640, conf=.25, iou=.70, max_det=710,
                     retina_masks=True, half=False, augment=False, device=args.device)
    sources = [*Path(__file__).parent.glob("*.py"),
               *(PROJECT_ROOT / "src/ultralytics").rglob("*.py"),
               *(PROJECT_ROOT / "src/ultralytics").rglob("*.yaml"),
               *(PROJECT_ROOT / "src/postprocessing").rglob("*.py"),
               *(PROJECT_ROOT / "src/postprocessing").rglob("*.md")]
    frozen = dict(version="fieldbcq-postprocess-inference-v1", entries=entries,
                  weights=str(weights), weights_sha256=digest(weights), data_sha256=digest(data),
                  inference=inference, threads=args.threads, precision="fp32", scoring="joint",
                  source_sha256={p.relative_to(PROJECT_ROOT).as_posix(): digest(p) for p in sorted(sources)})
    manifest = output / "inference_manifest.json"
    if manifest.exists():
        if load(manifest) != frozen:
            raise ValueError("Frozen inference changed; choose a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("A new inference requires an empty output directory")
    save(manifest, frozen)
    rows, model = [], None
    for number, entry in enumerate(entries, 1):
        folder = output / "predictions" / entry["scene"]
        vector, done = folder / "prediction.pixel.geojson", folder / "completed.json"
        if done.exists():
            record = load(done)
            if digest(vector) != record["vector_sha256"] or digest(folder / "prediction.original_masks.npz") != record["masks_sha256"]:
                raise ValueError("Frozen prediction cache changed: " + entry["scene"])
        else:
            if model is None:
                import torch
                from ultralytics import YOLO
                from paper_runtime import configuration_of
                from quality_scoring import set_quality_scoring
                torch.set_num_threads(args.threads)
                model = YOLO(str(weights), task="segment")
                if configuration_of(model.model) != "fieldbcq":
                    raise ValueError("Expected the FieldBCQ A+B+C checkpoint")
                model.model.float()
                set_quality_scoring(model.model, "joint")
            start = time.perf_counter()
            result = model.predict(source=entry["image"], save=False, verbose=False, **inference)[0]
            vector, record = export_result(result, folder, entry["image_sha256"])
            record["inference_seconds"] = time.perf_counter() - start
            save(done, record)
        rows.append({**entry, "prediction_vector": str(vector),
                     "prediction_vector_sha256": digest(vector)})
        print(f"PREDICT {number}/{len(entries)} {entry['scene']}", flush=True)
    save(output / "inputs.json", dict(coordinate_space="image_pixels", inputs=rows,
        split=args.split, development_only=args.split != "test", labels_offline_only=True,
        inference_manifest_sha256=digest(manifest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
