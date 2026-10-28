"""Run FieldBQC followed by the conservative FieldBQC-Agent post-processor.

The script keeps the original Ultralytics predictions untouched and writes the
agent candidate masks and JSON traces to a separate output directory.  This
makes it suitable for an E0/E1 paired evaluation before changing production
outputs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from fieldbqc_agent import AgentConfig, FieldBqcAgent, LocalReinfer, parcels_from_ultralytics
from fieldbqc_policy import DomainRouter, image_metadata, load_metadata_manifest


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="FieldBQC/Ultralytics segmentation checkpoint")
    parser.add_argument("--source", required=True, help="Image, directory, or source list")
    parser.add_argument("--output", type=Path, required=True, help="Separate agent output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit directory sources to the first N images")
    parser.add_argument(
        "--predictor",
        choices=("standard", "btq", "harm"),
        default="standard",
        help="Standard NMS with MQC capture, or an explicitly selected legacy predictor.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=710)
    parser.add_argument("--accept-quality", type=float, default=0.72)
    parser.add_argument("--review-quality", type=float, default=0.45)
    parser.add_argument(
        "--min-peak-distance",
        type=int,
        default=12,
        help="Minimum pixel distance between merge-detector peaks.",
    )
    parser.add_argument(
        "--min-peak-value",
        type=float,
        default=4.0,
        help="Minimum distance-transform value for a merge-detector peak.",
    )
    parser.add_argument(
        "--min-edge-alignment",
        type=float,
        default=0.18,
        help="Minimum mask/image edge agreement before boundary review.",
    )
    parser.add_argument(
        "--split-merge-risk",
        type=float,
        default=0.5,
        help="Merge-warning threshold used to route distance-transform adhesion splitting.",
    )
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--sensor", default="unknown")
    parser.add_argument("--resolution", default="unknown")
    parser.add_argument("--domain", choices=("auto", "unknown", "ddfhapd", "ftw", "s2", "planet", "ortho", "maxar"), default="auto")
    parser.add_argument("--manifest", type=Path, help="CSV, JSON or JSONL image metadata; no labels are loaded")
    parser.add_argument("--policy-config", type=Path, help="JSON object mapping domains to policy overrides")
    parser.add_argument("--scene-id", default=None, help="Known acquisition/scene id for GeoTIFF association")
    parser.add_argument("--disable-local", action="store_true")
    parser.add_argument("--disable-boundary", action="store_true")
    parser.add_argument("--disable-split", action="store_true", help="Disable distance-transform adhesion splitting")
    parser.add_argument("--disable-merge", action="store_true", help="Disable disconnected-fragment merging")
    parser.add_argument("--disable-simplify", action="store_true", help="Disable contour simplification")
    parser.add_argument("--max-contour-vertices", type=int, default=80)
    parser.add_argument("--max-split-parts", type=int, default=6)
    parser.add_argument("--max-local-calls", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--fuse-tiles", action="store_true", help="Run verified cross-tile association after image tools")
    return parser.parse_args()


def _safe_stem(path: Any, index: int) -> str:
    candidate = Path(str(path)).stem
    return candidate if candidate else f"image_{index:06d}"


def _patch_legacy_btq_predictor(predictor: Any) -> Any:
    """Bridge BTQ overlays written for Ultralytics versions with construct_results()."""
    if hasattr(predictor, "construct_results") or not hasattr(predictor, "construct_result"):
        return predictor

    def construct_results(self: Any, predictions: Any, img: Any, orig_imgs: Any, protos: Any) -> Any:
        return [
            self.construct_result(pred, img, orig_img, img_path, protos[index])
            for index, (pred, orig_img, img_path) in enumerate(zip(predictions, orig_imgs, self.batch[0]))
        ]

    predictor.construct_results = construct_results
    return predictor


def _find_model_head(auto_backend: Any) -> Optional[Any]:
    """Return the final detection/segmentation head from an Ultralytics backend."""
    model = getattr(auto_backend, "model", None)
    layers = getattr(model, "model", None)
    if layers is not None and len(layers):
        return layers[-1]
    return None


class _QualityCapture:
    """Capture the two-channel MQC/RBQC logits emitted by a FieldBQC head."""

    def __init__(self, head: Any):
        self.head = head
        self.outputs: Dict[int, Any] = {}
        self.handles: List[Any] = [head.register_forward_pre_hook(self._clear)]
        for index, branch in enumerate(head.cv5):
            self.handles.append(branch.register_forward_hook(self._make_hook(index)))

    def _clear(self, _module: Any, _inputs: Any) -> None:
        self.outputs.clear()

    def _make_hook(self, index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self.outputs[index] = output.detach()

        return hook

    def logits(self) -> Optional[Any]:
        if len(self.outputs) != len(self.head.cv5):
            return None
        import torch

        tensors = [self.outputs[index] for index in range(len(self.head.cv5))]
        batch_size = tensors[0].shape[0]
        return torch.cat([tensor.view(batch_size, 2, -1) for tensor in tensors], dim=2)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _build_quality_btq_predictor(base_predictor: Any, use_topology_nms: bool = True) -> Any:
    """Wrap the repository BTQ predictor and retain MQC scores through NMS.

    The quality head is an auxiliary output and is not part of the standard
    Ultralytics ``Results`` contract.  Anchor ids are temporarily appended to
    each candidate row before NMS, allowing the raw quality logits to be
    indexed after box and topology mask suppression.
    """
    import torch
    from ultralytics.engine.results import Results
    from ultralytics.utils import ops
    if use_topology_nms:
        from ultralytics.utils.btq import topology_mask_nms

    class QualityBTQSegmentationPredictor(base_predictor):
        def setup_model(self, model: Any, verbose: bool = True) -> None:
            super().setup_model(model, verbose=verbose)
            head = _find_model_head(self.model)
            self.quality_capture = (
                _QualityCapture(head) if head is not None and hasattr(head, "cv5") else None
            )

        def postprocess(self, preds: Any, img: Any, orig_imgs: Any) -> List[Any]:
            if isinstance(preds[0], tuple):
                prediction, proto = preds[0]
            else:
                prediction = preds[0]
                proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]

            quality_logits = (
                self.quality_capture.logits()
                if getattr(self, "quality_capture", None) is not None
                else None
            )
            capture_quality = (
                quality_logits is not None
                and quality_logits.ndim == 3
                and quality_logits.shape[2] == prediction.shape[2]
            )
            if capture_quality:
                prediction = prediction.float()
                anchor_ids = torch.arange(
                    prediction.shape[2], device=prediction.device, dtype=prediction.dtype
                ).view(1, 1, -1)
                anchor_ids = anchor_ids.expand(prediction.shape[0], 1, -1)
                prediction = torch.cat((prediction, anchor_ids), dim=1)

            candidates = ops.non_max_suppression(
                prediction,
                self.args.conf,
                self.args.btq_box_nms_iou if use_topology_nms else self.args.iou,
                self.args.classes,
                self.args.agnostic_nms,
                max_det=self.args.btq_mask_nms_candidates if use_topology_nms else self.args.max_det,
                nc=len(self.model.names),
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

            results: List[Any] = []
            mask_nms_iou = getattr(self.args, "btq_mask_nms_iou", 0.70)
            mask_nms_candidates = getattr(self.args, "btq_mask_nms_candidates", 1000)
            for batch_index, (pred, orig_img, img_path) in enumerate(
                zip(candidates, orig_imgs, self.batch[0])
            ):
                if capture_quality:
                    kept_anchor_ids = pred[:, -1].long()
                    mask_coefficients = pred[:, 6:-1]
                else:
                    kept_anchor_ids = None
                    mask_coefficients = pred[:, 6:]

                if not len(pred):
                    masks = None
                elif self.args.retina_masks:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    masks = ops.process_mask_native(
                        proto[batch_index], mask_coefficients, pred[:, :4], orig_img.shape[:2]
                    )
                else:
                    masks = ops.process_mask(
                        proto[batch_index],
                        mask_coefficients,
                        pred[:, :4],
                        img.shape[2:],
                        upsample=True,
                    )
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

                quality = None
                if capture_quality and len(pred):
                    quality = quality_logits[batch_index, :, kept_anchor_ids].sigmoid().float()

                if masks is not None:
                    nonempty = masks.sum((-2, -1)) > 0
                    pred, masks = pred[nonempty], masks[nonempty]
                    if quality is not None:
                        quality = quality[:, nonempty]
                    if use_topology_nms:
                        keep = topology_mask_nms(
                            masks, pred[:, 4], pred[:, 5], boxes=pred[:, :4],
                            iou_threshold=mask_nms_iou, max_det=self.args.max_det,
                            candidate_limit=mask_nms_candidates, agnostic=self.args.agnostic_nms,
                        )
                        pred, masks = pred[keep], masks[keep]
                        if quality is not None:
                            quality = quality[:, keep]

                result = Results(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred[:, :6],
                    masks=masks,
                )
                if quality is not None:
                    result.mqc_mask_iou_scores = quality[0].detach().cpu().numpy()
                    result.mqc_boundary_f1_scores = quality[1].detach().cpu().numpy()
                    result.mqc_combined_scores = quality.prod(0).sqrt().detach().cpu().numpy()
                results.append(result)
            return results

    return QualityBTQSegmentationPredictor


def _serializable_trace(output: Any, image_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image": image_path,
        "metadata": metadata,
        "accepted": bool(output.accepted),
        "needs_review": bool(output.needs_review),
        "review_ids": output.review_ids,
        "domain_policy": output.domain_policy,
        "instances": len(output.parcels),
        "final_state": {
            "mean_quality": output.final_state.mean_quality,
            "mean_edge_alignment": output.final_state.mean_edge_alignment,
            "tta_disagreement": output.final_state.tta_disagreement,
            "high_risk_count": output.final_state.high_risk_count,
            "topology_valid": output.final_state.topology_valid,
            "observations": [
                {
                    "index": item.index,
                    "quality": item.quality,
                    "edge_alignment": item.edge_alignment,
                    "border_risk": item.border_risk,
                    "merge_risk": item.merge_risk,
                    "fragment_risk": item.fragment_risk,
                    "topology_valid": item.topology_valid,
                    "issue_type": item.issue_type,
                    "issue_types": item.issue_types,
                    "parcel_id": item.parcel_id,
                    "area_fraction": item.area_fraction,
                    "component_count": item.component_count,
                    "contour_vertices": item.contour_vertices,
                    "merge_partner": item.merge_partner,
                    "distance_peak_count": item.distance_peak_count,
                    "severity": item.severity,
                    "reasons": item.reasons,
                }
                for item in output.final_state.observations
            ],
        },
        "action_trace": output.action_trace,
    }


def resolve_sources(source: str, limit: int = 0) -> List[Path]:
    path = Path(source).resolve()
    if path.is_dir():
        paths = sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    elif path.suffix.lower() == ".txt":
        paths = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                item = Path(line.strip())
                paths.append(item.resolve() if item.is_absolute() else (path.parent / item).resolve())
    else:
        paths = [path]
    paths = paths[:limit] if limit > 0 else paths
    if not paths or any(not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES for item in paths):
        raise ValueError("Source must contain existing image files")
    if len({item.stem.casefold() for item in paths}) != len(paths):
        raise ValueError("Duplicate image stems would overwrite output; use a source list with unique image ids")
    return paths


def save_parcels(path: Path, parcels: List[Any], shape: Any) -> None:
    def values(name):
        return np.asarray([getattr(parcel, name) if parcel.quality_valid and getattr(parcel, name) is not None
                           else np.nan for parcel in parcels], dtype=np.float32)
    np.savez_compressed(
        path, masks=np.stack([parcel.mask for parcel in parcels]) if parcels else np.zeros((0, *shape), dtype=bool),
        scores=np.asarray([parcel.score for parcel in parcels], dtype=np.float32),
        classes=np.asarray([parcel.cls for parcel in parcels], dtype=np.int64),
        parcel_ids=np.asarray([parcel.parcel_id for parcel in parcels], dtype=str),
        qualities=values("quality"), mask_qualities=values("mask_quality"),
        boundary_qualities=values("boundary_quality"),
        quality_valid=np.asarray([parcel.quality_valid and parcel.quality is not None for parcel in parcels]),
        quality_sources=np.asarray([parcel.quality_source for parcel in parcels], dtype=str),
    )


def main() -> int:
    args = parse_args()
    sources = resolve_sources(args.source, args.limit)
    metadata_index = load_metadata_manifest(args.manifest)
    policy_values = json.loads(args.policy_config.read_text(encoding="utf-8-sig")) if args.policy_config else {}
    router = DomainRouter(policy_values)
    output_root = args.output
    mask_root = output_root / "masks"
    trace_root = output_root / "traces"
    original_root = output_root / "original_masks"
    if (output_root / "run_config.json").exists() or (trace_root.exists() and any(trace_root.glob("*.json"))):
        raise FileExistsError("Choose a new output directory to keep runs and original predictions separate")
    mask_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    original_root.mkdir(parents=True, exist_ok=True)

    # Import the project-specific Ultralytics package only when this entrypoint
    # is executed, so fieldbqc_agent.py remains usable for dependency-free tests.
    from ultralytics import YOLO

    if args.predictor == "standard":
        from ultralytics.models.yolo.segment.predict import SegmentationPredictor
        predictor = _build_quality_btq_predictor(SegmentationPredictor, use_topology_nms=False)
    else:
        try:
            if args.predictor == "btq":
                from btq_runtime import BTQSegmentationPredictor

                predictor = _build_quality_btq_predictor(BTQSegmentationPredictor)
            else:
                from harm_runtime import HARMSegmentationPredictor

                predictor = HARMSegmentationPredictor
        except ImportError:
            # Running from the workspace root rather than the full Ultralytics
            # fork: use the checked-in overlay modules as a fallback.
            try:
                if args.predictor == "btq":
                    from legacy_experiments_20260923.btq_runtime import BTQSegmentationPredictor

                    predictor = _build_quality_btq_predictor(BTQSegmentationPredictor)
                else:
                    from legacy_experiments_20260923.harm_runtime import HARMSegmentationPredictor

                    predictor = HARMSegmentationPredictor
            except ImportError:
                if args.predictor == "btq":
                    from mpts_work.btq_runtime import BTQSegmentationPredictor

                    predictor = _build_quality_btq_predictor(BTQSegmentationPredictor)
                else:
                    from mpts_work.harm_runtime import HARMSegmentationPredictor

                    predictor = HARMSegmentationPredictor

    model = YOLO(args.weights, task="segment")
    config = AgentConfig(
            accept_quality=args.accept_quality,
            review_quality=args.review_quality,
            min_edge_alignment=args.min_edge_alignment,
            min_peak_distance=args.min_peak_distance,
            min_peak_value=args.min_peak_value,
            split_merge_risk=args.split_merge_risk,
            max_iterations=args.max_iterations,
            enable_boundary_tool=not args.disable_boundary,
            enable_split_tool=not args.disable_split,
            enable_merge_tool=not args.disable_merge,
            enable_simplify_tool=not args.disable_simplify,
            max_contour_vertices=args.max_contour_vertices,
            max_split_parts=args.max_split_parts,
            max_local_calls=args.max_local_calls,
            max_seconds=args.max_seconds,
        )
    overrides = {
        "stream": False,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "retina_masks": True,
        "save": False,
        "verbose": False,
    }
    if predictor is not None:
        overrides["predictor"] = predictor
    if args.predictor == "btq":
        overrides.update(
            {
                "btq_box_nms_iou": 0.95,
                "btq_mask_nms_iou": 0.70,
                "btq_mask_nms_candidates": 1000,
            }
        )
    if args.device is not None:
        overrides["device"] = args.device

    def predict_crop(crop, imgsz, conf):
        local_args = dict(overrides, imgsz=imgsz, conf=conf)
        return parcels_from_ultralytics(model.predict(source=crop, **local_args)[0])

    agent = FieldBqcAgent(config, local_tool=None if args.disable_local else LocalReinfer(predict_crop, config), router=router)
    run_config = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "agent_config": asdict(config), "policies": {key: value.to_dict() for key, value in router.policies.items()},
                  "sources": [str(path) for path in sources], "thresholds_calibrated": False}
    (output_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    count = 0
    # Finish each prediction before invoking the same model on an ROI. Nested
    # calls into an active stream would reuse predictor state and can deadlock.
    for count, source in enumerate(sources, start=1):
        inference_started = time.perf_counter()
        result = model.predict(source=str(source), **overrides)[0]
        initial_inference_seconds = time.perf_counter() - inference_started
        image_path = str(getattr(result, "path", f"image_{count:06d}"))
        stem = _safe_stem(image_path, count)
        parcels = parcels_from_ultralytics(result)
        supplied = dict(metadata_index.get(source.stem, {}))
        for key, value in (("sensor", args.sensor), ("resolution", args.resolution),
                           ("domain", args.domain), ("scene_id", args.scene_id)):
            if value not in (None, "auto") and (key == "domain" or value != "unknown"):
                supplied[key] = value
                if key == "resolution":
                    supplied["resolution_m"] = value
        metadata = image_metadata(image_path, result.orig_shape[1], result.orig_shape[0], supplied)
        metadata["quality_source"] = "mqc" if any(parcel.quality is not None for parcel in parcels) else "unavailable"
        save_parcels(original_root / f"{stem}.npz", parcels, result.orig_shape)
        output = agent.run(result.orig_img, parcels, metadata=metadata)
        save_parcels(mask_root / f"{stem}.npz", output.parcels, result.orig_shape)
        trace = _serializable_trace(output, image_path, metadata)
        trace["initial_inference_seconds"] = initial_inference_seconds
        (trace_root / f"{stem}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            f"[{count}] {image_path} instances={len(output.parcels)} "
            f"accepted={output.accepted} review={output.needs_review} "
            f"quality={metadata['quality_source']}", flush=True,
        )
    if args.fuse_tiles:
        from fuse_fieldbqc_tiles import run as run_fusion
        run_fusion(argparse.Namespace(input=output_root, output=output_root / "fused", coordinate_mode="auto",
                   seam_band=8, seam_iou=0.70, overlap_iou=0.50, max_gap=2, min_quality=args.review_quality,
                   min_score=args.conf, min_mask_quality=args.review_quality, no_gate_fallback=True,
                   verified=True, match_margin=0.10))
    print(f"Saved FieldBQC-Agent outputs to {output_root.resolve()} ({count} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
