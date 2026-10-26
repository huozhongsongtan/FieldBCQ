"""One inference pass for native mask AP and the requested fixed-threshold metrics."""

from pathlib import Path

from paper_data import load_targets, square_evaluation_masks
from paper_metrics import aggregate, image_metrics


def metric_scope(tables, expected, mode):
    scope = set()
    if 5 in tables and mode != "class":
        scope.update(("boundary", "structure", "calibration"))
    if 6 in tables and expected == "fieldbcq":
        scope.update(("boundary", "structure", "calibration", "small"))
    if mode != "class" and expected in {"baseline", "fieldbcq"}:
        if 7 in tables:
            scope.update(("boundary", "small", "area"))
        if 8 in tables:
            scope.add("boundary")
    fixed = bool(scope) or (9 in tables and expected in {"baseline", "fieldbcq"} and mode != "class")
    native = 4 in tables or (6 in tables and expected == "fieldbcq")
    return scope, fixed, native


def validator_class(records, labels, size, fixed, scope=None, native=True):
    import numpy as np
    import torch
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import ap_per_class

    class PaperValidator(SegmentationValidator):
        def get_desc(self):
            return "Paper mask evaluation"

        def init_metrics(self, model):
            super().init_metrics(model)
            self.args.save_json = False
            self.stats = dict(tp_m=[], conf=[], pred_cls=[], target_cls=[])

        def get_dataloader(self, dataset_path, batch_size):
            loader = super().get_dataloader(dataset_path, batch_size)
            actual = {Path(p).resolve() for p in loader.dataset.im_files}
            if actual != set(labels) or len(loader.dataset) != len(labels):
                raise ValueError("Validator dropped/changed test images; repair corrupt data/labels first")
            return loader

        def update_metrics(self, preds, batch):
            for si, (pred, proto) in enumerate(zip(*preds)):
                self.seen += 1
                prepared = self._prepare_batch(si, batch)
                cls, boxes = prepared.pop("cls"), prepared.pop("bbox")
                gt_masks = prepared.pop("masks")
                correct = torch.zeros(len(pred), self.niou, dtype=torch.bool, device=self.device)
                if native and len(pred):
                    predn, low_masks = self._prepare_pred(pred, prepared, proto)
                    if len(cls):
                        correct = self._process_batch(predn, boxes, cls, low_masks, gt_masks,
                                                      self.args.overlap_mask, masks=True)
                if native:
                    for key, value in dict(tp_m=correct, conf=pred[:, 4], pred_cls=pred[:, 5], target_cls=cls).items():
                        self.stats[key].append(value)
                if fixed:
                    path = Path(batch["im_file"][si]).resolve()
                    if path in records:
                        raise ValueError(f"Duplicate evaluated image: {path}")
                    gt, gc, _ = load_targets(labels[path], prepared["ori_shape"], size)
                    keep = pred[:, 4] >= 0.25
                    retained = pred[keep]
                    if len(retained):
                        masks = ops.process_mask(proto, retained[:, 6:], retained[:, :4],
                                                 shape=prepared["imgsz"], upsample=True).cpu().numpy().astype(bool)
                        masks = square_evaluation_masks(masks, prepared["ori_shape"], size)
                    else:
                        masks = np.zeros((0, size, size), dtype=bool)
                    records[path] = image_metrics(masks, gt, retained[:, 4].cpu().numpy(),
                                                  retained[:, 5].cpu().numpy(), gc, scope=scope)

        def get_stats(self):
            if not native:
                return self.metrics.results_dict
            stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
            # Native segmentation AP, including all empty/negative images; no box AP calculation.
            if stats["tp_m"].any():
                values = ap_per_class(stats["tp_m"], stats["conf"], stats["pred_cls"],
                                      stats["target_cls"], plot=False, names=self.names)[2:]
                self.metrics.seg.nc = self.nc
                self.metrics.seg.update(values)
            return self.metrics.results_dict

        def print_results(self):
            pass

    return PaperValidator


def evaluate(weights, expected, args, data, labels, output, mode=None):
    import gc
    import torch
    from ultralytics import YOLO
    from paper_runtime import configuration_of, sha256
    from quality_scoring import set_quality_scoring

    model = YOLO(str(weights))
    actual = configuration_of(model.model)
    if actual != expected:
        raise ValueError(f"Checkpoint configuration mismatch: expected {expected}, got {actual}: {weights}")
    scoring = set_quality_scoring(model.model, mode)
    seed = (model.ckpt or {}).get("train_args", {}).get("seed")
    records = {}
    scope, fixed, native = metric_scope(args.tables, expected, mode)
    result = model.val(validator=validator_class(records, labels, args.imgsz, fixed, scope, native),
                       data=str(data), split="test", device=args.device, imgsz=args.imgsz, batch=args.batch,
                       workers=args.workers, half=False, conf=0.001, iou=0.70, max_det=710,
                       rect=True, augment=False, plots=False, save=False, save_json=False, save_txt=False,
                       save_hybrid=False, single_cls=False, agnostic_nms=False, classes=None,
                       fraction=1.0, val_fraction=1.0, overlap_mask=True, mask_ratio=4,
                       project=str(output), name="native", exist_ok=True, verbose=False)
    m = result.seg
    p, r = float(m.mp), float(m.mr)
    report = dict(native=dict(P=p, R=r, F1=2*p*r/(p+r) if p+r else 0.,
                              AP50=float(m.map50), AP75=float(m.map75), AP50_95=float(m.map)),
                  checkpoint=str(weights), checkpoint_sha256=sha256(weights), seed=seed,
                  configuration=actual, **scoring)
    if fixed:
        if set(records) != set(labels):
            raise ValueError("Fixed-threshold results do not cover the complete test set")
        report["fixed"] = aggregate(list(records.values()))
    del model, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report, records
