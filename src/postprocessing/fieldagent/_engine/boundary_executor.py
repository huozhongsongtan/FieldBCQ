"""Shared, audited executor for the expanded boundary tools (no GT or GPT).

Separate simplification and realignment gates; thresholds are development
settings, not a claim of semantic correctness or industrial qualification.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import time
import numpy as np
import shapely
from cvr_agent import Audit, dump, save_items, scene_metrics, state_key, sha
from fieldbqc_geometry import GeometryConfig, GeometryTools, geometry_mask, holes, polygons
from fieldbqc_vertex_optimization import anchor_reasons
from boundary_tools import BoundaryToolbox, REGISTRY, VERSION, catalog, parameters


@dataclass(frozen=True)
class BoundaryConfig:
    budget: int = 64
    seconds: float = 120.
    simplify_max_shift: float = 2.
    simplify_max_area_change: float = .03
    simplify_min_iou: float = .98
    simplify_edge_drop: float = .01
    realign_max_shift: float = 4.
    realign_max_area_change: float = .12
    realign_min_support_gain: float = .01

    def __post_init__(self):
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError('Budget must be positive integer')
        for k, v in asdict(self).items():
            if type(v) not in (int, float) or not np.isfinite(v) or v <= 0:
                raise ValueError('Invalid config: '+k)
        for k in ('simplify_max_area_change', 'simplify_min_iou', 'simplify_edge_drop',
                  'realign_max_area_change', 'realign_min_support_gain'):
            if getattr(self, k) > 1:
                raise ValueError('Fraction outside (0,1]: '+k)


class BoundaryExecutor:
    def __init__(self, image, items, root, config=None):
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError('Output directory must be new or empty')
        self.config = config or BoundaryConfig()
        if len({p.identifier for p in items}) != len(items) or not items:
            raise ValueError('Nonempty scene with distinct IDs required')
        self.original = list(items); self.current = list(items); self.simplify_anchors = list(items)
        self.box = BoundaryToolbox(image)
        c = self.config
        self.geometry = GeometryTools(image, GeometryConfig(max_boundary_shift=c.simplify_max_shift,
            max_area_change=c.simplify_max_area_change, min_raster_iou=c.simplify_min_iou,
            max_edge_regression=c.simplify_edge_drop, min_vertices=3, target_vertices=3))
        self.audit = Audit(self.root); self.history = []; self.tried = set(); self.started = time.monotonic()
        self.baseline = scene_metrics(items)
        dump(self.root/'tool_catalog.json', dict(version=VERSION, tools=catalog(),
            config=asdict(c), runtime_gt_access=False, model_calls=0,
            image_support='Oriented RGB-grayscale gradient proxy; not semantic truth',
            audit_scope='Actual dispatched methods/tool/verification/export boundaries; not every library primitive',
            source_sha256={str(Path(p).resolve()): sha(p) for p in [__file__, Path(__file__).with_name('boundary_tools.py')]}))
        save_items(self.root/'baseline.pixel.geojson', self.original)

    @property
    def state_id(self):
        return state_key(self.current)

    def _common(self, candidate, ids):
        reasons = []; metrics = {}
        old = {p.identifier: p for p in self.current}; new = {p.identifier: p for p in candidate}
        if [p.identifier for p in self.current] != [p.identifier for p in candidate]:
            return ['instance_identity_changed'], metrics
        for uid in old:
            a, b = old[uid], new[uid]
            if (a.score, a.cls, a.source_ids) != (b.score, b.cls, b.source_ids):
                reasons.append('prediction_metadata_changed')
            g = b.geometry
            if g.is_empty or g.geom_type not in ('Polygon', 'MultiPolygon') or not g.is_valid:
                reasons.append('empty_or_invalid_geometry')
            if not np.isfinite(shapely.get_coordinates(g)).all():
                reasons.append('nonfinite_coordinate')
            if uid not in ids and not a.geometry.equals_exact(g, 0):
                reasons.append('unrelated_instance_changed')
        if reasons:
            return sorted(set(reasons)), metrics
        h, w = self.box.shape
        for uid in ids:
            g = new[uid].geometry
            x0, y0, x1, y1 = g.bounds
            if x0 < -1e-7 or y0 < -1e-7 or x1 > w+1e-7 or y1 > h+1e-7:
                reasons.append('geometry_outside_image')
        return sorted(set(reasons)), metrics

    def verify(self, candidate, ids, task):
        reasons, metrics = self._common(candidate, ids)
        if reasons:
            return dict(accept=False, reasons=reasons, metrics=metrics)
        if task == 'simplify':
            verdict = self.geometry.verify(self.current, candidate, 'simplify_contour', ids)
            extra = anchor_reasons(self.geometry, self.simplify_anchors, candidate, ids)
            original = {p.identifier: p.geometry for p in self.original}
            for p in candidate:
                if p.identifier in ids and original[p.identifier].boundary.hausdorff_distance(p.geometry.boundary) > self.config.realign_max_shift+1e-7:
                    extra.append('cumulative_total_boundary_drift')
            reasons = ([] if verdict['accept'] else verdict['reasons'])+extra
            return dict(accept=not reasons, reasons=sorted(set(reasons)), metrics=verdict['metrics'])
        old = {p.identifier: p for p in self.current}; new = {p.identifier: p for p in candidate}
        anchors = {p.identifier: p for p in self.original}
        metrics['targets'] = {}
        for uid in ids:
            a, b, origin = old[uid].geometry, new[uid].geometry, anchors[uid].geometry
            if task == 'legalize':
                if a.is_valid and not a.is_empty:
                    reasons.append('no_invalid_geometry_to_repair')
                continue
            if len(polygons(a)) != len(polygons(b)) or holes(a) != holes(b):
                reasons.append('ring_or_component_structure_changed')
            if not a.equals(b) and a.symmetric_difference(b).area <= 1e-8:
                reasons.append('no_positional_change')
            if a.equals(b):
                reasons.append('unchanged')
            shift = float(origin.boundary.hausdorff_distance(b.boundary))
            area = abs(b.area-origin.area)/max(origin.area, 1.)
            before_support, after_support = self.box.evidence(a), self.box.evidence(b)
            metrics['targets'][uid] = dict(cumulative_shift=shift, cumulative_area_change=area,
                support_before=before_support, support_after=after_support,
                support_gain=after_support-before_support,
                vertex_change=sum(len(r.coords)-1 for p in polygons(b) for r in [p.exterior, *p.interiors])-
                              sum(len(r.coords)-1 for p in polygons(a) for r in [p.exterior, *p.interiors]))
            if shift > self.config.realign_max_shift+1e-7:
                reasons.append('cumulative_realign_drift')
            if area > self.config.realign_max_area_change:
                reasons.append('cumulative_realign_area')
            if after_support-before_support < self.config.realign_min_support_gain:
                reasons.append('insufficient_image_support_gain')
        # Local pair checks protect neighbours even if total overlap decreases.
        for i, a in enumerate(self.current):
            for b in self.current[i+1:]:
                if not {a.identifier, b.identifier} & set(ids):
                    continue
                a0, b0 = a.geometry, b.geometry
                a1, b1 = new[a.identifier].geometry, new[b.identifier].geometry
                # Explicit make_valid may change polygonal structure; normalized
                # source is used only for safety measurement, never hidden output.
                if task == 'legalize':
                    a0, b0 = shapely.make_valid(a0), shapely.make_valid(b0)
                if a1.intersection(b1).area > a0.intersection(b0).area+1e-7:
                    reasons.append('new_neighbor_overlap')
                shared = a0.boundary.intersection(b0.boundary)
                if shared.length <= 1e-7 and a0.intersection(b0).area <= 1e-7 and a1.boundary.intersection(b1.boundary).length > 1e-7:
                    reasons.append('new_adjacency')
                if shared.length > 1e-7:
                    shared_after = a1.boundary.intersection(b1.boundary)
                    if shared.difference(shared_after.buffer(1e-7)).length > 1e-6:
                        reasons.append('shared_boundary_changed_without_joint_reconstruction')
        return dict(accept=not reasons, reasons=sorted(set(reasons)), metrics=metrics)

    def attempt(self, tool, ids, values=None, *, expected_state, source='tool_api'):
        if expected_state != self.state_id:
            raise ValueError('Stale state; observe current geometry again')
        if len(self.history) >= self.config.budget or time.monotonic()-self.started >= self.config.seconds:
            raise RuntimeError('Candidate or time budget exhausted')
        cfg = parameters(tool, values)
        import json
        key = (self.state_id, tool, tuple(ids), json.dumps(cfg, sort_keys=True))
        if key in self.tried:
            raise ValueError('Repeated candidate in identical state')
        self.tried.add(key)
        tid = f't{len(self.history):05d}'; folder = self.root/'candidates'/tid
        base_state = self.state_id; began = time.monotonic(); candidate = None; info = None
        with self.audit.call('boundary_candidate', dict(tool=tool, ids=list(ids), parameters=cfg,
                state_id=base_state, source=source, tool_version=VERSION), ['candidate_dispatch']) as event:
            try:
                with self.audit.call('generate_candidate', methods=list(REGISTRY[tool].methods)) as detail:
                    candidate, info = self.box.propose(tool, self.current, ids, cfg)
                    detail.update(info)
                with self.audit.call('typed_boundary_verifier', dict(task=REGISTRY[tool].task),
                        ['common_scene_safety', 'typed_action_gate', 'cumulative_origin_guard']) as detail:
                    verdict = self.verify(candidate, list(ids), REGISTRY[tool].task)
                    detail.update(verdict)
            except Exception as error:
                verdict = dict(accept=False, reasons=['tool_error'], error=f'{type(error).__name__}: {error}', metrics={})
            candidate_file = save_items(folder/'attempted.pixel.geojson', candidate) if candidate is not None else None
            before = scene_metrics(self.current)
            if verdict['accept']:
                with self.audit.call('commit_boundary_candidate', dict(trial_id=tid), ['verified_atomic_commit']) as detail:
                    self.current = list(candidate)
                    if REGISTRY[tool].task in ('realign', 'legalize'):
                        lookup = {p.identifier: p for p in self.current}
                        self.simplify_anchors = [lookup[p.identifier] if p.identifier in ids else p for p in self.simplify_anchors]
                    detail['checkpoint'] = save_items(self.root/'checkpoints'/(tid+'.pixel.geojson'), self.current)
            row = dict(trial_id=tid, base_state=base_state, tool=tool, target_ids=list(ids), parameters=cfg,
                source=source, proposal=info, committed=verdict['accept'], verdict=verdict,
                candidate_file=candidate_file, before=before, after=scene_metrics(self.current),
                seconds=time.monotonic()-began, output_state=self.state_id)
            dump(folder/'result.json', row); event.update(trial_id=tid, committed=verdict['accept'], reasons=verdict['reasons'])
        self.history.append(row)
        return row

    def finish(self, stop='finished'):
        with self.audit.call('export_boundary_result', methods=['GeoJSON serialization', 'rasterio.rasterize', 'np.savez_compressed']) as event:
            vector = save_items(self.root/'final/prediction.repaired.pixel.geojson', self.current)
            path = self.root/'final/prediction.repaired_masks.npz'
            masks = np.stack([geometry_mask(p.geometry, self.box.shape) for p in self.current]).astype(np.uint8)
            np.savez_compressed(path, masks=masks, parcel_ids=np.array([p.identifier for p in self.current]),
                scores=np.array([p.score for p in self.current]), classes=np.array([p.cls for p in self.current]),
                quality_valid=np.zeros(len(self.current), dtype=bool),
                quality_sources=np.array(['boundary_proxy_verified_not_GT']*len(self.current)))
            event.update(vector=vector, masks=dict(path=str(path), sha256=sha(path)))
        trace = dict(version=VERSION, baseline=self.baseline, final=scene_metrics(self.current),
            history=self.history, candidates=len(self.history), commits=sum(r['committed'] for r in self.history),
            model_calls=0, runtime_gt_access=False, semantic_accuracy_verified=False,
            stop_reason=stop, seconds=time.monotonic()-self.started,
            limitations=['Image support is proxy only', 'Shared-edge movement requires a future joint constructor',
                          'No GPT controller connected to this new API yet'])
        dump(self.root/'trace.json', trace)
        dump(self.root/'methods.json', dict(tool_version=VERSION, actual_method_calls=dict(self.audit.methods),
            source_sha256={str(Path(p).resolve()): sha(p) for p in [__file__, Path(__file__).with_name('boundary_tools.py')]}))
        return trace
