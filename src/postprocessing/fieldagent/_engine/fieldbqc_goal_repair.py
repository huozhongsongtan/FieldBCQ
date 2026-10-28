"""V9 two-task goal-directed local repair. Runtime never reads reference labels.

The model requests evidence, discovers registered local capabilities, creates
parameterized trials and commits/reverts them. No precomputed candidate-ID menu.
This is A1-GOAL-LOCAL, not the planned open-web discovery system.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
import time

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from fieldbqc_geometry import (GeometryTools, holes, normalized, polygons,
                              scene_metrics, to_parcels, vertices)
from fieldbqc_planner import APIPlanner, visual_inputs
from fieldbqc_selective_repair import RepairConfig, regularize_geometry, state_key
from fieldbqc_vertex_optimization import anchor_reasons

VERSION = "goal-two-task-local-v9.2"


@dataclass(frozen=True)
class GoalConfig:
    model_calls: int = 6
    tool_budget: int = 192
    seconds: float = 360.0
    max_commands: int = 8
    max_boundary_shift: float = 2.0
    max_area_change: float = .05
    min_raster_iou: float = .95
    max_edge_regression: float = .03
    duplicate_containment: float = .95
    max_suppression_union_loss: float = .02
    min_clip_retained_fraction: float = .15
    min_clip_pixels: int = 16

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {key}")
        for key in ("model_calls", "tool_budget", "max_commands", "min_clip_pixels"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"Expected integer {key}")
        for key in ("max_area_change", "min_raster_iou", "max_edge_regression",
                    "duplicate_containment", "max_suppression_union_loss", "min_clip_retained_fraction"):
            if getattr(self, key) > 1:
                raise ValueError(f"Fraction > 1: {key}")


TOOL_CARDS = [
    dict(tool_id="contour_dp", task="simplify_contour", source="local:shapely.simplify",
         description="Douglas Peucker, remove redundant vertices within tolerance; preserve topology per polygon.",
         parameters="target_ids, tolerance in (0, max_boundary_shift]; each ID independently verified",
         effects="No identity/count changes; holes, components, neighbors and cumulative drift protected."),
    dict(tool_id="contour_vw", task="simplify_contour", source="local:shapely.coverage_simplify",
         description="Visvalingam Whyatt coverage simplification, effective triangle area; joint shared boundary supported.",
         parameters="target_ids, tolerance in (0, max_boundary_shift]",
         effects="Multiple targets require valid noded nonoverlapping coverage; no identity/count changes."),
    dict(tool_id="contour_lines", task="simplify_contour", source="local:regularize_geometry",
         description="Fit merge and assemble near straight boundary line segments; protect real corners and curves.",
         parameters="target_ids, tolerance in (0, max_boundary_shift]",
         effects="Deterministic line fitting, not a B2PNet network reproduction; per-ID verification."),
    dict(tool_id="overlap_allocate", task="resolve_overlap", source="local:GeometryTools._allocate",
         description="Allocate disputed pixels to exclusive instance anchors using distance; keep all real parcels.",
         parameters="target_ids of overlapping instances; no tolerance or keep_id",
         effects="Changes only disputed cells; needs an exclusive anchor for each target; preserves union."),
    dict(tool_id="overlap_clip", task="resolve_overlap", source="local:shapely.difference",
         description="Keep one parcel and subtract its overlap from the other distinct parcel.",
         parameters="exactly two target_ids, keep_id; no tolerance",
         effects="Preserves pair union and non-disputed coverage; reject empty, newly fragmented or collapsed survivors. Components wholly in disputed region may disappear."),
    dict(tool_id="overlap_suppress", task="resolve_overlap", source="local:duplicate_suppression",
         description="Deactivate a duplicate detection of the same real parcel; retain its better representative.",
         parameters="exactly two target_ids, keep_id; no tolerance",
         effects="Requires high containment and low pair union loss; these are geometric gates NOT proof of duplicate identity. Logs lineage."),
]


def goal_contract(config):
    return dict(goal_id=VERSION, tasks=["simplify_contour", "resolve_overlap"],
                priority=["preserve_real_parcels_and_structures", "correct_conflict_ownership",
                          "reduce_redundant_vertices", "control_cost"],
                quality_constraints=asdict(config), fixed_vertex_target=None,
                runtime_ground_truth=False, scope="A1: local capability discovery only; no online discovery",
                stop="Keep best valid state when no evidence-backed gain remains or budget exhausted",
                uncertainty="keep or review; numeric validity does not certify semantic correctness")


def overlap_pairs(items):
    if not items:
        return []
    geoms = [p.geometry for p in items]
    tree = STRtree(geoms)
    result = []
    for i, p in enumerate(items):
        for j in sorted(int(x) for x in tree.query(p.geometry)):
            if j <= i:
                continue
            q = items[j]
            area = p.geometry.intersection(q.geometry).area
            if area <= 1e-8:
                continue
            result.append(dict(target_ids=[p.identifier, q.identifier], overlap_area=area,
                               containment=area / max(1e-9, min(p.geometry.area, q.geometry.area)),
                               pair_iou=area / max(1e-9, p.geometry.union(q.geometry).area),
                               fraction_by_id={p.identifier: area/max(1e-9,p.geometry.area),
                                               q.identifier: area/max(1e-9,q.geometry.area)}))
    return sorted(result, key=lambda p: (-p["overlap_area"], p["target_ids"]))


def conflict_components(items):
    groups = []
    for pair in overlap_pairs(items):
        merged = set(pair["target_ids"])
        keep = []
        for group in groups:
            if group & merged:
                merged |= group
            else:
                keep.append(group)
        groups = keep + [merged]
    return [sorted(g) for g in sorted(groups, key=lambda g: sorted(g))]


class GoalEngine:
    def __init__(self, image, original, config=None):
        self.config = config or GoalConfig()
        cfg = RepairConfig(max_boundary_shift=self.config.max_boundary_shift,
                           max_area_change=self.config.max_area_change,
                           min_raster_iou=self.config.min_raster_iou,
                           max_edge_regression=self.config.max_edge_regression)
        self.tools = GeometryTools(image, cfg.geometry_config())
        self.raw = list(original)
        self.initial = [p.with_geometry(normalized(p.geometry)) for p in original]
        if any(p.geometry.is_empty or not p.geometry.is_valid for p in self.initial):
            raise ValueError("Common legalization did not produce nonempty valid faces")
        if len({p.identifier for p in self.initial}) != len(self.initial):
            raise ValueError("Duplicate original IDs")
        self.current = list(self.initial)
        self.anchors = list(self.initial)
        self.trials = {}
        self.generated = 0
        self.history = []
        self.lineage = []
        self.reviewed = set()
        self.checkpoints = []
        self.started = time.monotonic()
        self.deadline = self.started + self.config.seconds

    def remaining(self):
        return max(0., self.deadline-time.monotonic())

    def inspect(self, ids=()):
        selected = set(ids) if ids else {p.identifier for p in self.current}
        active = {p.identifier for p in self.current}
        if not selected <= active:
            raise ValueError("Unknown/inactive inspection ID")
        records = []
        for index, p in enumerate(self.current):
            if p.identifier not in selected:
                continue
            records.append(dict(parcel_id=p.identifier, index=index, severity=vertices(p.geometry),
                                vertices=vertices(p.geometry), area=p.geometry.area,
                                holes=holes(p.geometry), components=len(polygons(p.geometry)),
                                bbox=list(p.geometry.bounds), confidence=p.score, class_id=p.cls,
                                edge_support=self.tools.edge_alignment(self.tools.mask(p))))
        pairs = overlap_pairs(self.current)
        return dict(instances=records, scene=scene_metrics(self.current),
                    conflicts=[p for p in pairs if set(p["target_ids"]) & selected],
                    conflict_components=conflict_components(self.current),
                    coordinate_units="pixel cells; x right, y down", ground_truth_available=False)

    def find_tools(self, query):
        words = set(query.lower().replace("_", " ").split())
        ranked = sorted(TOOL_CARDS, key=lambda c: (-len(words & set(json.dumps(c).lower().replace("_", " ").split())), c["tool_id"]))
        return dict(source="registered_local_capabilities", query=query, tools=ranked,
                    implementation_version=VERSION, shapely_version=shapely.__version__)

    def _reserve(self):
        if self.generated >= self.config.tool_budget:
            raise RuntimeError("tool_budget_exhausted")
        if self.remaining() <= 0:
            raise RuntimeError("time_budget_exhausted")
        self.generated += 1

    def _simplify(self, items, ids, tool, tolerance):
        lookup = {p.identifier: p for p in items}
        geoms = [lookup[i].geometry for i in ids]
        if tool == "contour_dp":
            geoms = [g.simplify(tolerance, preserve_topology=True) for g in geoms]
        elif tool == "contour_lines":
            geoms = [regularize_geometry(g, tolerance) for g in geoms]
        else:
            array = np.asarray(geoms, dtype=object)
            if len(ids) > 1:
                reference = shapely.union_all(shapely.boundary(array))
                array = np.asarray([shapely.snap(g, reference, 1e-8) for g in array])
                if not shapely.coverage_is_valid(array):
                    raise ValueError("VW joint simplification requires valid noded coverage")
            geoms = shapely.coverage_simplify(array, tolerance, simplify_boundary=True)
        replacements = dict(zip(ids, geoms))
        candidate = [p.with_geometry(replacements[p.identifier]) if p.identifier in replacements else p for p in items]
        verdict = self.tools.verify(items, candidate, "simplify_contour", ids)
        reasons = [] if verdict["accept"] else list(verdict["reasons"])
        reasons += anchor_reasons(self.tools, self.anchors, candidate, ids)
        verdict.update(accept=not reasons, reasons=sorted(set(reasons)))
        return candidate, verdict

    def _conflict(self, items, ids, tool, keep_id):
        old = {p.identifier: p for p in items}
        targets = [old[i] for i in ids]
        pairs = overlap_pairs(targets)
        if not pairs or len({p.cls for p in targets}) != 1:
            raise ValueError("Needs overlapping targets of the same class")
        occupied = sum((self.tools.mask(p).astype(np.int32) for p in targets), np.zeros(self.tools.shape,np.int32))
        removed = None
        if tool == "overlap_allocate":
            replacements = {p.identifier:p for p in self.tools._allocate(targets, self.tools.mask(targets[0]) | (occupied>0), overlap_only=True)}
            candidate = [replacements.get(p.identifier,p) for p in items]
        else:
            if len(ids) != 2 or keep_id not in ids:
                raise ValueError("Clip/suppress requires two targets and one of them as keep_id")
            kept = old[keep_id]
            other = next(p for p in targets if p.identifier != keep_id)
            if tool == "overlap_clip":
                clipped = other.geometry.difference(kept.geometry)
                candidate = [p.with_geometry(clipped) if p.identifier == other.identifier else p for p in items]
            else:
                if pairs[0]["containment"] < self.config.duplicate_containment:
                    raise ValueError("Insufficient geometric containment for suppression")
                removed = other.identifier
                candidate = [p for p in items if p.identifier != removed]
        return candidate, self.verify_conflict_candidate(items, candidate, ids, removed, keep_id)

    def verify_conflict_candidate(self, items, candidate, ids, removed=None, keep_id=None):
        """Common ownership gates, also used for model-authored candidates.

        Candidate generation and validation are separate; extending an algorithm
        must not extend its authority to change identities or exclusive cells.
        """
        old = {p.identifier:p for p in items}
        targets = [old[i] for i in ids]
        if not overlap_pairs(targets) or len({p.cls for p in targets}) != 1:
            raise ValueError("Needs overlapping targets of the same class")
        before = scene_metrics(items)
        union0 = unary_union([p.geometry for p in targets])
        occupied = sum((self.tools.mask(p).astype(np.int32) for p in targets), np.zeros(self.tools.shape,np.int32))
        exclusive = {p.identifier: self.tools.mask(p) & (occupied == 1) for p in targets}
        effects = {}
        new = {p.identifier:p for p in candidate}
        reasons = []
        expected = set(old) - ({removed} if removed else set())
        if set(new) != expected:
            reasons.append("unexpected_identity_change")
        for uid in set(old) - set(ids):
            if not old[uid].geometry.equals_exact(new[uid].geometry,0):
                reasons.append("unrelated_instance_changed")
        for uid in set(ids) & set(new):
            p, q = old[uid], new[uid]
            g = q.geometry
            if g.is_empty or not g.is_valid or g.geom_type not in {"Polygon","MultiPolygon"}:
                reasons.append("empty_or_invalid_target")
                continue
            # Conflict ownership can legitimately remove a component wholly
            # inside the disputed region, or expose a hole belonging to a
            # distinct retained field. Subset and exclusive-cell checks protect
            # prior holes and all non-disputed material. Reject NEW fragmentation.
            if len(polygons(g)) > len(polygons(p.geometry)):
                reasons.append("new_target_fragmentation")
            if g.difference(p.geometry).area > 1e-7:
                reasons.append("conflict_operation_expanded_instance")
            fraction = g.area/max(1e-9,p.geometry.area)
            if fraction < self.config.min_clip_retained_fraction or self.tools.mask(q).sum() < self.config.min_clip_pixels:
                reasons.append("survivor_collapsed")
            lost_exclusive = int(np.count_nonzero(exclusive[uid] & ~self.tools.mask(q)))
            if lost_exclusive:
                reasons.append("non_disputed_pixels_removed")
            effects[uid] = dict(retained_fraction=fraction, exclusive_pixels_lost=lost_exclusive,
                                edge_before=self.tools.edge_alignment(self.tools.mask(p)),
                                edge_after=self.tools.edge_alignment(self.tools.mask(q)))
        union1 = unary_union([p.geometry for p in candidate if p.identifier in ids])
        pair_loss = union0.difference(union1).area/max(1e-9,union0.area)
        if removed:
            if pair_loss > self.config.max_suppression_union_loss:
                reasons.append("suppression_union_loss")
            effects[removed] = dict(deactivated=True, replaced_by=keep_id)
        elif union0.symmetric_difference(union1).area > 1e-7:
            reasons.append("pair_coverage_changed")
        after = scene_metrics(candidate)
        if after["invalid_geometries"]:
            reasons.append("invalid_geometry")
        if after["overlap_excess_area"] >= before["overlap_excess_area"]-1e-7:
            reasons.append("no_overlap_improvement")
        return dict(accept=not reasons, reasons=sorted(set(reasons)),
                               metrics=dict(before=before, candidate=after, targets=effects,
                                            pair_union_loss_fraction=pair_loss, removed_id=removed,
                                            numeric_checks_do_not_certify_identity=True))

    def trial(self, tool, ids, tolerance=None, keep_id=None):
        if tool not in {c["tool_id"] for c in TOOL_CARDS}:
            raise ValueError("Unknown tool; discover actual capabilities with find_tools")
        if self.generated >= self.config.tool_budget or self.remaining() <= 0:
            raise RuntimeError("tool_or_time_budget_exhausted")
        known = {p.identifier for p in self.current}
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= known or set(ids) & self.reviewed:
            raise ValueError("Targets must be unique active non-reviewed IDs")
        if tool.startswith("contour_"):
            if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or not 0 < tolerance <= self.config.max_boundary_shift:
                raise ValueError("Tolerance outside frozen simplification range")
            if keep_id is not None:
                raise ValueError("Simplification has no keep_id")
        elif tolerance is not None:
            raise ValueError("Overlap tools have no tolerance")
        candidate = list(self.current)
        details = []
        # DP/line batches execute independent per-ID validated edits. Every
        # individual attempt consumes budget and is reported, including rejects.
        groups = [ids] if tool.startswith("overlap_") or tool == "contour_vw" else [[i] for i in ids]
        # Single-ID VW is also available; multi-ID VW is a joint coverage trial.
        for group in groups:
            if self.generated >= self.config.tool_budget or self.remaining() <= 0:
                details.append(dict(target_ids=group,accept=False,reasons=["budget_exhausted"],metrics={}))
                break
            self._reserve()
            try:
                if tool.startswith("contour_"):
                    proposed, verdict = self._simplify(candidate,group,tool,tolerance)
                else:
                    proposed, verdict = self._conflict(candidate,group,tool,keep_id)
            except Exception as error:
                proposed = candidate
                verdict = dict(accept=False,reasons=["tool_error"],error=f"{type(error).__name__}: {error}",metrics={})
            details.append(dict(target_ids=group,**verdict))
            if verdict["accept"]:
                candidate = proposed
        accepted = any(v["accept"] for v in details)
        trial_id = f"t{len(self.trials):04d}"
        summary = dict(trial_id=trial_id, tool_id=tool, target_ids=ids, tolerance=tolerance, keep_id=keep_id,
                       base_state=state_key(self.current), accept=accepted,
                       before=scene_metrics(self.current), candidate=scene_metrics(candidate), details=details)
        self.trials[trial_id] = (summary,candidate)
        return summary

    def commit(self, trial_id, reason):
        if trial_id not in self.trials:
            raise ValueError("Unknown trial_id")
        record, candidate = self.trials[trial_id]
        if record["base_state"] != state_key(self.current):
            raise ValueError("Stale trial; current state changed, regenerate trial")
        if not record["accept"]:
            raise ValueError("Rejected trial cannot be committed")
        self.checkpoints.append((list(self.current),list(self.anchors),deepcopy(self.lineage)))
        self.current = list(candidate)
        if record["tool_id"].startswith("overlap_"):
            changed = set(record["target_ids"])
            # A declared conflict edit establishes the new per-ID shape anchor;
            # later simplification cannot accumulate unlimited extra drift.
            anchors = {p.identifier:p for p in self.anchors}
            anchors.update({p.identifier:p for p in self.current if p.identifier in changed})
            self.anchors = [anchors[p.identifier] for p in self.current]
            for detail in record["details"]:
                removed = detail.get("metrics",{}).get("removed_id")
                if detail["accept"] and removed:
                    for previous in self.lineage:
                        if previous.get("replaced_by")==removed:
                            previous.setdefault("replacement_chain",[]).append(removed)
                            previous["replaced_by"]=record["keep_id"]
                    self.lineage.append(dict(original_id=removed,active=False,replaced_by=record["keep_id"],
                                             operation="overlap_suppress",trial_id=trial_id,reason=reason))
        return dict(committed=True,trial_id=trial_id,tool_id=record["tool_id"],reason=reason,
                    scene=scene_metrics(self.current))

    def restore(self):
        if not self.checkpoints:
            raise ValueError("No committed checkpoint to restore")
        self.current,self.anchors,self.lineage = self.checkpoints.pop()
        return dict(restored=True,scene=scene_metrics(self.current))


COMMAND = dict(type="object",additionalProperties=False,properties={
    "kind":dict(type="string",enum=["inspect","find_tools","trial","apply","commit","restore","review","stop"]),
    "tool_id":dict(type=["string","null"]),"target_ids":dict(type="array",items=dict(type="string")),
    "tolerance":dict(type=["number","null"]),"keep_id":dict(type=["string","null"]),
    "trial_id":dict(type=["string","null"]),"query":dict(type=["string","null"]),
    "reason":dict(type="string")},
    required=["kind","tool_id","target_ids","tolerance","keep_id","trial_id","query","reason"])
PACKET = dict(type="object",additionalProperties=False,properties={
    "diagnosis":dict(type="string"),"plan":dict(type="string"),
    "commands":dict(type="array",items=COMMAND)},required=["diagnosis","plan","commands"])

PROMPT = """You are a goal-directed agricultural parcel VECTOR optimization agent.
Follow goal_contract, not a fixed candidate menu. No ground truth is available.
Goals in order: preserve real fields and true structures; resolve ownership of
overlaps/duplicate detections correctly; remove redundant vertices within quality
limits; control cost. No fixed vertex target. Do NOT fill holes or split/merge fields.
Inspect raw imagery, complete labeled parcels, conflict crops, and trial feedback.
Different fields overlapping require ownership/clipping/allocation; true duplicate
detections may require suppressing one. High containment alone does NOT prove a
duplicate. Scores and geometric checks are uncertain evidence, not semantic labels.
Use discoverable local tools; find_tools returns real cards. This experiment has
NO online tool search. You choose target IDs, algorithm, numeric tolerance, order
and stopping. All registered tools and parameter limits are shared with the rule
baseline. You may inspect more context, trial alternatives, commit, restore, review.
Return supplied JSON: short observable diagnosis, executable plan and commands.
Each command has every schema field; irrelevant fields null, target_ids [].
inspect: target_ids (empty=all). find_tools: query. trial: tool_id, target_ids,
tolerance only for contour tools, keep_id only for clip/suppress. trial does NOT
modify the committed scene. commit: trial_id from prior feedback, reason. apply:
allowed ONLY for contour tools; trial and independently verified commit combined.
OVERLAP TOOLS REQUIRE trial THEN a subsequent response committing a known trial_id
after viewing feedback and preview. Never predict a future trial_id. All trials in
one packet share current state unless an apply/commit/restore changes it; such
changes stale prior trials. Commit at most one alternative from a given state.
DP and line tools accept batches and individually reject unsafe IDs; VW with >1
IDs is JOINT and requires valid non-overlapping noded coverage. Use VW separately
on IDs with existing overlaps, or resolve conflicts first. Tolerances continuous
within (0, max_boundary_shift]. Single-ID trial comparisons are allowed.
review: explicit uncertain IDs, retained in evaluation. stop: finish and keep current.
There is no mandatory final simplification. Plan enough coverage within remaining
calls and tool budget; a contour batch may include many IDs. Prefer tested simple
tools when sufficient, and stop rather than repeat rejected operations. API plans
do not themselves certify correctness; every operation is actually executed.
"""


def validate_packet(packet, context):
    if not isinstance(packet,dict) or set(packet) != {"diagnosis","plan","commands"}:
        raise ValueError("Expected diagnosis/plan/commands")
    if not all(isinstance(packet[k],str) and 1 <= len(packet[k]) <= 3000 for k in ("diagnosis","plan")):
        raise ValueError("Short diagnosis and plan required")
    commands = packet["commands"]
    if not isinstance(commands,list) or not 1 <= len(commands) <= context["max_commands"]:
        raise ValueError("Invalid command count")
    known = {p["parcel_id"] for p in context["instances"]}
    for c in commands:
        if not isinstance(c,dict) or set(c) != set(COMMAND["required"]):
            raise ValueError("Invalid command schema")
        if c["kind"] not in COMMAND["properties"]["kind"]["enum"]:
            raise ValueError("Unknown command")
        ids = c["target_ids"]
        if not isinstance(ids,list) or any(not isinstance(i,str) for i in ids) or len(ids)!=len(set(ids)) or not set(ids)<=known:
            raise ValueError("Unknown/duplicate target ID")
        if not isinstance(c["reason"],str) or not 1<=len(c["reason"])<=2000:
            raise ValueError("Observable reason required")
        for name in ("tool_id","keep_id","trial_id","query"):
            if c[name] is not None and not isinstance(c[name],str):
                raise ValueError(f"Invalid {name}")
        if c["tolerance"] is not None and (isinstance(c["tolerance"],bool) or not isinstance(c["tolerance"],(int,float)) or not math.isfinite(c["tolerance"])):
            raise ValueError("Invalid tolerance")
    return packet


class GoalPlanner(APIPlanner):
    def __init__(self, config, engine, cache):
        self.engine = engine
        self.focus_ids = []
        self.preview_trials = []
        super().__init__(config,cache,schema=PACKET,system_prompt=PROMPT,validator=validate_packet,
                         prompt_version=VERSION,visualizer=self.images)

    def images(self,image,parcels):
        urls = visual_inputs(image,parcels)
        # Stable full-scene context plus requested conflict crops and actual
        # generated trial previews. No reference image/geometry is ever used.
        lookup = {p.identifier:p for p in self.engine.current}
        groups = [self.focus_ids] if self.focus_ids else conflict_components(self.engine.current)[:4]
        panels = []
        for ids in groups:
            targets = [lookup[i] for i in ids if i in lookup]
            if not targets:
                continue
            bounds = unary_union([p.geometry for p in targets]).bounds
            x0,y0,x1,y1 = [int(v) for v in bounds]
            x0,y0=max(0,x0-8),max(0,y0-8)
            x1,y1=min(image.shape[1],x1+9),min(image.shape[0],y1+9)
            frames=[]
            for state,label in [(None,"raw"),(self.engine.current,"current")]+[(self.engine.trials[t][1],t) for t in self.preview_trials[-2:]]:
                crop=image[y0:y1,x0:x1].copy()
                if state is not None:
                    for k,p in enumerate(q for q in state if q.identifier in ids):
                        mask=self.engine.tools.mask(p)[y0:y1,x0:x1].astype(np.uint8)
                        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                        color=[(0,255,255),(255,0,255),(0,255,0),(255,160,0)][k%4]
                        cv2.drawContours(crop,contours,-1,color,1)
                        if contours:
                            x,y,_,_=cv2.boundingRect(max(contours,key=cv2.contourArea))
                            cv2.putText(crop,p.identifier,(x,max(10,y)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
                crop=cv2.resize(crop,(220,220),interpolation=cv2.INTER_AREA)
                crop=cv2.copyMakeBorder(crop,20,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
                cv2.putText(crop,label,(3,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                frames.append(crop)
            while len(frames)<4:
                frames.append(np.zeros_like(frames[0]))
            row=np.hstack(frames[:4])
            panels.append(row)
        if panels:
            urls.append(visual_inputs(np.vstack(panels),[],max_side=1600)[0])
        return urls


def run_rule(engine):
    """Strong declared rule: containment dedup; least-risk verified conflict;
    then per-ID best compression over 3 algorithms and 2 tolerances.
    """
    events=[]
    seen=set()
    while engine.remaining()>0 and engine.generated<engine.config.tool_budget:
        pairs=[p for p in overlap_pairs(engine.current) if tuple(p["target_ids"]) not in seen]
        if not pairs:
            break
        pair=pairs[0]
        ids=pair["target_ids"]
        seen.add(tuple(ids))
        lookup={p.identifier:p for p in engine.current}
        alternatives=[]
        specs=[("overlap_allocate",None),("overlap_clip",ids[0]),("overlap_clip",ids[1])]
        if pair["containment"]>=engine.config.duplicate_containment:
            specs=[("overlap_suppress",ids[0]),("overlap_suppress",ids[1])]+specs
        for tool,keep in specs:
            if engine.generated>=engine.config.tool_budget or engine.remaining()<=0:
                break
            record=engine.trial(tool,ids,keep_id=keep)
            events.append(dict(action="trial",**record))
            if record["accept"]:
                candidate=engine.trials[record["trial_id"]][1]
                new={p.identifier:p for p in candidate}
                removed=set(ids)-set(new)
                edge=sum(engine.tools.edge_alignment(engine.tools.mask(new[i]))-engine.tools.edge_alignment(engine.tools.mask(lookup[i])) for i in set(ids)&set(new))
                area_risk=sum(abs(new[i].geometry.area-lookup[i].geometry.area)/max(1,lookup[i].geometry.area) for i in set(ids)&set(new))
                # Favor a geometrically eligible duplicate removal; among those,
                # keep the representative with the strongest score/edge support.
                if removed:
                    score=10+lookup[keep].score+engine.tools.edge_alignment(engine.tools.mask(lookup[keep]))
                else:
                    score=edge-.2*area_risk
                alternatives.append((score,record["trial_id"]))
        if alternatives:
            _,chosen=max(alternatives,key=lambda r:(r[0],r[1]))
            events.append(dict(action="commit",accepted=True,**engine.commit(chosen,"Frozen containment/geometry/edge rule; no reference")))
    # Highest original complexity first, deterministic tie-break; no fixed ten.
    ordered=sorted(engine.current,key=lambda p:(-vertices(p.geometry),p.identifier))
    for p in ordered:
        alternatives=[]
        for tool in ("contour_dp","contour_vw","contour_lines"):
            for tolerance in (.5*engine.config.max_boundary_shift,engine.config.max_boundary_shift):
                if engine.generated>=engine.config.tool_budget or engine.remaining()<=0:
                    break
                r=engine.trial(tool,[p.identifier],tolerance=tolerance)
                events.append(dict(action="trial",**r))
                if r["accept"]:
                    gain=r["before"]["vertices"]-r["candidate"]["vertices"]
                    detail=r["details"][0].get("metrics",{}).get("targets",{}).get(p.identifier,{})
                    risk=max(1-detail.get("raster_iou",1),detail.get("area_change",0))
                    alternatives.append((gain,-risk,r["trial_id"]))
        if alternatives:
            chosen=max(alternatives)[2]
            events.append(dict(action="commit",accepted=True,**engine.commit(chosen,"Best verified vertex reduction; tie-break by shape risk")))
        if engine.generated>=engine.config.tool_budget or engine.remaining()<=0:
            break
    return events,"tool_budget" if engine.generated>=engine.config.tool_budget else "time_budget" if engine.remaining()<=0 else "rule_complete",False


def run_goal(engine,planner,progress=None):
    events=[]
    stop="model_call_budget"
    failed=False
    for iteration in range(engine.config.model_calls):
        if engine.remaining()<=0 or engine.generated>=engine.config.tool_budget:
            stop="time_budget" if engine.remaining()<=0 else "tool_budget"
            break
        context=dict(engine.inspect(),goal_contract=goal_contract(engine.config),available_actions=[],
                     local_tool_index=[{k:c[k] for k in ("tool_id","task","description","parameters")} for c in TOOL_CARDS],
                     max_commands=engine.config.max_commands,
                     remaining_budget=dict(iterations=engine.config.model_calls-iteration,
                                           tools=engine.config.tool_budget-engine.generated,seconds=round(engine.remaining(),2)),
                     history=events[-18:],active_trial_ids=[t for t,(r,_) in engine.trials.items() if r["base_state"]==state_key(engine.current)],
                     review_ids=sorted(engine.reviewed))
        # Avoid recursively embedding contexts and duplicate full trials.
        context["history"]=[{k:v for k,v in e.items() if k not in {"input_context","raw_output","usage","billed_usage","cache_key"}} for e in context["history"]]
        try:
            receipt=planner.plan(context,engine.tools.image,to_parcels(engine.current,engine.tools.shape),engine.remaining())
        except Exception as error:
            record=dict(getattr(error,"record",{}))
            record.update(action="plan",iteration=iteration,api_request=True,api_attempts=1,
                          error=f"{type(error).__name__}: {error}",input_context=context)
            events.append(record)
            failed=True
            stop="planner_error"
            break
        receipt.update(action="plan",iteration=iteration,api_request=not receipt.get("cache_hit",False),api_attempts=int(not receipt.get("cache_hit",False)))
        events.append(receipt)
        packet=receipt["goal"]
        for command in packet["commands"]:
            c=command
            event=dict(action=c["kind"],iteration=iteration,command=c,diagnosis=packet["diagnosis"],plan=packet["plan"])
            try:
                if c["kind"]=="stop":
                    event.update(accepted=True)
                    events.append(event)
                    return events,"controller_stop",False
                if c["kind"]=="inspect":
                    planner.focus_ids=c["target_ids"]
                    event["result"]=engine.inspect(c["target_ids"])
                elif c["kind"]=="find_tools":
                    event["result"]=engine.find_tools(c["query"] or "parcel vector simplification overlap")
                elif c["kind"] in {"trial","apply"}:
                    if c["kind"]=="apply" and not (c["tool_id"] or "").startswith("contour_"):
                        raise ValueError("Overlap requires preview trial then explicit later commit")
                    record=engine.trial(c["tool_id"],c["target_ids"],c["tolerance"],c["keep_id"])
                    planner.preview_trials.append(record["trial_id"])
                    event["result"]=record
                    if c["kind"]=="apply" and record["accept"]:
                        event["commit"]=engine.commit(record["trial_id"],c["reason"])
                        event["accepted"]=True
                elif c["kind"]=="commit":
                    # Only trial IDs observable before this model response may
                    # be committed: overlap previews must actually reach model.
                    if c["trial_id"] not in context["active_trial_ids"]:
                        raise ValueError("Commit must reference an observed non-stale trial")
                    event["result"]=engine.commit(c["trial_id"],c["reason"])
                    event["accepted"]=True
                elif c["kind"]=="restore":
                    event["result"]=engine.restore()
                    event["accepted"]=True
                elif c["kind"]=="review":
                    if not c["target_ids"]:
                        raise ValueError("Review requires explicit IDs")
                    engine.reviewed.update(c["target_ids"])
                    event["result"]=dict(review_ids=sorted(engine.reviewed))
            except Exception as error:
                event.update(accepted=False,error=f"{type(error).__name__}: {error}")
            events.append(event)
            if progress:
                progress(iteration,event)
        if progress:
            progress(iteration,dict(action="round_complete",scene=scene_metrics(engine.current)))
    return events,stop,failed


def run(engine,method,planner=None,progress=None):
    if method=="B0-RAW":
        engine.current=list(engine.raw)
        events,stop,failed=[],"raw",False
    elif method=="B0-NORM":
        events,stop,failed=[],"common_legalization_only",False
    elif method=="B1-RULE":
        events,stop,failed=run_rule(engine)
    elif method=="A1-GOAL-LOCAL":
        if planner is None:
            raise ValueError("Goal branch requires a real model planner")
        events,stop,failed=run_goal(engine,planner,progress)
    else:
        raise ValueError("Unknown v9 method")
    current_lookup={p.identifier:p for p in engine.current}
    initial_lookup={p.identifier:p for p in engine.initial}
    lineage=[]
    removed={r["original_id"]:r for r in engine.lineage}
    for p in engine.initial:
        lineage.append(removed.get(p.identifier,dict(original_id=p.identifier,active=p.identifier in current_lookup,replaced_by=None)))
    survivor_original=sum(vertices(initial_lookup[i].geometry) for i in current_lookup)
    final=scene_metrics(engine.current)
    trace=dict(version=VERSION,method=method,initial=scene_metrics(engine.raw),normalized_initial=scene_metrics(engine.initial),
               final=final,goal_contract=goal_contract(engine.config),action_trace=events,
               stop_reason=stop,planner_failed=failed,review_ids=sorted(engine.reviewed),lineage=lineage,
               candidate_evaluations=engine.generated,postprocess_seconds=time.monotonic()-engine.started,
               survivor_original_vertices=survivor_original,
               survivor_compression=1-final["vertices"]/max(1,survivor_original),
               suppressed_original_vertices=sum(vertices(p.geometry) for p in engine.initial if p.identifier not in current_lookup),
               initial_conflict_components=conflict_components(engine.initial),
               scope="two tasks; local tools; no online discovery; no hole filling or final compaction")
    return engine.current,trace
