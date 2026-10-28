"""Image-grounded, model-authored geometry edits with observed-feedback commits.

No reference labels, algorithm selection, generated code, or implicit simplifier
are used by D1/D2. Existing regional search is retained for the B1 control.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import split, unary_union

import fieldbqc_region_repair as region
from fieldbqc_geometry import polygons, scene_metrics, to_parcels, vertices
from fieldbqc_planner import APIPlanner, PlanningError
from fieldbqc_selective_repair import state_key
from fieldbqc_vertex_optimization import anchor_reasons

VERSION = "goal-direct-edit-v13.0"
DIRECT_METHODS = ("D1-LLM-ONCE", "D2-AGENT-EDIT")
SIMPLIFY = {"retain_vertices", "replace_boundary_arc"}
OVERLAP = {"partition_overlap", "shared_boundary"}
EPS = 1e-7


@dataclass(frozen=True)
class DirectConfig(region.RegionConfig):
    model_calls: int = 8
    seconds: float = 480.0
    preserve_all_ids: bool = True
    max_edits: int = 64
    label_page_size: int = 40
    max_label_pages: int = 3
    preview_size: int = 768

    def __post_init__(self):
        super().__post_init__()
        for key in ("max_edits", "label_page_size", "max_label_pages", "preview_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Invalid {key}")
        if not self.preserve_all_ids:
            raise ValueError("v13 direct protocol preserves all IDs; suppression is a separate experiment")
        if self.label_page_size > 80 or not 128 <= self.preview_size <= 2048:
            raise ValueError("Unreadable label pages or excessive preview size")


def goal_contract(config):
    return dict(goal_id=VERSION, tasks=["simplify_contour", "resolve_overlap"],
                priority=["protect_real_identity_and_boundary", "correct_ownership", "reduce_vertices_and_overlap", "cost"],
                objective="Use imagery and numbered coordinates to author explicit geometry edits; inspect actual candidate feedback before revising or committing.",
                quality_constraints=asdict(config), fixed_vertex_target=None,
                development_targets_only=dict(vertex_reduction=config.target_vertex_reduction,
                                              overlap_reduction=config.target_overlap_reduction),
                runtime_ground_truth=False, preserve_all_ids=True,
                scope="Direct coordinate/index edits. No algorithm search, network tools, code generation or hidden rule finish in D1/D2.",
                uncertainty="May stop or review with unmet goals. Valid geometry is not semantic proof.",
                atomicity="Edits sharing an ID or existing common boundary are an atomic batch; independent batches are checked against the entire accumulated candidate.")


def ring_records(geometry):
    result = []
    for index, part in enumerate(polygons(geometry)):
        for name, ring in [("outer", part.exterior)] + [(f"hole-{k}", r) for k, r in enumerate(part.interiors)]:
            result.append(dict(ring_id=f"part-{index}/{name}",
                               coordinates=[[float(x), float(y)] for x, y in ring.coords[:-1]]))
    return result


def _point(value, shape):
    if not isinstance(value, list) or len(value) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError("Expected finite [x,y] in original pixel-cell coordinates")
    x, y = value
    if not 0 <= x <= shape[1] or not 0 <= y <= shape[0]:
        raise ValueError("coordinate_outside_image")
    return (float(x), float(y))


def _path(value, shape, minimum=2):
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError("Too few path vertices")
    coords = [_point(p, shape) for p in value]
    if any(a == b for a, b in zip(coords, coords[1:])):
        raise ValueError("Consecutive duplicate coordinates")
    return coords


def edit_ids(edit):
    return [edit["parcel_id"]] if edit["kind"] in SIMPLIFY else edit["target_ids"]


def replace_ring(geometry, ring_id, coordinates):
    parts = polygons(geometry)
    known = {r["ring_id"] for r in ring_records(geometry)}
    if ring_id not in known:
        raise ValueError("Unknown current-state ring_id")
    modified = []
    for index, part in enumerate(parts):
        shell = coordinates if ring_id == f"part-{index}/outer" else list(part.exterior.coords[:-1])
        holes = [coordinates if ring_id == f"part-{index}/hole-{k}" else list(r.coords[:-1])
                 for k, r in enumerate(part.interiors)]
        modified.append(Polygon(shell, holes))
    # Never make_valid/simplify a model candidate: invalid edits must be rejected.
    return modified[0] if len(modified) == 1 else MultiPolygon(modified)


def _object(properties):
    return dict(type="object", additionalProperties=False, properties=properties, required=list(properties))


STRING = dict(type="string")
NUMBER = dict(type="number")
INTEGER = dict(type="integer")
XY = dict(type="array", items=NUMBER, minItems=2, maxItems=2)
PATH = dict(type="array", items=XY)
IDS = dict(type="array", items=STRING)
EDIT_SCHEMAS = [
    _object(dict(kind=dict(type="string", enum=["retain_vertices"]), parcel_id=STRING, ring_id=STRING,
                 keep_indices=dict(type="array", items=INTEGER), reason=STRING, evidence_refs=IDS)),
    _object(dict(kind=dict(type="string", enum=["replace_boundary_arc"]), parcel_id=STRING, ring_id=STRING,
                 start_index=INTEGER, end_index=INTEGER, path=PATH, reason=STRING, evidence_refs=IDS)),
    _object(dict(kind=dict(type="string", enum=["partition_overlap"]), target_ids=IDS,
                 allocations=dict(type="array", items=_object(dict(parcel_id=STRING, polygon=PATH))),
                 reason=STRING, evidence_refs=IDS)),
    _object(dict(kind=dict(type="string", enum=["shared_boundary"]), target_ids=IDS,
                 path=PATH, side_point=XY, side_owner=STRING, reason=STRING, evidence_refs=IDS)),
]
VIEW = _object(dict(parcel_id=STRING, ring_id=STRING, start_index=INTEGER, count=INTEGER))
COMMAND = _object(dict(kind=dict(type="string", enum=["propose", "commit", "inspect", "review", "restore", "stop"]),
                       target_ids=IDS, edits=dict(type="array", items=dict(anyOf=EDIT_SCHEMAS)),
                       trial_id=dict(type=["string", "null"]), view=dict(anyOf=[VIEW, dict(type="null")]), reason=STRING))
PACKET = _object(dict(state_id=STRING, diagnosis=STRING, commands=dict(type="array", items=COMMAND)))


def validate_edit(edit, context):
    kind = edit.get("kind") if isinstance(edit, dict) else None
    spec = next((s for s in EDIT_SCHEMAS if s["properties"]["kind"]["enum"] == [kind]), None)
    if spec is None or set(edit) != set(spec["required"]):
        raise ValueError("Unknown edit or incorrect fields")
    known = {p["parcel_id"]: p for p in context["instances"]}
    ids = edit_ids(edit)
    if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids) or not set(ids) <= set(known)):
        raise ValueError("Unknown/duplicate edit target")
    if not isinstance(edit["reason"], str) or not 1 <= len(edit["reason"]) <= 2000:
        raise ValueError("Short observable edit reason required")
    refs = edit["evidence_refs"]
    if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
        raise ValueError("Invalid evidence references")
    available = set(context.get("evidence_ids", []))
    if not set(refs) <= available:
        raise ValueError("Unobserved evidence reference")
    if kind in SIMPLIFY:
        rings = {r["ring_id"]: r for r in known[edit["parcel_id"]]["rings"]}
        if not isinstance(edit["ring_id"], str) or edit["ring_id"] not in rings:
            raise ValueError("Unknown ring")
        n = len(rings[edit["ring_id"]]["coordinates"])
        if kind == "retain_vertices":
            idx = edit["keep_indices"]
            if (not isinstance(idx, list) or not 3 <= len(idx) <= n
                    or any(type(i) is not int or not 0 <= i < n for i in idx)
                    or len(set(idx)) != len(idx)
                    or sum((b-a) % n for a, b in zip(idx, idx[1:] + idx[:1])) != n):
                raise ValueError("Keep indices must be unique and preserve cyclic order, at least three per ring")
        else:
            if any(type(edit[k]) is not int or not 0 <= edit[k] < n for k in ("start_index", "end_index")):
                raise ValueError("Invalid arc endpoint index")
            if edit["start_index"] == edit["end_index"]:
                raise ValueError("Arc cannot replace the entire ring")
            _path(edit["path"], context["image_shape"])
    elif kind == "partition_overlap":
        if len(ids) < 2 or not isinstance(edit["allocations"], list):
            raise ValueError("Partition requires overlapping IDs and allocations")
        owners = []
        for a in edit["allocations"]:
            if not isinstance(a, dict) or set(a) != {"parcel_id", "polygon"}:
                raise ValueError("Invalid allocation")
            owners.append(a["parcel_id"])
            if a["polygon"] != []:
                coords = _path(a["polygon"], context["image_shape"], 3)
                if len(set(coords)) < 3 or not Polygon(coords).is_valid or Polygon(coords).area <= EPS:
                    raise ValueError("Invalid allocation polygon")
        if any(not isinstance(i, str) for i in owners) or len(owners) != len(set(owners)) or set(owners) != set(ids):
            raise ValueError("Supply exactly one allocation per target; [] allocates nothing")
    else:
        if len(ids) != 2 or edit["side_owner"] not in ids:
            raise ValueError("Shared boundary requires two IDs and an explicit side owner")
        _path(edit["path"], context["image_shape"])
        _point(edit["side_point"], context["image_shape"])


def validate_packet(packet, context):
    if not isinstance(packet, dict) or set(packet) != set(PACKET["required"]):
        raise ValueError("Expected state_id/diagnosis/commands")
    if packet["state_id"] != context["state_id"]:
        raise ValueError("Stale state_id")
    if not isinstance(packet["diagnosis"], str) or not 1 <= len(packet["diagnosis"]) <= 3000:
        raise ValueError("Short observable diagnosis required")
    commands = packet["commands"]
    if not isinstance(commands, list) or not 1 <= len(commands) <= context["max_commands"]:
        raise ValueError("Invalid command count")
    known = {p["parcel_id"]: p for p in context["instances"]}
    proposals = 0
    for c in commands:
        if not isinstance(c, dict) or set(c) != set(COMMAND["required"]):
            raise ValueError("Invalid command fields")
        if c["kind"] not in COMMAND["properties"]["kind"]["enum"]:
            raise ValueError("Unknown command")
        ids = c["target_ids"]
        if (not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids) or not set(ids) <= set(known)):
            raise ValueError("Unknown/duplicate command target")
        if not isinstance(c["reason"], str) or not 1 <= len(c["reason"]) <= 2000:
            raise ValueError("Command reason required")
        if not isinstance(c["edits"], list):
            raise ValueError("Expected edits array")
        if c["kind"] == "propose":
            proposals += 1
            if not 1 <= len(c["edits"]) <= context["config"]["max_edits"]:
                raise ValueError("Invalid edit count")
            for edit in c["edits"]:
                validate_edit(edit, context)
            if set(ids) != {i for e in c["edits"] for i in edit_ids(e)}:
                raise ValueError("Declared target IDs must match edited IDs")
        elif c["edits"]:
            raise ValueError("Only propose accepts edits")
        if c["kind"] == "commit":
            if not isinstance(c["trial_id"], str):
                raise ValueError("Commit requires an observed trial_id")
        elif c["trial_id"] is not None:
            raise ValueError("Only commit accepts trial_id")
        if c["kind"] == "inspect" and c["view"] is not None:
            view = c["view"]
            if not isinstance(view, dict) or set(view) != set(VIEW["required"]) or view["parcel_id"] not in known:
                raise ValueError("Invalid vertex page view")
            rings = {r["ring_id"]: r for r in known[view["parcel_id"]]["rings"]}
            if (view["ring_id"] not in rings or type(view["start_index"]) is not int
                    or not 0 <= view["start_index"] < len(rings[view["ring_id"]]["coordinates"])
                    or type(view["count"]) is not int or not 1 <= view["count"] <= context["config"]["label_page_size"]):
                raise ValueError("Invalid vertex page range")
        elif c["view"] is not None:
            raise ValueError("Only inspect accepts view")
        if c["kind"] == "review" and not ids:
            raise ValueError("Review needs explicit IDs")
        if context["method"] == "D1-LLM-ONCE" and c["kind"] not in {"propose", "review", "stop"}:
            raise ValueError("Single-shot has one response; no inspect/commit/restore feedback commands")
    if proposals > 1:
        raise ValueError("Use one propose containing all edits per response")
    return packet


class DirectEngine(region.RegionEngine):
    def __init__(self, image, original, config=None):
        super().__init__(image, original, config or DirectConfig())
        self.revision = 0
        self.direct_attempts = {}
        self.observed_trials = set()
        self.focus_ids = []
        self.focus_view = None
        self.evidence_root = None
        self.evidence_counter = 0

    @property
    def state_id(self):
        return f"s{self.revision}-" + state_key(self.current)[:24]

    def fork(self):
        # Keep the stronger shared ownership verifier in ordinary search forks.
        child = super().fork()
        child.__class__ = type(self)
        return child

    def numbered_evidence(self):
        evidence = self.evidence()
        lookup = {p.identifier: p for p in self.current}
        for row in evidence["instances"]:
            row["rings"] = ring_records(lookup[row["parcel_id"]].geometry)
        evidence.update(state_id=self.state_id, image_shape=list(self.tools.shape),
                        coordinate_contract="Original pixel-cell coordinates, x right/y down; ring indices exclude closing duplicate; all points addressable. Display transforms are supplied per frame.")
        return evidence

    def verify_conflict_candidate(self, items, candidate, ids, removed=None, keep_id=None):
        verdict = super().verify_conflict_candidate(items, candidate, ids, removed, keep_id)
        if removed:
            return verdict
        lookup = {p.identifier: p for p in items}
        new = {p.identifier: p for p in candidate}
        disputed = self.disputed([lookup[i] for i in ids])
        reasons = [] if verdict["accept"] else list(verdict["reasons"])
        for uid in ids:
            if uid in new and lookup[uid].geometry.difference(disputed).difference(new[uid].geometry).area > EPS:
                reasons.append("non_disputed_geometry_removed")
        verdict.update(accept=not reasons, reasons=sorted(set(reasons)))
        return verdict

    @staticmethod
    def disputed(items):
        return unary_union([a.geometry.intersection(b.geometry) for k, a in enumerate(items) for b in items[k+1:]])

    def atomic_batches(self, edits):
        """Same target or existing shared edge connects edit units, transitively."""
        lookup = {p.identifier: p for p in self.current}
        parent = list(range(len(edits)))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        targets = [set(edit_ids(e)) for e in edits]
        for i in range(len(edits)):
            for j in range(i):
                linked = bool(targets[i] & targets[j]) or any(
                    lookup[a].geometry.boundary.intersection(lookup[b].geometry.boundary).length > EPS
                    for a in targets[i] for b in targets[j])
                if linked:
                    parent[root(i)] = root(j)
        batches = {}
        for i, edit in enumerate(edits):
            batches.setdefault(root(i), []).append(edit)
        return list(batches.values())

    def _simplify_edits(self, items, edits):
        lookup = {p.identifier: p for p in items}
        touched = set()
        for edit in edits:
            uid, ring_id = edit["parcel_id"], edit["ring_id"]
            if (uid, ring_id) in touched:
                raise ValueError("Only one edit per state ring in an atomic batch")
            touched.add((uid, ring_id))
            geometry = lookup[uid].geometry
            coords = next(r["coordinates"] for r in ring_records(geometry) if r["ring_id"] == ring_id)
            if edit["kind"] == "retain_vertices":
                replacement = [coords[i] for i in edit["keep_indices"]]
            else:
                start, end = edit["start_index"], edit["end_index"]
                path = _path(edit["path"], self.tools.shape)
                if not np.allclose(path[0], coords[start], atol=EPS, rtol=0) or not np.allclose(path[-1], coords[end], atol=EPS, rtol=0):
                    raise ValueError("Replacement path endpoints must match the indexed original arc endpoints")
                rest, i = [], (end + 1) % len(coords)
                while i != start:
                    rest.append(coords[i])
                    i = (i + 1) % len(coords)
                replacement = path + rest
            if len({tuple(p) for p in replacement}) < 3:
                raise ValueError("Ring collapsed")
            lookup[uid] = lookup[uid].with_geometry(replace_ring(geometry, ring_id, replacement))
        return [lookup[p.identifier] for p in items]

    def _overlap_edit(self, items, edit):
        old = {p.identifier: p for p in items}
        ids = edit["target_ids"]
        disputed = self.disputed([old[i] for i in ids])
        if disputed.area <= EPS:
            raise ValueError("No disputed area among targets")
        if edit["kind"] == "partition_overlap":
            allocation = {a["parcel_id"]: Polygon(a["polygon"]) if a["polygon"] else Polygon()
                          for a in edit["allocations"]}
        else:
            line = LineString(_path(edit["path"], self.tools.shape))
            if not line.is_simple or line.length <= EPS:
                raise ValueError("Shared boundary must be a simple nonempty path")
            faces = polygons(split(disputed, line))
            if len(faces) != 2 or abs(sum(g.area for g in faces) - disputed.area) > EPS:
                raise ValueError("Path must divide the disputed area into exactly two faces; use explicit allocations otherwise")
            point = Point(_point(edit["side_point"], self.tools.shape))
            matches = [i for i, g in enumerate(faces) if g.contains(point)]
            if len(matches) != 1:
                raise ValueError("side_point must lie strictly inside one disputed face")
            owner = edit["side_owner"]
            other = next(i for i in ids if i != owner)
            allocation = {owner: faces[matches[0]], other: faces[1-matches[0]]}
        # Clipping is the declared executor contract, not inferred ownership:
        # model-authored polygons select only the original disputed material.
        assigned = {uid: disputed.intersection(allocation[uid]).intersection(old[uid].geometry) for uid in ids}
        if disputed.symmetric_difference(unary_union(list(assigned.values()))).area > EPS:
            raise ValueError("Allocation leaves uncovered disputed area")
        if sum(g.area for g in assigned.values()) - unary_union(list(assigned.values())).area > EPS:
            raise ValueError("Allocations still overlap")
        replacements = {uid: old[uid].geometry.difference(disputed).union(assigned[uid]) for uid in ids}
        return [p.with_geometry(replacements[p.identifier]) if p.identifier in replacements else p for p in items]

    def propose_edits(self, edits, expected_state):
        if expected_state != self.state_id:
            raise ValueError("Stale direct edit state")
        context = context_for(self, [], 0, "D2-AGENT-EDIT")
        # Execution revalidates numbers, IDs and fields; evidence provenance is
        # checked against actual request context separately by validate_packet.
        context["evidence_ids"] = list({r for e in edits for r in e.get("evidence_refs", [])})
        if not 1 <= len(edits) <= self.config.max_edits:
            raise ValueError("Invalid edit count")
        for edit in edits:
            validate_edit(edit, context)
        origin = list(self.current)
        candidate = list(origin)
        details, overlap_ids = [], set()
        tid = f"d{len(self.trials):05d}"
        for batch_index, batch in enumerate(self.atomic_batches(edits)):
            ids = sorted({i for e in batch for i in edit_ids(e)})
            simplifications = [e for e in batch if e["kind"] in SIMPLIFY]
            overlaps = [e for e in batch if e["kind"] in OVERLAP]
            task = "joint" if simplifications and overlaps else "simplify" if simplifications else "overlap"
            attempted = list(candidate)
            try:
                self._reserve()
                stages = []
                if len(overlaps) > 1:
                    raise ValueError("One complete overlap allocation per atomic component")
                if simplifications:
                    simplify_ids = sorted({i for e in simplifications for i in edit_ids(e)})
                    attempted = self._simplify_edits(candidate, simplifications)
                    verdict = self.tools.verify(candidate, attempted, "simplify_contour", simplify_ids)
                    reasons = [] if verdict["accept"] else list(verdict["reasons"])
                    reasons += anchor_reasons(self.tools, self.anchors, attempted, simplify_ids)
                    verdict.update(accept=not reasons, reasons=sorted(set(reasons)))
                    stages.append(dict(task="simplify", **deepcopy(verdict)))
                if overlaps and (not simplifications or verdict["accept"]):
                    if simplifications:
                        self._reserve()  # Both stage checks consume budget.
                    phase_before = list(attempted)
                    attempted = self._overlap_edit(phase_before, overlaps[0])
                    verdict = self.verify_conflict_candidate(phase_before, attempted, overlaps[0]["target_ids"])
                    stages.append(dict(task="overlap", **deepcopy(verdict)))
                if stages:
                    verdict["stage_checks"] = stages
                if verdict["accept"] and any(p.geometry.is_empty or not p.geometry.is_valid for p in attempted):
                    verdict.update(accept=False, reasons=["invalid_geometry"])
            except Exception as error:
                verdict = dict(accept=False, reasons=[str(error)], error_type=type(error).__name__, metrics={})
            attempt_id = f"{tid}-b{batch_index:03d}"
            record = dict(attempt_id=attempt_id, target_ids=ids, task=task, edits=deepcopy(batch), **verdict)
            self.direct_attempts[attempt_id] = (record, list(attempted))
            details.append(record)
            if verdict["accept"]:
                candidate = attempted
                if overlaps:
                    overlap_ids.update(overlaps[0]["target_ids"])
        # Individual gates already inspect full accumulated scene; final guard
        # catches any unexpected combination or mutation before a commit exists.
        safe = (len(candidate) == len(origin)
                and {p.identifier for p in candidate} == {p.identifier for p in origin}
                and all(p.geometry.is_valid and not p.geometry.is_empty for p in candidate)
                and scene_metrics(candidate)["overlap_excess_area"] <= scene_metrics(origin)["overlap_excess_area"] + EPS)
        if not safe:
            candidate = origin
            for d in details:
                d["accept"] = False
                d["reasons"].append("final_scene_guard")
            overlap_ids.clear()
        summary = dict(trial_id=tid, tool_id="direct_edit", target_ids=sorted({i for e in edits for i in edit_ids(e)}),
                       state_id=self.state_id, base_state=state_key(origin), accept=any(d["accept"] for d in details),
                       before=scene_metrics(origin), candidate=scene_metrics(candidate), details=details,
                       edits=deepcopy(edits), overlap_anchor_ids=sorted(overlap_ids))
        self.trials[tid] = (summary, list(candidate))
        return summary

    def commit(self, trial_id, reason, *, single_shot=False):
        if trial_id not in self.trials or self.trials[trial_id][0].get("tool_id") != "direct_edit":
            return super().commit(trial_id, reason)
        record, candidate = self.trials[trial_id]
        if record["state_id"] != self.state_id or record["base_state"] != state_key(self.current):
            raise ValueError("Stale direct candidate")
        if not record["accept"]:
            raise ValueError("Rejected candidate cannot be committed")
        if not single_shot and trial_id not in self.observed_trials:
            raise ValueError("Commit requires candidate feedback delivered in an earlier model response")
        self.checkpoints.append((list(self.current), list(self.anchors), deepcopy(self.lineage)))
        self.current = list(candidate)
        anchors = {p.identifier: p for p in self.anchors}
        anchors.update({p.identifier: p for p in self.current if p.identifier in record["overlap_anchor_ids"]})
        self.anchors = [anchors[p.identifier] for p in self.current]
        self.revision += 1
        self.observed_trials.clear()
        self.focus_view = None
        record["committed_to_scene"] = True
        return dict(committed=True, trial_id=trial_id, state_id=self.state_id,
                    protocol="single_shot_automatic_validation" if single_shot else "observed_feedback_commit",
                    scene=scene_metrics(self.current), reason=reason)

    def restore(self):
        result = super().restore()
        self.revision += 1  # Rollback never resurrects old vertex/trial IDs.
        self.observed_trials.clear()
        self.focus_view = None
        return dict(result, state_id=self.state_id)


def context_for(engine, events, iteration, method):
    feedback = [r for r, _ in engine.trials.values()
                if r.get("tool_id") == "direct_edit" and r.get("state_id") == engine.state_id][-2:]
    return dict(engine.numbered_evidence(), method=method, goal_contract=goal_contract(engine.config),
                config=asdict(engine.config), available_actions=[], max_commands=engine.config.max_commands,
                objective_status=engine.objective_status(), remaining_budget=dict(
                    model_responses=(1 if method == "D1-LLM-ONCE" else engine.config.model_calls)-iteration,
                    edit_checks=engine.config.tool_budget-engine.generated, seconds=round(engine.remaining(), 2)),
                feedback=deepcopy(feedback), feedback_trial_ids=[r["trial_id"] for r in feedback],
                active_trial_ids=[r["trial_id"] for r in feedback if r["accept"]],
                evidence_ids=[], history=[{k: v for k, v in e.items() if k not in {
                    "input_context", "raw_output", "usage", "billed_usage", "cache_key"}} for e in events[-8:]],
                review_ids=sorted(engine.reviewed))


PROMPT = """You directly edit agricultural parcel polygons using original imagery
and all numbered ring coordinates. This is spatial judgment, NOT tool selection.
Protect real parcels, real turns, holes, components and non-disputed material.
Then reduce redundant points and resolve overlaps as far as the evidence allows.
No fixed ten-point target. 40%/99% are development diagnostics, not proof of quality.
Return state_id/diagnosis/commands in the schema. Reasons must be short, observable
and refer only to evidence IDs actually delivered. No hidden chain of thought.

retain_vertices: give all indices to keep for a named ring, unique, in its original
cyclic order (at least 3). Do not list an algorithm or tolerance. Every original
point is addressable in rings.coordinates. Labels/pages merely help locate them.
replace_boundary_arc: replace the forward cyclic arc start_index to end_index
with path, whose endpoints equal those exact original vertices. Other arc stays.
partition_overlap: supply one polygon allocation per target ID ([] means none).
Executor clips your polygon to disputed area AND that target's old geometry;
exclusive areas stay unchanged. All disputed material must have exactly one owner.
shared_boundary: two IDs, your explicit path splitting their disputed area into
exactly two faces; side_point strictly inside one disputed face and side_owner
specify ownership; the other face goes to the other ID. Complex cases need polygons.
All path/polygon coordinates use ORIGINAL pixel cells, NOT preview coordinates.
Every ID is preserved; uncertain duplicates go to review, never deactivation.

At most one propose per response, containing all edits and exactly their target_ids.
Edits on the same ID or existing common edge form an atomic batch: all pass or none.
Use at most one edit per ring and one overlap allocation per atomic component.
Joint simplification may edit both neighboring rings. A mixed atomic batch executes
ALL simplification edits first using the input's indices, verifies that phase,
then applies the explicit overlap allocation to the resulting disputed area and
verifies ownership. BOTH phases must pass, otherwise the whole batch stays original.
Each phase consumes a check. This order is identical in single-shot and feedback
modes, so single-shot can express both tasks without a further response. No
intermediate feedback is provided within a proposal. New rounds use new indices.
No implicit simplify, make_valid, rule finish or guessed ownership rescues your edit.

D1-LLM-ONCE: initial evidence and ONE complete response, propose/review/stop only.
Program validates atomic batches and automatically commits the accepted result;
you cannot inspect feedback. Cover the important targets within this response.
D2-AGENT-EDIT: propose, read ACTUAL candidate previews/numbers in the next response,
then commit a previously observed accepted trial_id or revise. No same-response
propose-and-commit. Any commit/restore changes state_id and invalidates old indices.
Use inspect with explicit target_ids and optionally view(parcel_id,ring_id,
start_index,count) for another numbered page. All points remain in the numeric table.
Only propose accepts edits; only commit accepts trial_id; only inspect accepts view;
otherwise supply []/null. review needs IDs. restore undoes one commit. stop is allowed
when uncertain or no safe progress: explain unresolved issues instead of forcing a
number. Last response should commit an ALREADY OBSERVED candidate or stop/review,
not create an unseen candidate. The same model/evidence/executor/gates serve D1/D2.
Runtime has NO ground truth. Instance score is not pixel probability. Geometry
validity alone does not establish correct boundaries or ownership. Judge imagery.
"""


def _crop_bounds(engine, ids, coordinates=None):
    h, w = engine.tools.shape
    if coordinates:
        values = np.asarray(coordinates)
        x0, y0 = values.min(axis=0)
        x1, y1 = values.max(axis=0)
    else:
        selected = [p.geometry for p in engine.current if p.identifier in ids]
        if not selected:
            return [0, 0, w, h]
        x0, y0, x1, y1 = unary_union(selected).bounds
    return [max(0, int(math.floor(x0))-8), max(0, int(math.floor(y0))-8),
            min(w, int(math.ceil(x1))+9), min(h, int(math.ceil(y1))+9)]


def _frame(engine, items, bbox, title, labels=None):
    x0, y0, x1, y1 = bbox
    raw = engine.tools.image[y0:y1, x0:x1].copy()
    scale = engine.config.preview_size / max(raw.shape[:2])
    width, height = max(1, round(raw.shape[1]*scale)), max(1, round(raw.shape[0]*scale))
    frame = cv2.resize(raw, (width, height), interpolation=cv2.INTER_CUBIC)
    sx, sy = width / raw.shape[1], height / raw.shape[0]
    def project(coords):
        return np.rint((np.asarray(coords, dtype=float) - [x0, y0]) * [sx, sy]).astype(np.int32)
    colors = [(0, 255, 255), (255, 100, 255), (80, 255, 80), (255, 170, 20), (50, 100, 255)]
    if items is not None:
        # Full vector paths, not raster contour approximations, show actual edits.
        for k, p in enumerate(items):
            color = colors[k % len(colors)]
            for part in polygons(p.geometry):
                for ring in [part.exterior, *part.interiors]:
                    xy = project(ring.coords)
                    cv2.polylines(frame, [xy], True, color, 1, cv2.LINE_AA)
            if not p.geometry.is_empty:
                xy = project([p.geometry.representative_point().coords[0]])[0]
                if 0 <= xy[0] < width and 0 <= xy[1] < height:
                    cv2.putText(frame, p.identifier, tuple(xy), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        if all(p.geometry.is_valid for p in items):
            occupied = sum((engine.tools.mask(p).astype(np.int32) for p in items), np.zeros(engine.tools.shape, np.int32))
            conflict = cv2.resize((occupied[y0:y1, x0:x1] > 1).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            frame[conflict] = (.55*frame[conflict] + .45*np.array([0, 0, 255])).astype(np.uint8)
    if labels:
        coords, indices = labels
        occupied_labels = []
        for index, xy in zip(indices, project([coords[i] for i in indices])):
            cv2.circle(frame, tuple(xy), 2, (0, 0, 255), -1)
            text = str(index)
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)
            choices = []
            for radius in (5, 15, 28, 42, 60, 85, 115):
                for ox, oy in ((1, -1), (1, 1), (-1, -1), (-1, 1), (0, -1), (0, 1), (1, 0), (-1, 0)):
                    tx = min(max(2, int(xy[0])+ox*radius), max(2, width-tw-3))
                    ty = min(max(th+3, int(xy[1])+oy*radius), max(th+3, height-baseline-3))
                    rect = (tx-2, ty-th-2, tx+tw+2, ty+baseline+2)
                    overlaps = sum(max(0, min(rect[2], r[2])-max(rect[0], r[0])) *
                                   max(0, min(rect[3], r[3])-max(rect[1], r[1])) for r in occupied_labels)
                    choices.append((overlaps, (tx-int(xy[0]))**2+(ty-int(xy[1]))**2, (tx, ty), rect))
            _, _, pos, rect = min(choices, key=lambda c: c[:2])
            occupied_labels.append(rect)
            center = (pos[0]+tw//2, pos[1]-th//2)
            cv2.line(frame, tuple(xy), center, (215, 215, 215), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (25, 25, 25), -1)
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    header = 34
    frame = cv2.copyMakeBorder(frame, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(frame, title[:110], (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
    return frame, dict(original_bbox=bbox, display_shape=list(frame.shape[:2]),
                        pixel_to_display=dict(scale_x=sx, scale_y=sy, offset_x=-x0*sx, offset_y=header-y0*sy),
                        coordinate_note="display_x=(original_x-x0)*scale_x; display_y=header+(original_y-y0)*scale_y; edit using ORIGINAL coordinates")


def visual_bundle(engine, context, output=None):
    """Return exactly the images delivered, with transforms and content hashes."""
    frames, urls, ids = [], [], []
    output = Path(output) if output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    def add(name, items, bbox, title, labels=None, **metadata):
        frame, transform = _frame(engine, items, bbox, title, labels)
        ok, encoded = cv2.imencode('.png', frame)
        if not ok:
            raise ValueError("Could not encode evidence")
        blob = encoded.tobytes()
        if output:
            (output / f"{name}.png").write_bytes(blob)
        frames.append(dict(evidence_id=name, sha256=hashlib.sha256(blob).hexdigest(),
                           file=f"{name}.png" if output else None, **transform, **metadata))
        urls.append('data:image/png;base64,' + base64.b64encode(blob).decode('ascii'))
        ids.append(name)
    h, w = engine.tools.shape
    add('raw-overview', None, [0, 0, w, h], 'original image; no reference annotations')
    add('current-overview', engine.current, [0, 0, w, h], engine.state_id + '; red=overlap')
    regions = region.linked_regions(engine.current)
    focus = engine.focus_ids or (regions[0]['target_ids'] if regions else [])
    bbox = _crop_bounds(engine, focus)
    add('raw-focus', None, bbox, 'raw focus ' + ','.join(focus))
    add('current-focus', engine.current, bbox, 'current focus; full vector paths; red=overlap')
    all_pages = []
    for instance in context['instances']:
        for ring in instance['rings']:
            for start in range(0, len(ring['coordinates']), engine.config.label_page_size):
                all_pages.append(dict(parcel_id=instance['parcel_id'], ring_id=ring['ring_id'],
                                      start_index=start, count=min(engine.config.label_page_size, len(ring['coordinates'])-start)))
    # Pages are a display aid, never a preselected subset of editable vertices.
    pages = sorted(all_pages, key=lambda p: (p['parcel_id'] not in focus, p['start_index'], p['parcel_id'], p['ring_id']))[:engine.config.max_label_pages]
    if engine.focus_view:
        if any(i['parcel_id'] == engine.focus_view['parcel_id'] and any(r['ring_id'] == engine.focus_view['ring_id'] for r in i['rings']) for i in context['instances']):
            pages = [dict(engine.focus_view)] + [p for p in pages if p != engine.focus_view]
            pages = pages[:engine.config.max_label_pages]
    for k, page in enumerate(pages):
        row = next(i for i in context['instances'] if i['parcel_id'] == page['parcel_id'])
        ring = next(r for r in row['rings'] if r['ring_id'] == page['ring_id'])
        coords = ring['coordinates']
        indices = [(page['start_index'] + i) % len(coords) for i in range(min(page['count'], len(coords))) ]
        page_bbox = _crop_bounds(engine, [row['parcel_id']], [coords[i] for i in indices])
        add(f'numbered-{k}', engine.current, page_bbox,
            f"{row['parcel_id']} {ring['ring_id']} indices {indices[0]}..{indices[-1]}",
            labels=(coords, indices), vertex_page=page)
    for trial_id in context['feedback_trial_ids']:
        record, candidate = engine.trials[trial_id]
        trial_bbox = _crop_bounds(engine, record['target_ids'])
        add(f'candidate-{trial_id}', candidate, trial_bbox,
            f"{trial_id} accepted={record['accept']}; actual executable candidate", trial_id=trial_id)
        rejected = [d for d in record['details'] if not d['accept']]
        for detail in rejected[:2]:
            attempt_id = detail['attempt_id']
            attempted = engine.direct_attempts[attempt_id][1]
            add(f'rejected-{attempt_id}', attempted, _crop_bounds(engine, detail['target_ids']),
                f"{attempt_id} REJECTED; not committed", attempt_id=attempt_id)
    manifest = dict(state_id=engine.state_id, frames=frames, available_vertex_pages=all_pages,
                    displayed_vertex_pages=pages, all_vertices_available_in_coordinates=True,
                    omitted_rejected_previews_note="At most two rejected batch previews per proposal; all attempted vectors and checks are retained.")
    if output:
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return urls, manifest, ids


class DirectPlanner(APIPlanner):
    def __init__(self, config, engine, cache):
        self.engine = engine
        self.pending_urls = []
        if int(config.get('max_instances', 24)) < len(engine.current):
            raise ValueError("Direct comparison requires all instance coordinates; raise max_instances explicitly")
        super().__init__(config, cache, schema=PACKET, system_prompt=PROMPT,
                         validator=validate_packet, prompt_version=VERSION,
                         visualizer=lambda image, parcels: self.pending_urls)

    def plan(self, context, image, parcels, remaining_seconds):
        started = time.monotonic()
        output = None
        if self.engine.evidence_root:
            output = Path(self.engine.evidence_root) / f'request-{self.engine.evidence_counter:03d}'
        self.engine.evidence_counter += 1
        try:
            if self.config.get('vision', False):
                self.pending_urls, manifest, ids = visual_bundle(self.engine, context, output)
            else:
                self.pending_urls, ids = [], []
                manifest = dict(state_id=self.engine.state_id, frames=[], vision=False,
                                note='Numeric-only ablation; no image evidence delivered')
        except Exception as error:
            raise PlanningError(str(error), dict(input_context=context, api_request=False,
                                                 api_attempts=0, failure_stage='local_evidence')) from error
        augmented = dict(context, evidence_ids=ids, visual_evidence=manifest,
                         feedback_observation='actual candidate images and checks' if self.pending_urls else 'numeric-only ablation')
        if output:
            output.mkdir(parents=True, exist_ok=True)
            (output / 'input_context.json').write_text(json.dumps(augmented, ensure_ascii=False, indent=2), encoding='utf-8')
        if remaining_seconds-(time.monotonic()-started) <= 0:
            raise PlanningError('No request time remains after local evidence preparation', dict(
                input_context=augmented, api_request=False, api_attempts=0, failure_stage='local_evidence_budget'))
        try:
            return super().plan(augmented, image, parcels, remaining_seconds-(time.monotonic()-started))
        except Exception as error:
            if hasattr(error, 'record'):
                raise
            raise PlanningError(str(error), dict(input_context=augmented, model=self.config['model'],
                                                 prompt_version=VERSION)) from error


def run_direct(engine, planner, method, progress=None):
    events, failed, stop = [], False, "model_call_budget"
    rounds = 1 if method == "D1-LLM-ONCE" else engine.config.model_calls
    for iteration in range(rounds):
        if engine.remaining() <= 0:
            stop = "time_budget"
            break
        context = context_for(engine, events, iteration, method)
        receipt = None
        try:
            receipt = planner.plan(context, engine.tools.image,
                                   to_parcels(engine.current, engine.tools.shape), engine.remaining())
            seen_context = receipt.get("input_context", context)
            packet = validate_packet(receipt["goal"], seen_context)
        except Exception as error:
            event = dict(getattr(error, "record", receipt or {}))
            event.update(action="plan", iteration=iteration,
                         api_request=event.get('api_request', not event.get('cache_hit', False)),
                         api_attempts=event.get('api_attempts', int(not event.get('cache_hit', False))),
                         error=f"{type(error).__name__}: {error}")
            event.setdefault('input_context', context)
            # An actual returned but invalid edit is a task-level observation,
            # not a service retry. The next bounded agent response sees why its
            # response was rejected; network/provider errors still stop at once.
            recoverable = (method == 'D2-AGENT-EDIT' and isinstance(error, PlanningError)
                           and bool(event.get('raw_output')) and 'request_id' in event)
            event['response_validation_rejected'] = recoverable
            events.append(event)
            if recoverable:
                if progress:
                    progress(iteration, event)
                if iteration+1 < rounds:
                    continue
                failed, stop = True, 'unresolved_response_validation'
            else:
                failed, stop = True, "planner_error"
            break
        receipt.update(action="plan", iteration=iteration, api_request=not receipt.get("cache_hit", False),
                       api_attempts=int(not receipt.get("cache_hit", False)))
        events.append(receipt)
        # Only feedback in this successful response's input was observed.
        if method == "D2-AGENT-EDIT":
            engine.observed_trials.update(seen_context.get("feedback_trial_ids", []))
        observed_before = set(engine.observed_trials)
        requested_stop = False
        for command in packet["commands"]:
            event = dict(action=command["kind"], iteration=iteration, command=command, diagnosis=packet["diagnosis"])
            try:
                kind = command["kind"]
                if kind == "propose":
                    result = engine.propose_edits(command["edits"], packet["state_id"])
                    event["result"] = result
                    if method == "D1-LLM-ONCE" and result["accept"]:
                        event["automatic_commit"] = engine.commit(result["trial_id"], command["reason"], single_shot=True)
                elif kind == "commit":
                    if command["trial_id"] not in observed_before:
                        raise ValueError("Commit requires feedback in an earlier completed model input")
                    event["result"] = engine.commit(command["trial_id"], command["reason"])
                elif kind == "inspect":
                    engine.focus_ids = list(command["target_ids"])
                    engine.focus_view = command["view"]
                    event["result"] = dict(focus_ids=engine.focus_ids, view=engine.focus_view)
                elif kind == "review":
                    engine.reviewed.update(command["target_ids"])
                    event["result"] = dict(review_ids=sorted(engine.reviewed))
                elif kind == "restore":
                    event["result"] = engine.restore()
                elif kind == "stop":
                    requested_stop = True
                    event["result"] = dict(reason=command["reason"], unmet_targets_disclosed=not engine.objective_status()["both_targets_met"])
                event["accepted"] = True
            except Exception as error:
                event.update(accepted=False, error=f"{type(error).__name__}: {error}")
            events.append(event)
            if progress:
                progress(iteration, event)
            if requested_stop:
                break
        if progress:
            progress(iteration, dict(action="round_complete", scene=scene_metrics(engine.current)))
        if requested_stop or method == "D1-LLM-ONCE":
            stop = "single_shot_complete" if method == "D1-LLM-ONCE" else "model_stop"
            break
    return events, stop, failed


def run(engine, method, planner=None, progress=None):
    if method in DIRECT_METHODS:
        if planner is None:
            raise ValueError("Direct model group needs a planner")
        events, stop, failed = run_direct(engine, planner, method, progress)
        final, trace = region.finish(engine, method, events, stop, failed)
    else:
        final, trace = region.run(engine, method, planner, progress)
    details = [d for r, _ in engine.trials.values() if r.get("tool_id") == "direct_edit" for d in r["details"]]
    trace.update(version=VERSION, goal_contract=goal_contract(engine.config),
                 scope="Direct model-authored geometry edits with shared verification; B1 retains strong regional rule search",
                 direct_edit_statistics=dict(atomic_batches=len(details), valid_batches=sum(d["accept"] for d in details),
                    response_validation_rejections=sum(bool(e.get('response_validation_rejected')) for e in trace['action_trace']),
                    committed_proposals=sum(bool(r.get("committed_to_scene")) for r, _ in engine.trials.values() if r.get("tool_id") == "direct_edit"),
                    semantic_edit_correctness=None, feedback_correction_rate=None,
                    independent_reference_status="N/A: requires independently annotated edit/ownership truth"))
    return final, trace
