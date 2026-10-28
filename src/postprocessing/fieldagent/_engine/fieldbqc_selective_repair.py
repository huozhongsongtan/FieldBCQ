"""V5 selective repair: shared finite candidates, independent controllers, no GT.

Legacy geometry experiments are intentionally left reproducible. This module has
four tool classes, no vertex-count target and no mandatory final compaction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from fieldbqc_geometry import (GeometryConfig, GeometryTools, geometry_mask, holes,
                              normalized, polygons, scene_metrics, to_parcels, vertices)
from fieldbqc_planner import APIPlanner, visual_inputs
from fieldbqc_vertex_optimization import anchor_reasons, contact_groups

VERSION = "selective-four-tools-v5.1"
TOOLS = ("simplify_contour", "resolve_overlap", "repair_holes", "repair_geometry")


@dataclass(frozen=True)
class RepairConfig:
    decisions: int = 8
    seconds: float = 300.0
    candidate_budget: int = 192
    menu_size: int = 24
    max_boundary_shift: float = 2.0
    max_area_change: float = 0.05
    min_raster_iou: float = 0.95
    max_edge_regression: float = 0.03
    max_hole_area: float = 16.0

    def __post_init__(self):
        for name in ("decisions", "candidate_budget", "menu_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name}")
        if self.max_area_change > 1 or self.min_raster_iou > 1 or self.max_edge_regression > 1:
            raise ValueError("Fraction thresholds must be <=1")

    def geometry_config(self):
        return GeometryConfig(max_iterations=self.decisions, max_seconds=self.seconds,
                              max_boundary_shift=self.max_boundary_shift,
                              max_area_change=self.max_area_change, min_raster_iou=self.min_raster_iou,
                              max_edge_regression=self.max_edge_regression, max_hole_area=self.max_hole_area,
                              min_vertices=3, target_vertices=3, compact_final=False)


def state_key(items):
    digest = hashlib.sha256()
    for p in items:
        digest.update(p.identifier.encode())
        digest.update(p.geometry.wkb)
    return digest.hexdigest()


def regularize_ring(ring, radius):
    """Ordered chain segmentation, TLS line fitting and bounded intersections.

    DP supplies segmentation anchors only; fitted segment intersections supply
    the new corners. Near-parallel lines fall back to their original anchor.
    Every output is subsequently validated; curves are not forced rectangular.
    """
    points = np.asarray(ring.coords[:-1], dtype=float)
    if len(points) < 4:
        return points
    anchors = cv2.approxPolyDP(points.astype(np.float32).reshape(-1, 1, 2),
                              max(radius * .5, 1e-7), True).reshape(-1, 2)
    indices = sorted(set(int(np.argmin(np.linalg.norm(points - a, axis=1))) for a in anchors))
    if len(indices) < 3:
        return points
    lines = []
    for i, first in enumerate(indices):
        last = indices[(i + 1) % len(indices)]
        chain = points[first:last + 1] if last > first else np.vstack((points[first:], points[:last + 1]))
        center = chain.mean(axis=0)
        centered = chain - center
        xx = float(np.sum(centered[:,0] ** 2))
        yy = float(np.sum(centered[:,1] ** 2))
        xy = float(np.sum(centered[:,0] * centered[:,1]))
        angle = .5 * math.atan2(2*xy, xx-yy)
        lines.append((center, np.array([math.cos(angle), math.sin(angle)])))
    corners = []
    for i, index in enumerate(indices):
        c0, d0 = lines[i - 1]
        c1, d1 = lines[i]
        determinant = float(d0[0]*d1[1]-d0[1]*d1[0])
        corner = points[index]
        if abs(determinant) > math.sin(math.radians(5)):
            displacement = c1-c0
            distance = (displacement[0]*d1[1]-displacement[1]*d1[0])/determinant
            intersection = c0 + distance*d0
            if np.linalg.norm(intersection - corner) <= radius + 1e-8:
                corner = intersection
        corners.append(corner)
    # Merge almost-collinear fitted segments only within a distance tolerance.
    corners = list(corners)
    changed = True
    while changed and len(corners) > 3:
        changed = False
        for i in range(len(corners)):
            a, b, c = corners[i - 1], corners[i], corners[(i + 1) % len(corners)]
            u, v = b - a, c - b
            product = np.linalg.norm(u) * np.linalg.norm(v)
            angle = math.degrees(math.acos(float(np.clip(np.dot(u, v) / max(product, 1e-12), -1, 1))))
            line = c - a
            residual = abs(float(line[0] * (b-a)[1] - line[1] * (b-a)[0])) / max(np.linalg.norm(line), 1e-12)
            if angle <= 5 and residual <= radius * .25:
                corners.pop(i)
                changed = True
                break
    return np.asarray(corners)


def regularize_geometry(geometry, radius):
    result = []
    for part in polygons(geometry):
        result.append(Polygon(regularize_ring(part.exterior, radius),
                              [regularize_ring(r, radius) for r in part.interiors]))
    # Do not silently make_valid an invalid fitted polygon.
    if any(not p.is_valid or p.is_empty for p in result):
        raise ValueError("Line assembly produced invalid geometry")
    return result[0] if len(result) == 1 else shapely.MultiPolygon(result)


class CandidateEngine:
    def __init__(self, image, original, config=None, contour_algorithms=("reg", "dp", "vw"),
                 enabled=TOOLS, protected=None):
        self.config = config or RepairConfig()
        self.tools = GeometryTools(image, self.config.geometry_config(), protected)
        self.original = list(original)
        self.contour_algorithms = tuple(contour_algorithms)
        self.enabled = set(enabled)
        if not self.enabled <= set(TOOLS) or not set(self.contour_algorithms) <= {"reg", "dp", "vw"}:
            raise ValueError("Unknown tool or contour algorithm")
        self.generated = 0
        self.cache, self.snapshots = {}, {}
        self.screened = []

    def remember(self, items):
        for p in items:
            values = self.snapshots.setdefault(p.identifier, [])
            if not any(g.equals_exact(p.geometry, 0) for g in values):
                values.append(p.geometry)

    def descriptors(self, items, reviewed):
        available, screened = [], []
        lookup = {p.identifier: p for p in items}
        geoms = [normalized(p.geometry) for p in items]
        tree = STRtree(geoms)
        def add(action, ids, severity, evidence, algorithm="native"):
            if action in self.enabled and not set(ids) & reviewed:
                for profile in ("conservative", "balanced"):
                    available.append(dict(action=action, target_ids=sorted(ids), severity=severity,
                                          evidence=evidence, algorithm=algorithm, profile=profile))
        for i, p in enumerate(items):
            if not p.geometry.is_valid or p.geometry.is_empty:
                add("repair_geometry", [p.identifier], 100, {"valid": p.geometry.is_valid})
            for part in polygons(geoms[i]):
                if any(Polygon(r).area <= self.config.max_hole_area for r in part.interiors):
                    add("repair_holes", [p.identifier], 50, {"holes": holes(p.geometry)})
                    break
            if "resolve_overlap" not in self.enabled:
                continue
            for j in tree.query(geoms[i]):
                j = int(j)
                if j <= i or p.cls != items[j].cls:
                    continue
                q, g, h = items[j], geoms[i], geoms[j]
                area = g.intersection(h).area
                if area <= 1e-8:
                    continue
                reason = None
                if area > self.config.max_area_change * (g.area + h.area) + 1e-7:
                    reason = "overlap_exceeds_pair_area_budget"
                elif not np.any(self.tools.mask(p) & ~self.tools.mask(q)) or not np.any(self.tools.mask(q) & ~self.tools.mask(p)):
                    reason = "missing_exclusive_anchor"
                if reason:
                    screened.append(dict(action="resolve_overlap", target_ids=[p.identifier, q.identifier],
                                         overlap_area=area, reason=reason))
                else:
                    add("resolve_overlap", [p.identifier, q.identifier], 80, {"overlap_area": area})
        if "simplify_contour" in self.enabled:
            # Joint groups protect existing common boundaries. Singleton fallbacks
            # retain exactly the same boundary validation as other candidates.
            groups = contact_groups(items)
            groups += [[p.identifier] for p in items if any(p.identifier in g and len(g) > 1 for g in groups)]
            for ids in groups:
                count = sum(vertices(lookup[uid].geometry) for uid in ids)
                if count <= sum(3 * (len(polygons(lookup[uid].geometry)) + holes(lookup[uid].geometry)) for uid in ids):
                    continue
                for algorithm in self.contour_algorithms:
                    add("simplify_contour", ids, 20 + math.log1p(count), {"vertices": count}, algorithm)
        # Recovery is a mode of the same original tool, available to both
        # controllers. It never restores unrelated instances.
        for p in items:
            if p.identifier in reviewed:
                continue
            history = self.snapshots.get(p.identifier, [])
            recoverable = history[:1] + history[-2:] if len(history) > 2 else history
            for index, geom in enumerate(recoverable):
                if not p.geometry.equals_exact(geom, 0):
                    available.append(dict(action="repair_geometry", target_ids=[p.identifier], severity=10,
                                          evidence={"restore_wkb": geom.wkb_hex},
                                          algorithm="restore", profile="conservative"))
        available.sort(key=lambda a: (-a["severity"], a["target_ids"], a["profile"], a["algorithm"]))
        self.screened = screened
        return available

    @staticmethod
    def descriptor_key(desc):
        return hashlib.sha256(repr((desc["action"], desc["target_ids"], desc["algorithm"],
                                    desc["profile"], desc["evidence"].get("restore_wkb"))).encode()).hexdigest()[:16]

    def propose(self, items, desc):
        action, ids = desc["action"], desc["target_ids"]
        algorithm = desc["algorithm"]
        if algorithm == "restore":
            geom = shapely.from_wkb(bytes.fromhex(desc["evidence"]["restore_wkb"]))
            return [p.with_geometry(geom) if p.identifier == ids[0] else p for p in items]
        if action != "simplify_contour":
            return self.tools.propose(action, ids, items, desc["profile"])[0]
        radius = self.config.max_boundary_shift * (.5 if desc["profile"] == "conservative" else 1)
        lookup = {p.identifier: p for p in items}
        geoms = [normalized(lookup[uid].geometry) for uid in ids]
        if algorithm == "reg":
            results = [regularize_geometry(g, radius) for g in geoms]
        elif algorithm == "dp":
            results = [g.simplify(radius, preserve_topology=True) for g in geoms]
        else:
            array = np.asarray(geoms, dtype=object)
            if len(ids) > 1:
                array = np.asarray([shapely.snap(g, shapely.union_all(shapely.boundary(array)), 1e-8) for g in array])
                if not shapely.coverage_is_valid(array):
                    raise ValueError("VW requires a valid noded coverage")
            results = shapely.coverage_simplify(array, radius, simplify_boundary=True)
        replacements = dict(zip(ids, results))
        return [p.with_geometry(replacements[p.identifier]) if p.identifier in replacements else p for p in items]

    def verify(self, before, candidate, desc):
        if len(candidate) != len(before) or len({p.identifier for p in candidate}) != len(candidate):
            return dict(accept=False,reasons=["instance_identity_changed"],metrics={})
        verdict = self.tools.verify(before, candidate, desc["action"], desc["target_ids"])
        reasons = [] if verdict["accept"] else list(verdict["reasons"])
        if desc["algorithm"] == "restore":
            reasons = [r for r in reasons if r != "no_objective_gain"]
        # Multiple invalid inputs can be repaired one at a time; unrelated
        # invalid inputs do not turn all individual repairs into dead ends.
        lookup = {p.identifier: p for p in candidate}
        if all(lookup[uid].geometry.is_valid and not lookup[uid].geometry.is_empty for uid in desc["target_ids"]):
            if scene_metrics(candidate)["invalid_geometries"] <= scene_metrics(before)["invalid_geometries"]:
                reasons = [r for r in reasons if r != "invalid_geometry"]
        reasons += anchor_reasons(self.tools, self.original, candidate, desc["target_ids"])
        for a, b in zip(before, candidate):
            if (a.identifier, a.cls, a.source_ids, a.score) != (b.identifier, b.cls, b.source_ids, b.score):
                reasons.append("instance_metadata_changed")
        return dict(accept=not reasons, reasons=sorted(set(reasons)), metrics=verdict["metrics"])

    def prepare(self, items, desc):
        key = (state_key(items), self.descriptor_key(desc))
        if key in self.cache:
            return self.cache[key]
        if self.generated >= self.config.candidate_budget:
            return None
        self.generated += 1
        record = dict(desc, choice_id=self.descriptor_key(desc))
        record["evidence"] = {k: v for k, v in desc["evidence"].items() if k != "restore_wkb"}
        try:
            candidate = self.propose(items, desc)
            verdict = self.verify(items, candidate, desc)
            record["verification"] = verdict
            record["candidate_metrics"] = scene_metrics(candidate)
            old, new = ({p.identifier: p for p in values} for values in (items, candidate))
            old_n = sum(vertices(old[uid].geometry) for uid in desc["target_ids"])
            new_n = sum(vertices(new[uid].geometry) for uid in desc["target_ids"])
            per = verdict["metrics"].get("targets", {}).values()
            risk = max((max(1-r["raster_iou"], r["area_change"]) for r in per), default=0)
            edge_gain = sum(r["edge_after"]-r["edge_before"] for r in verdict["metrics"].get("targets", {}).values())
            record.update(vertex_gain=old_n-new_n, compression=(old_n-new_n)/max(1, old_n),
                          risk=risk, edge_gain=edge_gain)
            # Predeclared, GT-free strong-rule ordering: objective gain minus
            # geometric risk, with small preference for image-edge improvement.
            objective = {"simplify_contour": 1, "resolve_overlap": 2,
                         "repair_holes": 1.2, "repair_geometry": 3}[desc["action"]]
            record["utility"] = ((objective + record["compression"] + 2*edge_gain - 10*risk)
                                 if desc["algorithm"] != "restore" else 2*edge_gain - 10*risk - .1)
        except Exception as error:
            candidate = items
            record.update(verification={"accept": False, "reasons": ["tool_error"], "metrics": {}},
                          error=f"{type(error).__name__}: {error}", utility=-1e6)
        self.cache[key] = (record, candidate, desc)
        return self.cache[key]

    def menu(self, items, reviewed, attempted, deadline):
        options = []
        key = state_key(items)
        descriptors = self.descriptors(items, reviewed)
        # Examine a bounded descriptor window each step, retaining rejection
        # records for both controllers. Unchanged cached states cost no new work.
        examined = 0
        for desc in descriptors:
            if time.monotonic() >= deadline:
                break
            if (key, self.descriptor_key(desc)) in attempted:
                continue
            result = self.prepare(items, desc)
            if result is None:
                continue
            record = result[0]
            examined += 1
            if record["verification"]["accept"]:
                options.append(result)
            else:
                attempted.add((key, record["choice_id"]))
            if len(options) >= self.config.menu_size or examined >= self.config.menu_size * 3:
                break
        return sorted(options, key=lambda row: (-row[0]["utility"], row[0]["choice_id"]))


CHOICE = {"type": "object", "additionalProperties": False, "properties": {
    "decision": {"type": "string", "enum": ["repair", "review", "stop"]},
    "candidate_id": {"type": ["string", "null"]},
    "review_ids": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string"}}, "required": ["decision", "candidate_id", "review_ids", "reason"]}
PACKET = {"type": "object", "additionalProperties": False,
          "properties": {"choices": {"type": "array", "items": CHOICE}}, "required": ["choices"]}
PROMPT = """You select agricultural parcel vector repairs. Return the supplied JSON schema.
The SAME four local tools, candidates, image evidence and hard limits are available
to the rule controller. Never generate coordinates or relax constraints. There is
NO fixed vertex-count goal and NO automatic final simplification. Prefer fewer
vertices only while protecting actual boundaries, curves, holes and parcel identity.
Candidates passed numeric checks, NOT semantic correctness. Inspect raw image,
initial/current overlays and numbered candidate previews. A small hole may be a
real pond; a bend may be a real field corner. Use stop to retain current geometry,
or review_ids to defer uncertain objects. Finite candidate IDs specify object,
tool, algorithm and strength. Use feedback and previous rejections; do not repeat
failed edits. Geometry verification will recheck edits before commit. Ground truth
is unavailable. Candidate utility is a fallible geometry/image heuristic, not a
label or command. Restore candidates recover only their named parcel from accepted
history and may increase vertices. Output short reasons tied to observable evidence.
"""


def validate_packet(packet, context):
    if not isinstance(packet, dict) or set(packet) != {"choices"}:
        raise ValueError("Expected choices packet")
    choices = packet["choices"]
    if not isinstance(choices, list) or not 1 <= len(choices) <= context["plan_limit"]:
        raise ValueError("Invalid plan length")
    available = {a["choice_id"] for a in context["available_actions"]}
    known = {p["parcel_id"] for p in context["instances"]}
    seen = set()
    for choice in choices:
        if not isinstance(choice, dict) or set(choice) != set(CHOICE["required"]):
            raise ValueError("Invalid choice fields")
        if not isinstance(choice["reason"], str) or not 1 <= len(choice["reason"]) <= 1500:
            raise ValueError("Invalid reason")
        ids = choice["review_ids"]
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)) or not set(ids) <= known:
            raise ValueError("Invalid review IDs")
        if choice["decision"] == "repair":
            if choice["candidate_id"] not in available or ids or choice["candidate_id"] in seen:
                raise ValueError("Unavailable or repeated candidate")
            seen.add(choice["candidate_id"])
        elif choice["decision"] not in {"review", "stop"} or choice["candidate_id"] is not None:
            raise ValueError("Invalid decision")
        elif (choice["decision"] == "review" and not ids) or (choice["decision"] == "stop" and ids):
            raise ValueError("Review requires targets; stop has none")
    return packet


class SelectivePlanner(APIPlanner):
    def __init__(self, config, engine, cache, once=False):
        self.engine, self.once = engine, once
        self.preview_menu, self.initial, self.current = [], [], []
        self.queue = None
        instruction = ("Return an ordered plan from this initial observation only; no replanning."
                       if once else "Return exactly ONE choice; the next request includes execution feedback.")
        super().__init__(config, cache, schema=PACKET, system_prompt=PROMPT + instruction,
                         validator=validate_packet, prompt_version=VERSION + ("-once" if once else "-loop"),
                         visualizer=self.images)

    def images(self, image, parcels):
        urls = visual_inputs(image, parcels)
        initial = to_parcels(self.initial, self.engine.tools.shape)
        urls.extend(visual_inputs(image, initial)[1:])
        # Candidate montage uses the SAME current geometry as numeric context.
        # Raw crop, current crop, proposed crop, changed-cell crop are numbered.
        boards = []
        for index, (record, candidate, _) in enumerate(self.preview_menu[:8]):
            ids = set(record["target_ids"])
            before = [p for p in self.current if p.identifier in ids]
            after = [p for p in candidate if p.identifier in ids]
            occupied = np.logical_or.reduce([self.engine.tools.mask(p) for p in before + after])
            yy, xx = np.nonzero(occupied)
            if not len(yy):
                continue
            y0, y1 = max(0, yy.min()-8), min(image.shape[0], yy.max()+9)
            x0, x1 = max(0, xx.min()-8), min(image.shape[1], xx.max()+9)
            raw = image[y0:y1, x0:x1].copy()
            parts = [raw.copy() for _ in range(4)]
            m0 = np.logical_or.reduce([self.engine.tools.mask(p) for p in before])[y0:y1, x0:x1]
            m1 = np.logical_or.reduce([self.engine.tools.mask(p) for p in after])[y0:y1, x0:x1]
            # Separate instance contours retain visibility of overlap ownership.
            for panel, targets in ((1, before), (2, after)):
                for k, p in enumerate(targets):
                    mask = self.engine.tools.mask(p)[y0:y1, x0:x1].astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(parts[panel], contours, -1, ((0,255,255) if k % 2 else (255,0,255)), 1)
            changed = np.zeros(m0.shape, bool)
            for a, b in zip(before, after):
                changed |= (self.engine.tools.mask(a) != self.engine.tools.mask(b))[y0:y1, x0:x1]
            parts[3][changed] = (0, 0, 255)
            row = np.hstack([cv2.resize(p, (160, 160), interpolation=cv2.INTER_AREA) for p in parts])
            label = f"{index}: {record['choice_id']} {record['action']} {record['algorithm']}/{record['profile']}"
            row = cv2.copyMakeBorder(row, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
            cv2.putText(row, label, (3,15), cv2.FONT_HERSHEY_SIMPLEX, .34, (255,255,255), 1)
            boards.append(row)
        if boards:
            # visual_inputs with no parcels produces raw and identical overlay;
            # only one raw montage image is transmitted.
            urls.append(visual_inputs(np.vstack(boards), [], max_side=1600)[0])
        return urls

    def choose(self, context, items, menu, seconds):
        if self.once and self.queue is not None:
            if self.queue:
                return self.queue.pop(0), {"api_request": False}
            return {"decision": "stop", "candidate_id": None, "review_ids": [],
                    "reason": "Frozen plan exhausted"}, {"api_request": False, "plan_exhausted": True}
        self.preview_menu, self.current = menu, items
        context = dict(context, plan_limit=context["remaining_budget"]["iterations"] if self.once else 1,
                       preview_candidate_ids=[r[0]["choice_id"] for r in menu[:8]])
        result = APIPlanner.plan(self, context, self.engine.tools.image,
                                 to_parcels(items, self.engine.tools.shape), seconds)
        result = dict(result, api_attempts=int(not result.get("cache_hit", False)))
        choices = result["goal"]["choices"]
        if self.once:
            self.queue = list(choices[1:])
        return choices[0], dict(result, api_request=not result.get("cache_hit", False))


def run_controller(engine, method, planner=None, observer=None):
    if method not in {"G0", "G1-RULE", "G2-ONCE", "G2-LOOP", "S-DP", "S-VW", "S-REG"}:
        raise ValueError("Unknown method")
    current = list(engine.original)
    engine.remember(current)
    if planner:
        planner.initial = list(current)
    started = time.monotonic()
    deadline = started + engine.config.seconds
    trace, attempted, reviewed = [], set(), set()
    initial_menu, stop = None, "decision_budget"
    for iteration in range(0 if method == "G0" else engine.config.decisions):
        if time.monotonic() >= deadline:
            stop = "time_budget"
            break
        # Frozen plans use only initial descriptors. Revalidation observes the
        # current state solely to accept/skip, never to invent replacement goals.
        frozen = planner is not None and planner.once and initial_menu is not None
        menu = initial_menu if frozen else engine.menu(current, reviewed, attempted, deadline)
        if initial_menu is None:
            initial_menu = menu
        if not menu:
            stop = "no_verified_candidate"
            break
        context = {"instances": [{"parcel_id": p.identifier, "index": i, "severity": vertices(p.geometry),
                                   "vertices": vertices(p.geometry), "area": normalized(p.geometry).area,
                                   "holes": holes(p.geometry), "valid": p.geometry.is_valid,
                                   "bbox": list(p.geometry.bounds)} for i, p in enumerate(current)],
                   "available_actions": [r[0] for r in menu], "policy": asdict(engine.config),
                   "remaining_budget": {"iterations": engine.config.decisions-iteration,
                                        "seconds": max(0, deadline-time.monotonic()),
                                        "candidates": engine.config.candidate_budget-engine.generated},
                   "history": [{k:v for k,v in e.items() if k in {"iteration","action","target_ids","accepted","reasons","controller_reason","choice"}}
                               for e in trace if e["action"] != "plan"][-8:], "screened_infeasible": engine.screened,
                   "recent_rejections": [r[0] for r in list(engine.cache.values())[-36:] if not r[0]["verification"]["accept"]],
                   "coordinate_units": "pixels", "ground_truth_available": False}
        if time.monotonic() >= deadline:
            stop = "time_budget"
            break
        if planner:
            try:
                choice, receipt = planner.choose(context, current, menu, deadline-time.monotonic())
                trace.append(dict(iteration=iteration, action="plan", **receipt))
            except Exception as error:
                # No silent rule fallback: a failed model group is explicitly
                # incomplete for controller-effect claims.
                receipt = dict(getattr(error, "record", {}))
                receipt.setdefault("input_context",context)
                receipt.update(iteration=iteration, action="plan", api_request=True,
                               error=f"{type(error).__name__}: {error}")
                trace.append(receipt)
                stop = "planner_error"
                break
        else:
            best = menu[0][0]
            choice = dict(decision="repair" if best["utility"] > 0 else "stop",
                          candidate_id=best["choice_id"], review_ids=[], reason="Frozen GT-free utility ranking")
        if choice["decision"] == "stop":
            stop = "frozen_plan_exhausted" if planner and planner.once and not planner.queue else "controller_stop"
            trace.append(dict(iteration=iteration, action="stop", choice=choice))
            break
        if choice["decision"] == "review":
            reviewed.update(choice["review_ids"])
            trace.append(dict(iteration=iteration, action="review", choice=choice))
            continue
        selected = next((r for r in menu if r[0]["choice_id"] == choice["candidate_id"]), None)
        if selected is None:
            raise ValueError("Controller selected an unavailable candidate")
        record, candidate, desc = selected
        if frozen:
            if set(desc["target_ids"]) & reviewed:
                trace.append(dict(iteration=iteration, action="stale_plan_skip", reason="reviewed_target"))
                continue
            selected = engine.prepare(current, desc)
            if selected is None:
                stop = "candidate_budget"
                break
            record, candidate, desc = selected
        key = (state_key(current), record["choice_id"])
        if key in attempted:
            trace.append(dict(iteration=iteration, action="stale_plan_skip", reason="already_attempted"))
            continue
        attempted.add(key)
        verdict = engine.verify(current, candidate, desc)
        event = dict(iteration=iteration, action=desc["action"], target_ids=desc["target_ids"],
                     choice_id=record["choice_id"], algorithm=desc["algorithm"], profile=desc["profile"],
                     accepted=verdict["accept"], rolled_back=not verdict["accept"], reasons=verdict["reasons"],
                     metrics=verdict["metrics"], controller_reason=choice["reason"], evidence=record)
        trace.append(event)
        if observer:
            observer(iteration, current, candidate, event)
        if verdict["accept"]:
            current = list(candidate)
            engine.remember(current)
    return current, dict(version=VERSION, method=method, initial=scene_metrics(engine.original),
                         final=scene_metrics(current), action_trace=trace, stop_reason=stop,
                         review_ids=sorted(reviewed), screened_infeasible=engine.screened,
                         candidate_evaluations=engine.generated,
                         rejected_candidates=sum(not r[0]["verification"]["accept"] for r in engine.cache.values()),
                         planner_failed=stop == "planner_error", postprocess_seconds=time.monotonic()-started,
                         compaction_status="not_part_of_v5_protocol")
