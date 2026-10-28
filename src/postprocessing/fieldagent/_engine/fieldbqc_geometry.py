"""Bounded geometry repair tools. Runtime code never reads ground-truth labels.

Coordinates follow pixel-cell boundaries: pixel (row, col) is the unit square
[col, col+1] x [row, row+1]. Rasterio preserves polygon holes on round trips.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Callable

import cv2
import numpy as np
from rasterio.features import rasterize, shapes
from scipy.ndimage import distance_transform_edt
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from fieldbqc_agent import Parcel

ACTIONS = ("repair_geometry", "resolve_overlap", "coordinate_shared_boundary", "repair_holes",
           "repair_spurs_necks", "repair_gaps", "simplify_contour", "repair_tile_seams")
PROFILES = {"conservative": 0.5, "balanced": 1.0}


@dataclass(frozen=True)
class GeometryConfig:
    max_iterations: int = 24
    max_seconds: float = 120.0
    max_boundary_shift: float = 2.0
    max_area_change: float = 0.05
    min_raster_iou: float = 0.95
    max_edge_regression: float = 0.03
    simplify_tolerance: float = 1.0
    max_hole_area: float = 16.0
    max_gap_area: int = 24
    max_gap_width: float = 2.0
    min_vertices: int = 10
    target_vertices: int = 10
    simplify_search_steps: int = 24
    compact_final: bool = True
    color_tolerance: float = 20.0
    max_edge_barrier: float = 0.25
    max_actions_per_context: int = 64

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "compact_final":
                if type(value) is not bool:
                    raise ValueError("compact_final must be boolean")
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        for name in ("max_area_change", "min_raster_iou", "max_edge_regression", "max_edge_barrier"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        for name in ("max_iterations", "max_gap_area", "min_vertices", "max_actions_per_context",
                     "target_vertices", "simplify_search_steps"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.max_seconds <= 0 or self.max_actions_per_context < 1:
            raise ValueError("Time and context limits must be positive")
        if self.target_vertices < 3 or not 4 <= self.simplify_search_steps <= 128:
            raise ValueError("target_vertices must be >=3; search steps must be 4..128")


def polygons(geometry):
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if hasattr(geometry, "geoms"):
        return [p for part in geometry.geoms for p in polygons(part)]
    return []


def polygonal(geometry):
    parts = polygons(geometry)
    return unary_union(parts) if parts else Polygon()


def normalized(geometry):
    geometry = geometry if geometry.is_valid else shapely.make_valid(geometry)
    return geometry if isinstance(geometry, (Polygon, MultiPolygon)) else polygonal(geometry)


def mask_geometry(mask):
    array = np.asarray(mask, dtype=np.uint8)
    return polygonal(unary_union([shape(g) for g, value in shapes(array, mask=array.astype(bool)) if value]))


def geometry_mask(geometry, image_shape):
    geometry = normalized(geometry)
    return (rasterize([(mapping(geometry), 1)], out_shape=image_shape, fill=0, dtype="uint8").astype(bool)
            if not geometry.is_empty else np.zeros(image_shape, bool))


def vertices(geometry):
    return sum(len(p.exterior.coords) - 1 + sum(len(r.coords) - 1 for r in p.interiors)
               for p in polygons(geometry))


def holes(geometry):
    return sum(len(p.interiors) for p in polygons(geometry))


def exterior(geometry):
    return unary_union([p.exterior for p in polygons(geometry)])


@dataclass(frozen=True)
class GeometryInstance:
    identifier: str
    geometry: object
    score: float = 1.0
    cls: int = 0
    source_ids: tuple[str, ...] = ()

    def with_geometry(self, geometry):
        return replace(self, geometry=geometry)


def from_parcels(parcels):
    return [GeometryInstance(str(p.parcel_id), mask_geometry(p.mask), p.score, p.cls,
                             (str(p.parcel_id),)) for p in parcels]


def to_parcels(items, image_shape, mask_getter=None):
    return [Parcel(mask_getter(p) if mask_getter else geometry_mask(p.geometry, image_shape),
                   p.score, p.cls, parcel_id=p.identifier,
                   quality_valid=False, quality_source="geometry_pipeline") for p in items]


def scene_metrics(items):
    geoms = [normalized(p.geometry) for p in items]
    union = unary_union(geoms)
    return {"instances": len(items), "vertices": sum(vertices(p.geometry) for p in items),
            "invalid_geometries": sum(not p.geometry.is_valid or p.geometry.is_empty for p in items),
            "overlap_excess_area": max(0.0, sum(g.area for g in geoms) - union.area),
            "union_area": union.area, "holes": sum(holes(g) for g in geoms),
            "components": sum(len(polygons(g)) for g in geoms)}


class GeometryTools:
    def __init__(self, image, config=None, protected_mask=None, seam_pairs=None):
        self.image = np.asarray(image)
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("Expected an HxWx3 image")
        self.shape = self.image.shape[:2]
        self.config = config or GeometryConfig()
        self.protected = (np.zeros(self.shape, bool) if protected_mask is None
                          else np.asarray(protected_mask, dtype=bool))
        if self.protected.shape != self.shape:
            raise ValueError("Protected mask shape mismatch")
        self.seam_pairs = {tuple(sorted(pair)) for pair in (seam_pairs or [])}
        self._mask_cache = {}
        self._metrics_cache = {}
        self._diagnosis_key, self._diagnosis = None, []
        gray = cv2.cvtColor(np.clip(self.image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(gray, 50, 150) > 0
        self.edge_band = cv2.dilate(self.edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    def mask(self, item):
        if item.geometry not in self._mask_cache:
            if len(self._mask_cache) >= 512:
                self._mask_cache.pop(next(iter(self._mask_cache)))
            mask = geometry_mask(item.geometry, self.shape)
            mask.setflags(write=False)
            self._mask_cache[item.geometry] = mask
        return self._mask_cache[item.geometry]

    def metrics(self, items):
        key = tuple(items)
        if key not in self._metrics_cache:
            if len(self._metrics_cache) >= 64:
                self._metrics_cache.pop(next(iter(self._metrics_cache)))
            self._metrics_cache[key] = scene_metrics(items)
        return self._metrics_cache[key]

    def edge_alignment(self, mask):
        border = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        return float(np.count_nonzero(border & self.edge_band) / max(1, np.count_nonzero(border)))

    def image_support(self, region, neighbors):
        """Conservative color/edge heuristic, not a certified semantic classifier."""
        if not np.any(region) or np.any(region & self.protected):
            return False
        ring = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~region & neighbors
        if not np.any(ring) or float(self.image.std()) < 1.0:
            return False
        delta = np.linalg.norm(np.median(self.image[region], axis=0).astype(float) -
                               np.median(self.image[ring], axis=0).astype(float))
        barrier = float(np.mean(self.edges[region]))
        return delta <= self.config.color_tolerance and barrier <= self.config.max_edge_barrier

    def gap_regions(self, items):
        """Only enclosed small multi-instance gaps; open corridors stay untouched."""
        masks = [self.mask(p) for p in items]
        union = np.logical_or.reduce(masks) if masks else np.zeros(self.shape, bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats((~union).astype(np.uint8), 8)
        result = []
        for k in range(1, count):
            if stats[k, cv2.CC_STAT_AREA] > self.config.max_gap_area:
                continue
            region = labels == k
            if region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any():
                continue
            if float(distance_transform_edt(region).max()) * 2 > self.config.max_gap_width:
                continue
            band = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ids = [p.identifier for p, mask in zip(items, masks) if np.any(mask & band)]
            if len(ids) >= 2 and self.image_support(region, union):
                result.append((tuple(sorted(ids)), region))
        return result

    def diagnose(self, items):
        key = tuple(items)
        if key == self._diagnosis_key:
            return list(self._diagnosis)
        by_id = {p.identifier: p for p in items}
        if len(by_id) != len(items) or any(not p.identifier for p in items):
            raise ValueError("Geometry instance IDs must be unique nonempty strings")
        available = []
        def add(action, ids, severity, evidence):
            available.append({"action": action, "target_ids": sorted(ids),
                              "severity": float(severity), "evidence": evidence})
        geoms = [normalized(p.geometry) for p in items]
        tree = STRtree(geoms)
        for index, (item, geom) in enumerate(zip(items, geoms)):
            if not item.geometry.is_valid or item.geometry.is_empty:
                add("repair_geometry", [item.identifier], 100, {"valid": item.geometry.is_valid})
            if geom.is_empty:
                continue
            if vertices(item.geometry) > self.config.min_vertices:
                add("simplify_contour", [item.identifier], 10 + math.log1p(vertices(item.geometry)),
                    {"vertices": vertices(item.geometry)})
            for part in polygons(geom):
                for ring in part.interiors:
                    hole = Polygon(ring)
                    if hole.area <= self.config.max_hole_area and self.image_support(
                            geometry_mask(hole, self.shape), self.mask(item)):
                        add("repair_holes", [item.identifier], 30, {"small_hole_area": hole.area})
                        break
            mask = self.mask(item)
            opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
            if np.any(mask != opened):
                add("repair_spurs_necks", [item.identifier], 15,
                    {"opening_changed_pixels": int(np.count_nonzero(mask != opened))})
            for j in tree.query(geom.buffer(self.config.max_gap_width)):
                j = int(j)
                if j <= index:
                    continue
                other = items[j]
                if item.cls != other.cls:
                    continue
                pair = tuple(sorted((item.identifier, other.identifier)))
                overlap = geom.intersection(geoms[j]).area
                if pair in self.seam_pairs:
                    add("repair_tile_seams", pair, 95, {"reciprocal_overlap_match": True})
                elif overlap > 1e-8:
                    add("resolve_overlap", pair, 90 + min(5, overlap / max(1, min(geom.area, geoms[j].area))),
                        {"overlap_area": overlap})
                shared = geom.boundary.intersection(geoms[j].boundary)
                if overlap < 1e-8 and shared.length > 2 and len(shapely.get_coordinates(shared)) > 4:
                    add("coordinate_shared_boundary", pair, 40,
                        {"shared_length": shared.length, "shared_coordinates": len(shapely.get_coordinates(shared))})
        for ids, region in self.gap_regions(items):
            add("repair_gaps", ids, 25, {"gap_pixels": int(region.sum())})
        # Multiple small holes/regions can nominate the same action.
        unique = {}
        for entry in available:
            unique.setdefault((entry["action"], tuple(entry["target_ids"])), entry)
        result = sorted(unique.values(), key=lambda e: (-e["severity"], e["action"], e["target_ids"]))
        self._diagnosis_key, self._diagnosis = key, result
        return list(result)

    def _allocate(self, targets, domain, *, overlap_only):
        masks = [self.mask(p) for p in targets]
        occupied = np.sum(masks, axis=0)
        region = domain & ((occupied > 1) if overlap_only else (occupied == 0))
        if not region.any():
            raise ValueError("No eligible pixels for allocation")
        distances = []
        for p, mask in zip(targets, masks):
            anchor = mask & (occupied == 1)
            if not anchor.any():
                raise ValueError("Ambiguous ownership: target has no exclusive anchor")
            distances.append(distance_transform_edt(~anchor) - min(max(p.score, 0), 1) * 1e-3)
        values = np.stack(distances)
        if overlap_only:
            values = np.where(np.stack(masks), values, np.inf)
        owners = values.argmin(axis=0)
        updated = []
        for k, (p, mask) in enumerate(zip(targets, masks)):
            # Cut only disputed cells; preserve the original vector outside them.
            assigned = region & (owners == k)
            geom = normalized(p.geometry)
            if overlap_only:
                geom = geom.difference(mask_geometry(region & ~assigned))
            else:
                geom = geom.union(mask_geometry(assigned))
            updated.append(p.with_geometry(polygonal(geom)))
        return updated

    def propose(self, action, target_ids, items, profile="conservative"):
        if action not in ACTIONS or profile not in PROFILES:
            raise ValueError("Unknown tool or parameter profile")
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("Duplicate targets")
        lookup = {p.identifier: p for p in items}
        if not set(target_ids) <= lookup.keys() or not target_ids:
            raise ValueError("Unknown or empty targets")
        targets = [lookup[i] for i in target_ids]
        factor = PROFILES[profile]
        tolerance = self.config.simplify_tolerance * factor
        evidence = {"parameter_profile": profile}
        if action == "repair_geometry":
            updated = [p.with_geometry(normalized(p.geometry)) for p in targets]
        elif action == "simplify_contour":
            from fieldbqc_vertex_optimization import search_simplification
            return search_simplification(self, items, target_ids, profile)
        elif action == "resolve_overlap":
            union = unary_union([normalized(p.geometry) for p in targets])
            updated = self._allocate(targets, geometry_mask(union, self.shape), overlap_only=True)
        elif action == "coordinate_shared_boundary":
            geoms = np.asarray([normalized(p.geometry) for p in targets], dtype=object)
            # Raster polygons often encode the same straight edge with different
            # vertex subdivisions. Insert shared nodes without moving the boundary.
            reference = shapely.union_all(shapely.boundary(geoms))
            geoms = np.asarray([shapely.snap(g, reference, 1e-8) for g in geoms], dtype=object)
            if len(targets) < 2 or not shapely.coverage_is_valid(geoms):
                raise ValueError("Shared-boundary tool requires an existing valid coverage")
            result = shapely.coverage_simplify(geoms, tolerance, simplify_boundary=False)
            updated = [p.with_geometry(g) for p, g in zip(targets, result)]
        elif action == "repair_holes":
            updated = []
            for p in targets:
                parts = []
                for part in polygons(normalized(p.geometry)):
                    keep = []
                    for ring in part.interiors:
                        hole = Polygon(ring)
                        remove = hole.area <= self.config.max_hole_area * factor and self.image_support(
                            geometry_mask(hole, self.shape), self.mask(p))
                        if not remove:
                            keep.append(ring.coords)
                    parts.append(Polygon(part.exterior.coords, keep))
                updated.append(p.with_geometry(polygonal(unary_union(parts))))
        elif action == "repair_spurs_necks":
            updated = []
            kernel = np.ones((3, 3), np.uint8)
            for p in targets:
                mask = self.mask(p)
                opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
                proposed = mask_geometry(opened)
                # An opening that disconnects a neck is not a valid repair.
                if len(polygons(proposed)) != len(polygons(normalized(p.geometry))):
                    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
                    if not self.image_support(closed & ~mask, mask):
                        raise ValueError("Narrow connector cannot be adjusted without changing connectivity")
                    proposed = mask_geometry(closed)
                updated.append(p.with_geometry(proposed))
        elif action == "repair_gaps":
            regions = [region for ids, region in self.gap_regions(items) if set(ids) == set(target_ids)]
            if not regions:
                raise ValueError("No image-supported enclosed gap for this target group")
            region = np.logical_or.reduce(regions)
            updated = self._allocate(targets, region, overlap_only=False)
        else:
            pair = tuple(sorted(target_ids))
            if len(targets) != 2 or pair not in self.seam_pairs:
                raise ValueError("Seam repair requires a certified reciprocal tile match")
            union = polygonal(unary_union([normalized(p.geometry) for p in targets]))
            updated = [replace(targets[0], geometry=union,
                               source_ids=tuple(sorted({s for p in targets for s in p.source_ids})),
                               score=max(p.score for p in targets))]
            evidence["associated_source_ids"] = list(updated[0].source_ids)
        replacements = {p.identifier: p for p in updated}
        candidate = [replacements.get(p.identifier, p) for p in items
                     if p.identifier not in target_ids or p.identifier in replacements]
        return candidate, evidence

    def verify(self, before, candidate, action, target_ids):
        reasons, per_target = [], {}
        old, new = ({p.identifier: p for p in values} for values in (before, candidate))
        seam = action == "repair_tile_seams" and tuple(sorted(target_ids)) in self.seam_pairs
        expected = set(old) - (set(target_ids[1:]) if seam else set())
        if set(new) != expected:
            return {"accept": False, "reasons": ["instance_identity_changed"], "metrics": {}}
        ordered_ids = list(old)
        old_geoms = [normalized(old[i].geometry) for i in ordered_ids]
        new_geoms = [normalized(new[i].geometry) for i in ordered_ids if i in new]
        old_tree, new_tree = STRtree(old_geoms), STRtree(new_geoms)
        new_order = [i for i in ordered_ids if i in new]
        for identifier, p in old.items():
            if identifier not in target_ids and not p.geometry.equals_exact(new[identifier].geometry, 0):
                reasons.append("unrelated_instance_changed")
        a, b = self.metrics(before), self.metrics(candidate)
        if b["invalid_geometries"]:
            reasons.append("invalid_geometry")
        if b["overlap_excess_area"] > a["overlap_excess_area"] + 1e-7:
            reasons.append("new_overlap")
        if action in {"resolve_overlap", "coordinate_shared_boundary"}:
            u0 = unary_union([normalized(p.geometry) for p in before])
            u1 = unary_union([normalized(p.geometry) for p in candidate])
            if u0.symmetric_difference(u1).area > 1e-7:
                reasons.append("joint_coverage_changed")
        changed = False
        for identifier in target_ids:
            if identifier not in new:
                continue
            source = old[identifier]
            g0, g1 = normalized(source.geometry), normalized(new[identifier].geometry)
            if action == "simplify_contour" and vertices(g1) > vertices(g0):
                reasons.append("target_vertex_increase")
            neighbor_ids = {ordered_ids[int(i)] for i in old_tree.query(g0.envelope.union(g1.envelope))}
            neighbor_ids.update(new_order[int(i)] for i in new_tree.query(g0.envelope.union(g1.envelope)))
            if not source.geometry.equals_exact(new[identifier].geometry, 0):
                changed = True
            if g1.is_empty or not g1.is_valid:
                reasons.append("empty_or_invalid_target")
                continue
            x0, y0, x1, y1 = g1.bounds
            if x0 < 0 or y0 < 0 or x1 > self.shape[1] or y1 > self.shape[0]:
                reasons.append("geometry_outside_image")
            if not seam and len(polygons(g1)) != len(polygons(g0)):
                reasons.append("connectivity_changed")
            if not seam and action not in {"repair_holes", "repair_gaps", "repair_geometry"} and holes(g1) != holes(g0):
                reasons.append("hole_topology_changed")
            if not seam:
                for other_id in sorted(neighbor_ids):
                    other = old[other_id]
                    if other_id == identifier:
                        continue
                    before_overlap = g0.intersection(normalized(other.geometry)).area
                    after_overlap = g1.intersection(normalized(new[other_id].geometry)).area
                    if after_overlap > before_overlap + 1e-7:
                        reasons.append("new_neighbor_overlap")
            if action in {"simplify_contour", "repair_spurs_necks", "repair_holes", "repair_geometry"}:
                for other_id in sorted(neighbor_ids):
                    other = old[other_id]
                    if other_id == identifier:
                        continue
                    shared = g0.boundary.intersection(normalized(other.geometry).boundary)
                    if action == "simplify_contour" and other_id in target_ids:
                        shared1 = g1.boundary.intersection(normalized(new[other_id].geometry).boundary)
                        if shared.length > 1e-7:
                            if shared1.length <= 1e-7 or shared.hausdorff_distance(shared1) > self.config.max_boundary_shift + 1e-7:
                                reasons.append("joint_adjacency_changed")
                        elif shared1.length > 1e-7:
                            reasons.append("new_adjacency")
                        continue
                    if shared.length > 1e-7 and shared.difference(g1.boundary.buffer(1e-7)).length > 1e-6:
                        reasons.append("shared_boundary_changed_independently")
            mask0, mask1 = self.mask(source), self.mask(new[identifier])
            if np.any((mask0 != mask1) & self.protected):
                reasons.append("protected_region_changed")
            union = int(np.count_nonzero(mask0 | mask1))
            raster_iou = float(np.count_nonzero(mask0 & mask1) / max(1, union))
            area_change = abs(g1.area - g0.area) / max(g0.area, 1)
            shift = float(exterior(g0).hausdorff_distance(exterior(g1)))
            ring_shift = float(g0.boundary.hausdorff_distance(g1.boundary))
            edge0, edge1 = self.edge_alignment(mask0), self.edge_alignment(mask1)
            per_target[identifier] = {"area_change": area_change, "raster_iou": raster_iou,
                                      "max_exterior_shift": shift, "max_ring_shift": ring_shift,
                                      "edge_before": edge0, "edge_after": edge1}
            if not seam:
                if area_change > self.config.max_area_change:
                    reasons.append("excessive_area_change")
                if raster_iou < self.config.min_raster_iou:
                    reasons.append("raster_drift")
                if not math.isfinite(shift) or shift > self.config.max_boundary_shift:
                    reasons.append("boundary_drift")
                if action == "simplify_contour" and ring_shift > self.config.max_boundary_shift + 1e-7:
                    reasons.append("interior_boundary_drift")
                if edge1 < edge0 - self.config.max_edge_regression:
                    reasons.append("edge_regression")
        if action == "simplify_contour" and len(target_ids) > 1:
            unions = [normalized(unary_union([normalized(values[i].geometry) for i in target_ids]))
                      for values in (old, new)]
            if holes(unions[0]) != holes(unions[1]) or len(polygons(unions[0])) != len(polygons(unions[1])):
                reasons.append("joint_coverage_topology_changed")
        if seam:
            old_union = unary_union([normalized(p.geometry) for p in before])
            new_union = unary_union([normalized(p.geometry) for p in candidate])
            if old_union.symmetric_difference(new_union).area > 1e-7:
                reasons.append("seam_coverage_changed")
            changed = len(before) != len(candidate)
        if not changed:
            reasons.append("unchanged")
        gain = {"repair_geometry": b["invalid_geometries"] < a["invalid_geometries"],
                "resolve_overlap": b["overlap_excess_area"] < a["overlap_excess_area"] - 1e-7,
                "simplify_contour": b["vertices"] < a["vertices"],
                "coordinate_shared_boundary": b["vertices"] < a["vertices"],
                "repair_holes": b["holes"] < a["holes"],
                "repair_gaps": b["union_area"] > a["union_area"] + 1e-7,
                "repair_spurs_necks": b["vertices"] < a["vertices"],
                "repair_tile_seams": seam and len(candidate) == len(before) - 1}[action]
        if not gain:
            reasons.append("no_objective_gain")
        return {"accept": not reasons, "reasons": list(dict.fromkeys(reasons)) or ["verified_geometry_gain"],
                "metrics": {"before": a, "candidate": b, "targets": per_target}}


class GeometryAgent:
    """Numeric feedback loop with bounded local tools and target-scoped review."""
    def __init__(self, tools, planner=None, mode="rule", observer: Callable | None = None):
        if mode not in {"rule", "fixed", "baseline", "compact"}:
            raise ValueError("Unknown routing mode")
        self.tools, self.planner, self.mode, self.observer = tools, planner, mode, observer

    def run(self, items):
        current = list(items)
        trace, attempted, reviewed = [], set(), set()
        started = time.monotonic()
        stop = "action_budget"
        priorities = {action: index for index, action in enumerate(ACTIONS)}
        for iteration in range(0 if self.mode in {"baseline", "compact"} else self.tools.config.max_iterations):
            if time.monotonic() - started >= self.tools.config.max_seconds:
                stop = "time_budget"
                break
            available = [entry for entry in self.tools.diagnose(current)
                         if (entry["action"], tuple(entry["target_ids"])) not in attempted
                         and not (set(entry["target_ids"]) & reviewed)]
            if not available:
                stop = "no_eligible_tool"
                break
            available = available[:self.tools.config.max_actions_per_context]
            if self.mode == "fixed":
                available.sort(key=lambda e: (priorities[e["action"]], e["target_ids"]))
            goal = {"action": available[0]["action"], "target_ids": available[0]["target_ids"],
                    "parameter_profile": "conservative", "decision": "repair"}
            if self.planner is not None:
                context = {"instances": [{"parcel_id": p.identifier, "index": i,
                                           "severity": max((e["severity"] for e in available
                                                            if p.identifier in e["target_ids"]), default=0),
                                           "vertices": vertices(p.geometry), "area": normalized(p.geometry).area,
                                           "holes": holes(p.geometry), "valid": p.geometry.is_valid,
                                           "bbox": list(p.geometry.bounds), "class_id": p.cls,
                                           "confidence": p.score} for i, p in enumerate(current)],
                           "available_actions": available, "policy": asdict(self.tools.config),
                           "remaining_budget": {"iterations": self.tools.config.max_iterations - iteration,
                                                "seconds": self.tools.config.max_seconds - (time.monotonic() - started)},
                           "history": [e for e in trace if e["action"] != "plan"][-6:],
                           "objectives": ["reduce_geometry_defects", "minimize_vertices_with_bounded_drift"],
                           "coordinate_units": "pixels", "ground_truth_available": False}
                try:
                    from fieldbqc_geometry_planner import validate_geometry_goal
                    result = self.planner.plan(context, self.tools.image,
                                               to_parcels(current, self.tools.shape, self.tools.mask),
                                               context["remaining_budget"]["seconds"])
                    goal = validate_geometry_goal(result["goal"], context)
                    trace.append({"iteration": iteration, "action": "plan", **result})
                    if goal["decision"] == "keep":
                        stop = "planner_keep"
                        break
                    if goal["decision"] == "review":
                        reviewed.update(goal["target_ids"])
                        trace.append({"iteration": iteration, "action": "local_review",
                                      "target_ids": goal["target_ids"], "reason": goal["reason"]})
                        continue
                except Exception as error:
                    trace.append({"iteration": iteration, "action": "plan", "fallback": "rule",
                                  **getattr(error, "record", {}), "error": f"{type(error).__name__}: {error}"})
            if time.monotonic() - started >= self.tools.config.max_seconds:
                stop = "time_budget"
                break
            action, ids = goal["action"], goal["target_ids"]
            # One attempt per action/group in the current state avoids no-gain cycles.
            key = (action, tuple(sorted(ids)))
            attempted.add(key)
            tick = time.monotonic()
            try:
                candidate, evidence = self.tools.propose(action, ids, current, goal["parameter_profile"])
                decision = self.tools.verify(current, candidate, action, ids)
                if self.observer:
                    self.observer(iteration, current, candidate, action, ids, decision)
                trace.append({"iteration": iteration, "action": action, "target_ids": ids,
                              "accepted": decision["accept"], "rolled_back": not decision["accept"],
                              "reasons": decision["reasons"], "metrics": decision["metrics"],
                              "evidence": evidence, "elapsed_seconds": time.monotonic() - tick})
                if decision["accept"]:
                    current = candidate
                    # Successful changes invalidate old failed candidates in the local neighborhood.
                    attempted = {k for k in attempted if not set(k[1]) & set(ids)} | {key}
            except Exception as error:
                trace.append({"iteration": iteration, "action": action, "target_ids": ids,
                              "accepted": False, "rolled_back": True, "reasons": ["tool_error"],
                              "error": f"{type(error).__name__}: {error}"})
        from fieldbqc_vertex_optimization import compact_scene, vertex_report
        compact_status = "disabled"
        if (self.mode != "baseline" and (self.tools.config.compact_final or self.mode == "compact")
                and stop != "planner_keep"):
            current, extra, compact_status = compact_scene(
                self.tools, current, items, reviewed, started + self.tools.config.max_seconds,
                self.observer, self.tools.config.max_iterations)
            trace.extend(extra)
        unresolved = self.tools.diagnose(current)
        return current, {"action_trace": trace, "initial": scene_metrics(items), "final": scene_metrics(current),
                         "stop_reason": stop, "review_ids": sorted(reviewed), "unresolved": unresolved,
                         "needs_review": bool(unresolved or reviewed),
                         "compaction_status": compact_status,
                         "vertex_report": vertex_report(items, current, self.tools.config.target_vertices, trace, reviewed),
                         "postprocess_seconds": time.monotonic() - started}


def feature_collection(items):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"instance_id": p.identifier, "class_id": p.cls,
                                              "score": p.score, "source_ids": list(p.source_ids),
                                              "vertex_count": vertices(p.geometry)},
         "geometry": mapping(p.geometry)} for p in items]}
