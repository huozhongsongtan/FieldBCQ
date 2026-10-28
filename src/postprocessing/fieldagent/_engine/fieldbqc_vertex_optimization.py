"""Finite, verified vertex minimization in pixel coordinates (no ground truth)."""
from __future__ import annotations

from collections import Counter
import time

import numpy as np
import shapely
from shapely.ops import unary_union
from shapely.strtree import STRtree

from fieldbqc_geometry import normalized, polygons, holes, vertices, exterior, PROFILES


def structural_floor(geometry):
    # Each retained exterior and interior ring needs at least three vertices.
    return 3 * (len(polygons(geometry)) + holes(geometry))


def contact_groups(items):
    """Connected shared edges must be simplified together, including across classes."""
    geoms = [normalized(p.geometry) for p in items]
    tree = STRtree(geoms)
    parent = list(range(len(items)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, geom in enumerate(geoms):
        for j in tree.query(geom):
            j = int(j)
            if j > i and geom.intersection(geoms[j]).area <= 1e-8:
                if geom.boundary.intersection(geoms[j].boundary).length > 1e-7:
                    parent[root(j)] = root(i)
    groups = {}
    for i, p in enumerate(items):
        groups.setdefault(root(i), []).append(p.identifier)
    return sorted(groups.values(), key=lambda ids: tuple(ids))


def anchor_reasons(tools, original, candidate, ids):
    """Final compaction may not accumulate drift from earlier repair steps."""
    old = {p.identifier: p for p in original}
    new = {p.identifier: p for p in candidate}
    reasons = []
    for uid in ids:
        if uid not in old:  # A certified tile merge has its own coverage check.
            continue
        a, b = old[uid], new[uid]
        if a.source_ids != b.source_ids:
            continue
        g0, g1 = normalized(a.geometry), normalized(b.geometry)
        m0, m1 = tools.mask(a), tools.mask(b)
        if exterior(g0).hausdorff_distance(exterior(g1)) > tools.config.max_boundary_shift + 1e-7:
            reasons.append("cumulative_boundary_drift")
        if holes(g0) == holes(g1) and g0.boundary.hausdorff_distance(g1.boundary) > tools.config.max_boundary_shift + 1e-7:
            reasons.append("cumulative_ring_drift")
        if abs(g1.area-g0.area) / max(g0.area, 1) > tools.config.max_area_change:
            reasons.append("cumulative_area_change")
        if np.count_nonzero(m0 & m1) / max(1, np.count_nonzero(m0 | m1)) < tools.config.min_raster_iou:
            reasons.append("cumulative_raster_drift")
        if tools.edge_alignment(m1) < tools.edge_alignment(m0) - tools.config.max_edge_regression:
            reasons.append("cumulative_edge_regression")
    return sorted(set(reasons))


def search_simplification(tools, items, target_ids, profile="balanced", deadline=None, original=None):
    """Choose the fewest vertices among a bounded set of verified candidates.

    All constraints stay hard; target_vertices is an aspiration, not permission
    to remove holes, merge parcels or exceed the boundary/area/raster limits.
    """
    lookup = {p.identifier: p for p in items}
    targets = [lookup[i] for i in target_ids]
    geoms = np.asarray([normalized(p.geometry) for p in targets], dtype=object)
    joint = len(targets) > 1
    if joint:
        reference = shapely.union_all(shapely.boundary(geoms))
        geoms = np.asarray([shapely.snap(g, reference, 1e-8) for g in geoms], dtype=object)
        if not shapely.coverage_is_valid(geoms):
            raise ValueError("Joint simplification requires a nonoverlapping noded coverage")
    radius = tools.config.max_boundary_shift * PROFILES[profile]
    tolerances = np.unique(np.r_[0., np.linspace(radius / tools.config.simplify_search_steps,
                                                radius, tools.config.simplify_search_steps)])
    # Every potentially affected neighbor is included in screening. The caller
    # still verifies the chosen proposal against the complete scene before commit.
    region = unary_union(list(geoms)).envelope.buffer(tools.config.max_boundary_shift + 1e-6)
    tree = STRtree([p.geometry for p in items])
    near = set(int(i) for i in tree.query(region))
    local = [p for i, p in enumerate(items) if i in near or p.identifier in target_ids]
    options, seen = [], set()
    for tolerance in tolerances:
        if deadline is not None and time.monotonic() >= deadline:
            break
        alternatives = [("coverage_vw", shapely.coverage_simplify(geoms, float(tolerance), simplify_boundary=True))]
        if not joint:
            alternatives.append(("douglas_peucker", [geoms[0].simplify(float(tolerance), preserve_topology=True)]))
        for algorithm, result in alternatives:
            signature = tuple(g.wkb for g in result)
            if signature in seen:
                continue
            seen.add(signature)
            counts = [vertices(g) for g in result]
            # Total vertex count is the optimization objective. Drift breaks ties.
            options.append((sum(counts), float(tolerance), algorithm, result))
    options.sort(key=lambda row: (row[0], row[1], row[2]))
    rejected = Counter()
    checked = 0
    selected, selected_tolerance, selected_algorithm = items, None, None
    for count, tolerance, algorithm, result in options:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if count >= sum(vertices(p.geometry) for p in targets):
            continue
        replacements = {p.identifier: p.with_geometry(g) for p, g in zip(targets, result)}
        candidate_local = [replacements.get(p.identifier, p) for p in local]
        verdict = tools.verify(local, candidate_local, "simplify_contour", target_ids)
        candidate = [replacements.get(p.identifier, p) for p in items]
        extra = anchor_reasons(tools, original, candidate, target_ids) if original is not None else []
        checked += 1
        if verdict["accept"] and not extra:
            selected, selected_tolerance, selected_algorithm = candidate, tolerance, algorithm
            break
        rejected.update(extra + ([] if verdict["accept"] else verdict["reasons"]))
    final_lookup = {p.identifier: p for p in selected}
    counts = {uid: vertices(final_lookup[uid].geometry) for uid in target_ids}
    return selected, {"parameter_profile": profile, "algorithm": "bounded_candidate_search",
                      "joint_coverage": joint, "target_vertices": tools.config.target_vertices,
                      "tolerances": len(tolerances), "unique_candidates": len(options),
                      "checked_candidates": checked, "selected_tolerance": selected_tolerance,
                      "selected_algorithm": selected_algorithm,
                      "vertices_after": counts,
                      "target_met": {uid: n <= tools.config.target_vertices for uid, n in counts.items()},
                      "rejected_constraints": dict(rejected),
                      "optimality": "fewest among sampled candidates passing constraints"}


def compact_scene(tools, items, original, reviewed, deadline, observer=None, iteration_offset=0):
    """A bounded pass over all parcels, separate from the GPT/routing budget."""
    current, events = list(items), []
    groups = contact_groups(current)
    queue = [(ids, False) for ids in groups]
    status = "complete"
    for ids, fallback in queue:
        if time.monotonic() >= deadline:
            status = "time_budget"
            break
        ids = [uid for uid in ids if uid not in reviewed]
        if not ids:
            continue
        lookup = {p.identifier: p for p in current}
        if all(vertices(lookup[uid].geometry) <= max(4, structural_floor(lookup[uid].geometry)) for uid in ids):
            continue
        tick = time.monotonic()
        iteration = iteration_offset + len(events)
        try:
            candidate, evidence = search_simplification(tools, current, ids, "balanced", deadline, original)
            decision = tools.verify(current, candidate, "simplify_contour", ids)
            if observer:
                observer(iteration, current, candidate, "simplify_contour", ids, decision)
            events.append({"iteration": iteration, "phase": "vertex_compaction", "action": "simplify_contour",
                           "target_ids": ids, "accepted": decision["accept"], "rolled_back": not decision["accept"],
                           "reasons": decision["reasons"], "metrics": decision["metrics"], "evidence": evidence,
                           "elapsed_seconds": time.monotonic()-tick})
            if decision["accept"]:
                current = candidate
            elif len(ids) > 1 and not fallback:
                queue.extend(([uid], True) for uid in ids)
        except Exception as error:
            events.append({"iteration": iteration, "phase": "vertex_compaction", "action": "simplify_contour",
                           "target_ids": ids, "accepted": False, "rolled_back": True,
                           "reasons": ["joint_search_unavailable" if len(ids) > 1 else "tool_error"],
                           "error": f"{type(error).__name__}: {error}"})
            if len(ids) > 1 and not fallback:
                queue.extend(([uid], True) for uid in ids)
    if time.monotonic() >= deadline:
        status = "time_budget"
    return current, events, status


def vertex_report(before, after, target, events=(), reviewed=()):
    old = {p.identifier: p for p in before}
    records = []
    for p in after:
        n = vertices(p.geometry)
        floor = structural_floor(p.geometry)
        related = [e for e in events if p.identifier in e.get("target_ids", [])
                   and e.get("action") == "simplify_contour"]
        constraints = sorted({r for e in related for r in e.get("evidence", {}).get("rejected_constraints", {})})
        reason = ("target_met" if n <= target else "review_deferred" if p.identifier in reviewed else
                  "topology_requires_more_rings" if floor > target else
                  "bounded_search_limit" if related else "not_processed")
        records.append({"instance_id": p.identifier, "vertices_before": vertices(old[p.identifier].geometry)
                        if p.identifier in old else None, "vertices_after": n, "target_vertices": target,
                        "target_met": n <= target, "structural_minimum": floor, "status": reason,
                        "blocking_constraints": constraints})
    counts = [r["vertices_after"] for r in records]
    return {"target_vertices": target, "instances": len(records),
            "target_met": sum(r["target_met"] for r in records),
            "target_met_fraction": sum(r["target_met"] for r in records)/max(1, len(records)),
            "mean": float(np.mean(counts)) if counts else 0.,
            "median": float(np.median(counts)) if counts else 0.,
            "p90": float(np.percentile(counts, 90)) if counts else 0., "parcels": records}
