"""Explicit scene mosaics and conservative reciprocal cross-tile association.

Only aligned images in a common pixel grid are accepted. Names are not parsed
for coordinates. Non-overlapping seam association remains for manual review.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from shapely.affinity import translate
from shapely.geometry import box

from fieldbqc_geometry import GeometryInstance, from_parcels, normalized
from run_agent_experiments import load_parcels


def load_tile_scene(manifest_path, max_pixels=16_000_000):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    tiles = manifest["tiles"]
    if not tiles or not manifest.get("scene_id") or manifest.get("coordinate_units") != "pixels":
        raise ValueError("Tile manifest needs scene_id, coordinate_units=pixels and tiles")
    if manifest.get("aligned_common_grid") is not True:
        raise ValueError("Explicit aligned_common_grid=true is required")
    threshold = float(manifest.get("match_iou", 0.7))
    margin = float(manifest.get("match_margin", 0.1))
    if not 0.5 <= threshold <= 1 or not 0 < margin <= 1:
        raise ValueError("Invalid reciprocal matching thresholds")
    records, seen = [], set()
    for tile in tiles:
        identifier = str(tile["id"])
        if not identifier or identifier in seen:
            raise ValueError("Tile IDs must be unique")
        seen.add(identifier)
        origin = tile["origin"]
        if len(origin) != 2 or any(type(v) is not int for v in origin):
            raise ValueError("Tile origin must be integer x,y in a common pixel grid")
        def resolve(value):
            p = Path(value)
            return p if p.is_absolute() else path.parent / p
        image = cv2.imread(str(resolve(tile["image"])), cv2.IMREAD_COLOR)
        parcels, shape = load_parcels(resolve(tile["masks"]))
        if image is None or image.shape[:2] != shape:
            raise ValueError("Tile image/mask mismatch")
        records.append((identifier, origin, image, parcels))
    x0 = min(o[0] for _, o, _, _ in records)
    y0 = min(o[1] for _, o, _, _ in records)
    x1 = max(o[0] + im.shape[1] for _, o, im, _ in records)
    y1 = max(o[1] + im.shape[0] for _, o, im, _ in records)
    if (x1 - x0) * (y1 - y0) > max_pixels:
        raise ValueError("Scene exceeds mosaic pixel limit; process smaller scene windows")
    image_sum = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    image_count = np.zeros((y1 - y0, x1 - x0), np.uint16)
    instances, footprints, sources = [], {}, {}
    for identifier, origin, image, parcels in records:
        x, y = origin[0] - x0, origin[1] - y0
        h, w = image.shape[:2]
        image_sum[y:y+h, x:x+w] += image
        image_count[y:y+h, x:x+w] += 1
        footprint = box(x, y, x+w, y+h)
        for p in from_parcels(parcels):
            uid = f"{identifier}:{p.identifier}"
            instances.append(GeometryInstance(uid, translate(p.geometry, xoff=x, yoff=y),
                                               p.score, p.cls, (uid,)))
            footprints[uid] = footprint
            sources[uid] = identifier
    mosaic = np.rint(image_sum / np.maximum(image_count[..., None], 1)).astype(np.uint8)
    choices = {p.identifier: [] for p in instances}
    for i, p in enumerate(instances):
        for q in instances[i+1:]:
            if sources[p.identifier] == sources[q.identifier] or p.cls != q.cls:
                continue
            common = footprints[p.identifier].intersection(footprints[q.identifier])
            if common.area <= 0:
                continue
            a, b = p.geometry.intersection(common), q.geometry.intersection(common)
            union = a.union(b).area
            score = a.intersection(b).area / union if union else 0
            # At least four common foreground pixels; tiny contacts are ambiguous.
            if score >= threshold and min(a.area, b.area) >= 4:
                choices[p.identifier].append((score, q.identifier))
                choices[q.identifier].append((score, p.identifier))
    best = {}
    for uid, entries in choices.items():
        entries.sort(reverse=True)
        if entries and (len(entries) == 1 or entries[0][0] - entries[1][0] >= margin):
            best[uid] = entries[0][1]
    pairs = sorted({tuple(sorted((a, b))) for a, b in best.items() if best.get(b) == a})
    metadata = {"scene_id": manifest["scene_id"], "origin": [x0, y0],
                "coordinate_units": "pixels", "aligned_common_grid": True,
                "tile_count": len(records), "certified_pairs": pairs,
                "match_iou": threshold, "match_margin": margin,
                "limitation": "Only reciprocal matches in overlapping tiles are associated"}
    return mosaic, instances, pairs, metadata
