"""Resolve trustworthy, aligned tile grids for cross-tile tools.

Rotated, differently sampled or subpixel-shifted grids require reprojection and
are reported for review. They are never coerced into integer pixel adjacency.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path


class GeoNeighborFinder:
    def __init__(self, coordinate_mode="auto"):
        self.coordinate_mode = coordinate_mode
        self.issues = []
        self.references = {}
        self.group_metadata = {}

    def resolve(self, trace):
        from fuse_fieldbqc_tiles import TileMeta, parse_tile_meta

        metadata = trace.get("metadata", {})
        path = str(trace.get("image") or metadata.get("source", ""))
        stem = Path(path).stem
        transform, crs = metadata.get("transform"), metadata.get("crs")
        parsed = parse_tile_meta(path, self.coordinate_mode)

        def reject(reason):
            self.issues.append({"image": path, "reason": reason})
            return None

        # Filename coordinates refer to the original image orientation.
        tokens = stem.lower().split("_")
        if any(token in {"flip", "rot", "rotate", "vec", "gau", "add", "noise", "con", "bri",
                         "blur", "contrast", "brightness", "aug"} for token in tokens):
            return reject("augmentation_transform_not_verified")
        if len(tokens) >= 7 and all(token.isdigit() for token in tokens[-4:]) and tokens[-4:-2] != ["0", "0"]:
            return reject("augmentation_transform_not_verified")

        if transform is not None and crs:
            if len(transform) < 6 or not all(math.isfinite(float(value)) for value in transform[:6]):
                return reject("invalid_affine_transform")
            a, b, c, d, e, f = map(float, transform[:6])
            if a <= 0 or e >= 0 or abs(b) > abs(a) * 1e-8 or abs(d) > abs(e) * 1e-8:
                return reject("rotated_or_unsupported_grid_requires_reprojection")
            scene = metadata.get("scene_id") or metadata.get("base_scene")
            date = metadata.get("acquired_at") or metadata.get("acquisition_date")
            if not scene and parsed is not None:
                scene = parsed.group
            if not scene and not date:
                return reject("geotiff_acquisition_identity_missing")
            scene = str(scene or date)
            sensor = str(metadata.get("sensor", "unknown"))
            group = f"geo:{scene}:{sensor}:{date or 'unknown-time'}:{crs}:{a:.12g}:{e:.12g}"
            reference = self.references.setdefault(group, (c, f))
            x, y = (c - reference[0]) / a, (f - reference[1]) / e
            if abs(x - round(x)) > 0.01 or abs(y - round(y)) > 0.01:
                return reject("subpixel_grid_offset_requires_reprojection")
            width, height = int(metadata.get("width", 0)), int(metadata.get("height", 0))
            if width <= 0 or height <= 0:
                return reject("image_dimensions_missing")
            self.group_metadata[group] = {
                "crs": crs, "transform": [a, 0, reference[0], 0, e, reference[1]],
                "scene_id": scene, "coordinate_source": "geotiff",
            }
            return TileMeta(path, stem, group, round(x), round(y), width, height)
        if parsed is None or not stem.lower().startswith(("ddfhapd_", "ftw_")):
            return reject("trusted_tile_coordinates_missing")
        width, height = int(metadata.get("width", parsed.width)), int(metadata.get("height", parsed.height))
        if (width, height) != (parsed.width, parsed.height):
            return reject("filename_grid_dimensions_disagree")
        # Sensor/domain separation also prevents joining different sources that
        # happen to reuse a scene name and pixel coordinates.
        group = f"{parsed.group}:{metadata.get('sensor', 'unknown')}:{metadata.get('acquired_at', 'unknown-time')}"
        self.group_metadata[group] = {"coordinate_source": "filename", "scene_id": parsed.group}
        return replace(parsed, group=group)

    @staticmethod
    def neighbors(tiles, max_gap=2):
        """Return overlapping or edge-adjacent pairs using spatial buckets."""
        if not tiles:
            return []
        step = max(1, min(min(tile.width, tile.height) for tile in tiles))
        buckets, pairs = {}, set()
        for index, tile in enumerate(tiles):
            x0, x1 = (tile.origin_x - max_gap) // step, (tile.origin_x + tile.width + max_gap) // step
            y0, y1 = (tile.origin_y - max_gap) // step, (tile.origin_y + tile.height + max_gap) // step
            candidates = set()
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    candidates.update(buckets.get((tile.group, x, y), ()))
            for other_index in candidates:
                other = tiles[other_index]
                dx = min(tile.origin_x + tile.width, other.origin_x + other.width) - max(tile.origin_x, other.origin_x)
                dy = min(tile.origin_y + tile.height, other.origin_y + other.height) - max(tile.origin_y, other.origin_y)
                if (dx > 0 and dy >= -max_gap) or (dy > 0 and dx >= -max_gap):
                    pairs.add((other_index, index))
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    buckets.setdefault((tile.group, x, y), []).append(index)
        return sorted(pairs)
