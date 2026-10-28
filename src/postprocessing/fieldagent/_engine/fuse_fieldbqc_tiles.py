"""Fuse FieldBQC-Agent masks across adjacent image tiles.

The predictor intentionally keeps each 256x256 tile independent.  This
second-stage tool uses the coordinates encoded in FieldSeg filenames to join
instances that continue across a tile seam.  It reads the existing ``masks``
and ``traces`` directories, so no neural-network inference is repeated.

The output is one compressed NPZ and one JSON manifest per scene/grid group.
Each NPZ stores variable-sized fused masks under keys ``mask_000000`` etc.;
the manifest records global bounding boxes, quality summaries, source members,
and whether an unresolved outer-boundary review remains.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_INTEGER = re.compile(r"^-?\d+$")


@dataclass(frozen=True)
class TileMeta:
    image_path: str
    stem: str
    group: str
    origin_x: int
    origin_y: int
    width: int
    height: int


@dataclass
class TileInstance:
    tile: TileMeta
    index: int
    mask: np.ndarray
    score: float
    quality: float
    mask_quality: float
    boundary_quality: float


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _is_integer(value: str) -> bool:
    return bool(_INTEGER.fullmatch(value))


def parse_tile_meta(image_path: str, coordinate_mode: str = "auto") -> Optional[TileMeta]:
    """Parse the supported FieldSeg tile filename conventions.

    ``grid`` treats the two coordinates before the trailing augmentation and
    size fields as row/column indices. ``pixel`` treats them as x/y pixel
    origins. ``auto`` uses a numeric scene id as the grid convention, which
    matches DDFHAPD domestic tiles; scene names such as XJ1 use pixel origins.
    """
    path = Path(image_path)
    tokens = path.stem.split("_")
    if len(tokens) >= 4 and "-" in tokens[-1]:
        hyphen_coordinates = tokens[-1].split("-")
        if len(hyphen_coordinates) == 2 and all(_is_integer(token) for token in hyphen_coordinates):
            first, second = int(hyphen_coordinates[0]), int(hyphen_coordinates[1])
            group_tokens = tokens[:-1]
            if not group_tokens:
                return None
            if coordinate_mode == "grid":
                origin_x, origin_y = second * 256, first * 256
            else:
                # FTW Brazil/scene exports use pixel origins in the final
                # hyphen-separated coordinate pair.
                origin_x, origin_y = first, second
            return TileMeta(
                image_path=str(path),
                stem=path.stem,
                group="_".join(group_tokens),
                origin_x=origin_x,
                origin_y=origin_y,
                width=256,
                height=256,
            )
    if len(tokens) < 3 or not all(_is_integer(token) for token in tokens[-2:]):
        return None
    has_explicit_size = len(tokens) >= 7 and all(_is_integer(token) for token in tokens[-4:])
    if has_explicit_size:
        width, height = int(tokens[-2]), int(tokens[-1])
        prefix = tokens[:-4]  # coordinate pair/augmentation pair + width/height
    else:
        # FTW uses ``ftw_<country>_<scene>_<row>_<column>.png`` and all
        # released tiles are 256x256.  The mask array is checked again while
        # loading, so a future size change is rejected rather than misjoined.
        width = height = 256
        prefix = tokens[:-2]
    if width <= 0 or height <= 0:
        return None

    if has_explicit_size:
        while prefix and not _is_integer(prefix[-1]):
            prefix.pop()  # remove augmentation labels such as add, gau, vec
    if has_explicit_size:
        if len(prefix) < 3 or not (_is_integer(prefix[-1]) and _is_integer(prefix[-2])):
            return None
        first, second = int(prefix[-2]), int(prefix[-1])
        group_tokens = prefix[:-2]
    else:
        first, second = int(tokens[-2]), int(tokens[-1])
        group_tokens = tokens[:-2]
    if not group_tokens:
        return None
    group = "_".join(group_tokens)

    mode = coordinate_mode
    if mode == "auto":
        # Domestic files use names such as ddfhapd_CQ_0000000017_0_1; the
        # scene id is numeric. XJ/FBIS-style files use a scene token such as
        # XJ1 followed by pixel origins.
        mode = (
            "grid"
            if _is_integer(group_tokens[-1]) or group_tokens[0].lower() == "ftw"
            else "pixel"
        )
    if mode == "grid":
        origin_x, origin_y = second * width, first * height
    elif mode == "pixel":
        origin_x, origin_y = first, second
    else:
        raise ValueError(f"Unsupported coordinate mode: {coordinate_mode}")

    return TileMeta(
        image_path=str(path),
        stem=path.stem,
        group=group,
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height,
    )


def _array_value(values: np.ndarray, index: int, fallback: float) -> float:
    return float(values[index]) if index < len(values) else float(fallback)


def load_instances(input_root: Path, coordinate_mode: str, finder=None) -> Tuple[List[TileMeta], List[TileInstance], int]:
    trace_root = input_root / "traces"
    mask_root = input_root / "masks"
    if not trace_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f"Expected traces/ and masks/ under {input_root}")

    tiles: List[TileMeta] = []
    instances: List[TileInstance] = []
    skipped = 0
    for trace_path in sorted(trace_root.glob("*.json")):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        image_path = str(trace.get("image") or trace.get("metadata", {}).get("source", ""))
        tile = finder.resolve(trace) if finder is not None else parse_tile_meta(image_path, coordinate_mode=coordinate_mode)
        if tile is None:
            skipped += 1
            continue
        mask_path = mask_root / f"{trace_path.stem}.npz"
        if not mask_path.is_file():
            skipped += 1
            continue
        with np.load(mask_path) as packed:
            masks = np.asarray(packed["masks"], dtype=bool)
            scores = np.asarray(packed.get("scores", np.zeros(len(masks))), dtype=np.float32)
            qualities = np.asarray(packed.get("qualities", np.full(len(masks), np.nan)), dtype=np.float32)
            mask_qualities = np.asarray(packed.get("mask_qualities", qualities), dtype=np.float32)
            boundary_qualities = np.asarray(
                packed.get("boundary_qualities", qualities), dtype=np.float32
            )
            validity = np.asarray(packed.get("quality_valid", np.isfinite(qualities)), dtype=bool)
            qualities = np.where(validity, qualities, np.nan)
            mask_qualities = np.where(validity, mask_qualities, np.nan)
        if masks.ndim != 3:
            skipped += 1
            continue
        tiles.append(tile)
        for index, mask in enumerate(masks):
            if mask.shape != (tile.height, tile.width):
                skipped += 1
                continue
            quality = _array_value(qualities, index, _array_value(scores, index, 0.0))
            instances.append(
                TileInstance(
                    tile=tile,
                    index=index,
                    mask=mask,
                    score=_array_value(scores, index, quality),
                    quality=quality,
                    mask_quality=_array_value(mask_qualities, index, quality),
                    boundary_quality=_array_value(boundary_qualities, index, quality),
                )
            )
    return tiles, instances, skipped


def _touches_side(mask: np.ndarray, side: str, band: int) -> bool:
    band = max(1, min(int(band), min(mask.shape)))
    if side == "left":
        return bool(mask[:, :band].any())
    if side == "right":
        return bool(mask[:, -band:].any())
    if side == "top":
        return bool(mask[:band, :].any())
    if side == "bottom":
        return bool(mask[-band:, :].any())
    raise ValueError(side)


def _projection_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 0.0


def _overlap_iou(left: TileInstance, right: TileInstance) -> float:
    x0 = max(left.tile.origin_x, right.tile.origin_x)
    y0 = max(left.tile.origin_y, right.tile.origin_y)
    x1 = min(left.tile.origin_x + left.tile.width, right.tile.origin_x + right.tile.width)
    y1 = min(left.tile.origin_y + left.tile.height, right.tile.origin_y + right.tile.height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    left_crop = left.mask[
        y0 - left.tile.origin_y : y1 - left.tile.origin_y,
        x0 - left.tile.origin_x : x1 - left.tile.origin_x,
    ]
    right_crop = right.mask[
        y0 - right.tile.origin_y : y1 - right.tile.origin_y,
        x0 - right.tile.origin_x : x1 - right.tile.origin_x,
    ]
    intersection = np.count_nonzero(left_crop & right_crop)
    union = np.count_nonzero(left_crop | right_crop)
    return float(intersection / union) if union else 0.0


def _seam_score(left: TileInstance, right: TileInstance, band: int, max_gap: int) -> float:
    """Return seam continuity score for exactly adjacent tile rectangles."""
    a, b = left.tile, right.tile
    x_overlap0 = max(a.origin_x, b.origin_x)
    x_overlap1 = min(a.origin_x + a.width, b.origin_x + b.width)
    y_overlap0 = max(a.origin_y, b.origin_y)
    y_overlap1 = min(a.origin_y + a.height, b.origin_y + b.height)

    # left/right seam
    horizontal_gap = max(a.origin_x, b.origin_x) - min(a.origin_x + a.width, b.origin_x + b.width)
    if horizontal_gap <= max_gap and y_overlap1 > y_overlap0:
        if a.origin_x <= b.origin_x:
            first, second = left, right
            first_side, second_side = "right", "left"
        else:
            first, second = right, left
            first_side, second_side = "right", "left"
        if not (_touches_side(first.mask, first_side, band) and _touches_side(second.mask, second_side, band)):
            return 0.0
        y0, y1 = y_overlap0, y_overlap1
        first_projection = first.mask[y0 - first.tile.origin_y : y1 - first.tile.origin_y, -band:].any(axis=1)
        second_projection = second.mask[y0 - second.tile.origin_y : y1 - second.tile.origin_y, :band].any(axis=1)
        return _projection_iou(first_projection, second_projection)

    # top/bottom seam
    vertical_gap = max(a.origin_y, b.origin_y) - min(a.origin_y + a.height, b.origin_y + b.height)
    if vertical_gap <= max_gap and x_overlap1 > x_overlap0:
        if a.origin_y <= b.origin_y:
            first, second = left, right
            first_side, second_side = "bottom", "top"
        else:
            first, second = right, left
            first_side, second_side = "bottom", "top"
        if not (_touches_side(first.mask, first_side, band) and _touches_side(second.mask, second_side, band)):
            return 0.0
        x0, x1 = x_overlap0, x_overlap1
        first_projection = first.mask[-band:, x0 - first.tile.origin_x : x1 - first.tile.origin_x].any(axis=0)
        second_projection = second.mask[:band, x0 - second.tile.origin_x : x1 - second.tile.origin_x].any(axis=0)
        return _projection_iou(first_projection, second_projection)
    return 0.0


def _can_match(
    left: TileInstance,
    right: TileInstance,
    seam_band: int,
    seam_iou: float,
    overlap_iou: float,
    max_gap: int,
    min_quality: float,
    min_score: float = 0.0,
    min_mask_quality: float = 0.0,
) -> Tuple[bool, float, str]:
    if left.tile.group != right.tile.group:
        return False, 0.0, "different_group"
    a, b = left.tile, right.tile
    x_overlap = max(0, min(a.origin_x + a.width, b.origin_x + b.width) - max(a.origin_x, b.origin_x))
    y_overlap = max(0, min(a.origin_y + a.height, b.origin_y + b.height) - max(a.origin_y, b.origin_y))
    if x_overlap > 0 and y_overlap > 0:
        score = _overlap_iou(left, right)
        threshold = overlap_iou
        relation = "overlap"
    else:
        score = _seam_score(left, right, seam_band, max_gap)
        threshold = seam_iou
        relation = "seam"
    quality_ok = all(np.isfinite(value) and value >= min_quality for value in (left.quality, right.quality))
    score_ok = min(left.score, right.score) >= min_score
    mask_quality_ok = all(np.isfinite(value) and value >= min_mask_quality for value in (left.mask_quality, right.mask_quality))
    return (
        quality_ok and score_ok and mask_quality_ok and score >= threshold,
        score,
        relation,
    )


def _adjacent_sides(left: TileMeta, right: TileMeta, max_gap: int) -> Tuple[Tuple[str, str], ...]:
    """Return side pairs that have a neighboring tile relationship."""
    sides: List[Tuple[str, str]] = []
    x_overlap = min(left.origin_x + left.width, right.origin_x + right.width) - max(
        left.origin_x, right.origin_x
    )
    y_overlap = min(left.origin_y + left.height, right.origin_y + right.height) - max(
        left.origin_y, right.origin_y
    )
    horizontal_gap = max(left.origin_x, right.origin_x) - min(
        left.origin_x + left.width, right.origin_x + right.width
    )
    vertical_gap = max(left.origin_y, right.origin_y) - min(
        left.origin_y + left.height, right.origin_y + right.height
    )
    if y_overlap > 0 and horizontal_gap <= max_gap and left.origin_x != right.origin_x:
        sides.append(("right", "left") if left.origin_x < right.origin_x else ("left", "right"))
    if x_overlap > 0 and vertical_gap <= max_gap and left.origin_y != right.origin_y:
        sides.append(("bottom", "top") if left.origin_y < right.origin_y else ("top", "bottom"))
    return tuple(sides)


def _group_instances(instances: Sequence[TileInstance]) -> Dict[str, List[TileInstance]]:
    grouped: Dict[str, List[TileInstance]] = {}
    for instance in instances:
        grouped.setdefault(instance.tile.group, []).append(instance)
    return grouped


def _group_tiles(tiles: Sequence[TileMeta]) -> Dict[str, List[TileMeta]]:
    grouped: Dict[str, List[TileMeta]] = {}
    for tile in tiles:
        grouped.setdefault(tile.group, []).append(tile)
    return grouped


def fuse_group(
    group: str,
    tiles: Sequence[TileMeta],
    instances: Sequence[TileInstance],
    seam_band: int,
    seam_iou: float,
    overlap_iou: float,
    max_gap: int,
    min_quality: float,
    min_score: float = 0.0,
    min_mask_quality: float = 0.0,
    no_gate_fallback: bool = False,
    verified: bool = False,
    match_margin: float = 0.10,
    max_component_tiles: int = 16,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    unique_tiles = {tile.stem: tile for tile in tiles}
    nodes = list(instances)
    dsu = _DisjointSet(len(nodes))
    component_members = {index: {index} for index in range(len(nodes))}
    available_sides: Dict[int, set] = {index: set() for index in range(len(nodes))}
    matched_sides: Dict[int, set] = {index: set() for index in range(len(nodes))}
    match_log: List[Dict[str, object]] = []
    candidate_log: List[Dict[str, object]] = []
    candidates_by_tile_pair: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    candidate_instance_pairs = 0
    eligible_candidate_pairs = 0
    positive_score_pairs = 0

    tile_list = list(unique_tiles.values())
    from fieldbqc_geo import GeoNeighborFinder
    neighbor_pairs = GeoNeighborFinder.neighbors(tile_list, max_gap)
    neighbor_tile_pairs = len(neighbor_pairs)
    neighboring_stems, indices_by_tile = {}, {}
    for index, node in enumerate(nodes):
        indices_by_tile.setdefault(node.tile.stem, []).append(index)
    for left_tile_index, right_tile_index in neighbor_pairs:
        left_stem, right_stem = tile_list[left_tile_index].stem, tile_list[right_tile_index].stem
        neighboring_stems.setdefault(left_stem, set()).add(right_stem)
        neighboring_stems.setdefault(right_stem, set()).add(left_stem)

    # Score only instances in spatially neighboring tiles.
    for left_index in range(len(nodes)):
        for right_index in (index for stem in sorted(neighboring_stems.get(nodes[left_index].tile.stem, ()))
                            for index in indices_by_tile.get(stem, ()) if index > left_index):
            left, right = nodes[left_index], nodes[right_index]
            if left.tile.stem == right.tile.stem:
                continue
            side_pairs = _adjacent_sides(left.tile, right.tile, max_gap)
            x_overlap = min(
                left.tile.origin_x + left.tile.width, right.tile.origin_x + right.tile.width
            ) - max(left.tile.origin_x, right.tile.origin_x)
            y_overlap = min(
                left.tile.origin_y + left.tile.height, right.tile.origin_y + right.tile.height
            ) - max(left.tile.origin_y, right.tile.origin_y)
            if not ((x_overlap > 0 and y_overlap > 0) or side_pairs):
                continue
            for left_side, right_side in side_pairs:
                available_sides[left_index].add(left_side)
                available_sides[right_index].add(right_side)
            threshold_passed, score, relation = _can_match(
                left,
                right,
                seam_band,
                seam_iou,
                overlap_iou,
                max_gap,
                min_quality,
                min_score,
                min_mask_quality,
            )
            candidate_instance_pairs += 1
            quality_ok = all(np.isfinite(value) and value >= min_quality for value in (left.quality, right.quality))
            score_ok = min(left.score, right.score) >= min_score
            mask_quality_ok = all(np.isfinite(value) and value >= min_mask_quality for value in (left.mask_quality, right.mask_quality))
            endpoint_gate_ok = quality_ok and score_ok and mask_quality_ok
            if endpoint_gate_ok:
                eligible_candidate_pairs += 1
            if score > 0:
                positive_score_pairs += 1
            threshold = overlap_iou if relation == "overlap" else seam_iou
            geometry_ok = score >= threshold
            record: Dict[str, object] = {
                "left": {"image": left.tile.image_path, "index": left.index},
                "right": {"image": right.tile.image_path, "index": right.index},
                "relation": relation,
                "score": score,
                "threshold": threshold,
                "geometry_ok": geometry_ok,
                "quality_ok": quality_ok,
                "score_ok": score_ok,
                "mask_quality_ok": mask_quality_ok,
                "min_endpoint_score": min(left.score, right.score),
                "min_endpoint_quality": min(left.quality, right.quality),
                "min_endpoint_mask_quality": min(
                    left.mask_quality, right.mask_quality
                ),
                "threshold_passed": threshold_passed,
                "matched": False,
                "assignment_selected": False,
                "rejection_reason": None,
                "_left_node": left_index,
                "_right_node": right_index,
                "_side_pairs": side_pairs,
            }
            tile_pair = tuple(sorted((left.tile.stem, right.tile.stem)))
            candidates_by_tile_pair.setdefault(tile_pair, []).append(record)

    # Enforce one-to-one assignment independently for each adjacent tile pair.
    # The same parcel may still connect once to the left tile and once to the
    # right tile, which is required for parcels spanning three or more tiles.
    one_to_one_conflicts = 0
    threshold_passed_pairs = 0
    for pair_candidates in candidates_by_tile_pair.values():
        used_nodes = set()
        endpoint_candidates = {}
        for record in pair_candidates:
            for endpoint in (record["_left_node"], record["_right_node"]):
                endpoint_candidates.setdefault(endpoint, []).append(record)
        for values in endpoint_candidates.values():
            values.sort(key=lambda item: float(item["score"]), reverse=True)
        for record in sorted(pair_candidates, key=lambda item: float(item["score"]), reverse=True):
            left_index = int(record["_left_node"])
            right_index = int(record["_right_node"])
            geometry_ok = bool(record["geometry_ok"])
            endpoint_gate_ok = (
                bool(record["score_ok"])
                and bool(record["quality_ok"])
                and bool(record["mask_quality_ok"])
            )
            if bool(record["threshold_passed"]):
                threshold_passed_pairs += 1

            if verified:
                best_left, best_right = endpoint_candidates[left_index], endpoint_candidates[right_index]
                if best_left[0] is not record or best_right[0] is not record:
                    record["rejection_reason"] = "not_mutual_best"
                    continue
                margin_left = float(record["score"]) - (float(best_left[1]["score"]) if len(best_left) > 1 else 0.0)
                margin_right = float(record["score"]) - (float(best_right[1]["score"]) if len(best_right) > 1 else 0.0)
                record["match_margin"] = min(margin_left, margin_right)
                if record["match_margin"] < match_margin:
                    record["rejection_reason"] = "ambiguous_match"
                    continue
                if not all(np.isfinite(value) for value in (nodes[left_index].quality, nodes[right_index].quality,
                                                            nodes[left_index].mask_quality, nodes[right_index].mask_quality)):
                    record["rejection_reason"] = "quality_unavailable"
                    continue
                roots = {dsu.find(left_index), dsu.find(right_index)}
                members = set().union(*(component_members[root] for root in roots))
                member_tiles = [nodes[index].tile.stem for index in members]
                if len(member_tiles) != len(set(member_tiles)) or len(member_tiles) > max_component_tiles:
                    record["rejection_reason"] = "component_identity_conflict"
                    continue

            # In engineering-safe mode the geometric assignment is locked
            # before endpoint gates are applied. A rejected best pair still
            # consumes its two endpoints, so a weaker alternative cannot be
            # promoted automatically.
            if no_gate_fallback:
                if not geometry_ok:
                    record["rejection_reason"] = "below_threshold"
                    continue
                if left_index in used_nodes or right_index in used_nodes:
                    record["rejection_reason"] = "one_to_one_conflict"
                    one_to_one_conflicts += 1
                    continue
                used_nodes.update((left_index, right_index))
                record["assignment_selected"] = True
                if not endpoint_gate_ok:
                    if not bool(record["score_ok"]):
                        record["rejection_reason"] = "low_detection_score"
                    elif not bool(record["quality_ok"]):
                        record["rejection_reason"] = "low_quality"
                    elif not bool(record["mask_quality_ok"]):
                        record["rejection_reason"] = "low_mask_quality"
                    continue
            else:
                if not bool(record["threshold_passed"]):
                    if not geometry_ok:
                        record["rejection_reason"] = "below_threshold"
                    elif not bool(record["score_ok"]):
                        record["rejection_reason"] = "low_detection_score"
                    elif not bool(record["quality_ok"]):
                        record["rejection_reason"] = "low_quality"
                    elif not bool(record["mask_quality_ok"]):
                        record["rejection_reason"] = "low_mask_quality"
                    else:
                        raise RuntimeError("Candidate rejection has no failed gate")
                    continue
                if left_index in used_nodes or right_index in used_nodes:
                    record["rejection_reason"] = "one_to_one_conflict"
                    one_to_one_conflicts += 1
                    continue
                used_nodes.update((left_index, right_index))
                record["assignment_selected"] = True
            record["matched"] = True
            previous_roots = {dsu.find(left_index), dsu.find(right_index)}
            merged_members = set().union(*(component_members[root] for root in previous_roots))
            dsu.union(left_index, right_index)
            for root in previous_roots:
                component_members.pop(root)
            component_members[dsu.find(left_index)] = merged_members
            for left_side, right_side in record["_side_pairs"]:
                matched_sides[left_index].add(left_side)
                matched_sides[right_index].add(right_side)
            match_log.append(
                {
                    "left": record["left"],
                    "right": record["right"],
                    "relation": record["relation"],
                    "score": record["score"],
                }
            )

    for pair_candidates in candidates_by_tile_pair.values():
        for record in pair_candidates:
            candidate_log.append(
                {key: value for key, value in record.items() if not key.startswith("_")}
            )

    components: Dict[int, List[int]] = {}
    for index in range(len(nodes)):
        components.setdefault(dsu.find(index), []).append(index)

    arrays: Dict[str, np.ndarray] = {}
    manifest_instances: List[Dict[str, object]] = []
    for output_index, member_indices in enumerate(components.values()):
        members = [nodes[index] for index in member_indices]
        x0 = min(item.tile.origin_x for item in members)
        y0 = min(item.tile.origin_y for item in members)
        x1 = max(item.tile.origin_x + item.tile.width for item in members)
        y1 = max(item.tile.origin_y + item.tile.height for item in members)
        fused = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        for item in members:
            iy0 = item.tile.origin_y - y0
            ix0 = item.tile.origin_x - x0
            iy1 = iy0 + item.tile.height
            ix1 = ix0 + item.tile.width
            fused[iy0:iy1, ix0:ix1] |= item.mask.astype(np.uint8)

        boundary_pending = any(
            _touches_side(nodes[index].mask, side, seam_band)
            and (side not in available_sides[index] or side not in matched_sides[index])
            for index in member_indices
            for side in ("left", "right", "top", "bottom")
        )
        mask_key = f"mask_{output_index:06d}"
        arrays[mask_key] = fused
        manifest_instances.append(
            {
                "instance_id": output_index,
                "mask_key": mask_key,
                "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
                "area": int(fused.sum()),
                "score_mean": float(np.mean([item.score for item in members])),
                "quality_mean": float(np.mean([item.quality for item in members])),
                "quality_min": float(np.min([item.quality for item in members])),
                "mask_quality_mean": float(np.mean([item.mask_quality for item in members])),
                "boundary_quality_mean": float(
                    np.mean([item.boundary_quality for item in members])
                ),
                "member_count": len(members),
                "boundary_pending": boundary_pending,
                "members": [
                    {"image": item.tile.image_path, "index": item.index} for item in members
                ],
            }
        )

    manifest = {
        "group": group,
        "tile_count": len(unique_tiles),
        "tiles_with_instances": len({item.tile.stem for item in nodes}),
        "empty_tile_count": len(unique_tiles) - len({item.tile.stem for item in nodes}),
        "neighbor_tile_pair_count": neighbor_tile_pairs,
        "input_instance_count": len(nodes),
        "fused_instance_count": len(manifest_instances),
        "candidate_instance_pair_count": candidate_instance_pairs,
        "eligible_candidate_pair_count": eligible_candidate_pairs,
        "positive_score_pair_count": positive_score_pairs,
        "threshold_passed_pair_count": threshold_passed_pairs,
        "one_to_one_conflict_count": one_to_one_conflicts,
        "gate_rejection_counts": {
            reason: sum(item.get("rejection_reason") == reason for item in candidate_log)
            for reason in (
                "low_detection_score",
                "low_quality",
                "low_mask_quality",
                "below_threshold",
                "one_to_one_conflict",
                "not_mutual_best",
                "ambiguous_match",
                "quality_unavailable",
                "component_identity_conflict",
            )
        },
        "gate_parameters": {
            "seam_iou": seam_iou,
            "overlap_iou": overlap_iou,
            "min_quality": min_quality,
            "min_score": min_score,
            "min_mask_quality": min_mask_quality,
            "no_gate_fallback": no_gate_fallback,
            "verified": verified,
            "match_margin": match_margin,
        },
        "max_candidate_score": max((item["score"] for item in candidate_log), default=0.0),
        "matched_pair_count": len(match_log),
        "boundary_pending_count": sum(
            bool(item["boundary_pending"]) for item in manifest_instances
        ),
        "tiles": [
            {
                "image": tile.image_path,
                "origin_xy": [tile.origin_x, tile.origin_y],
                "size_wh": [tile.width, tile.height],
                "instances": sum(item.tile.stem == tile.stem for item in nodes),
            }
            for tile in sorted(unique_tiles.values(), key=lambda item: (item.origin_y, item.origin_x))
        ],
        "top_candidates": sorted(
            candidate_log, key=lambda item: item["score"], reverse=True
        )[:20],
        "rejected_links": [item for item in candidate_log if not item["matched"] and item["geometry_ok"]],
        "matches": match_log,
        "instances": manifest_instances,
    }
    return arrays, manifest


def _safe_group_name(group: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", group).strip("_") or "group"


def _finite_json(value):
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> Dict[str, object]:
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    verified = getattr(args, "verified", False)
    finder = None
    if verified:
        from fieldbqc_geo import GeoNeighborFinder
        finder = GeoNeighborFinder(args.coordinate_mode)
    tiles, instances, skipped = load_instances(input_root, args.coordinate_mode, finder)
    tile_groups = _group_tiles(tiles)
    instance_groups = _group_instances(instances)
    group_summaries = []
    for group, group_tiles in sorted(tile_groups.items()):
        group_instances = instance_groups.get(group, [])
        policy = {}
        if verified:
            from fieldbqc_policy import DomainRouter
            policies = []
            for tile in group_tiles:
                trace = json.loads((input_root / "traces" / f"{tile.stem}.json").read_text(encoding="utf-8"))
                policies.append(trace.get("domain_policy") or DomainRouter().route(trace.get("metadata", {})).to_dict())
            # A group that spans policy variants uses its strictest link gates.
            for key, fallback in (("fusion_seam_iou", args.seam_iou), ("fusion_overlap_iou", args.overlap_iou),
                                  ("fusion_match_margin", getattr(args, "match_margin", 0.10))):
                policy[key] = max(float(item.get(key, fallback)) for item in policies)
        arrays, manifest = fuse_group(
            group,
            group_tiles,
            group_instances,
            seam_band=args.seam_band,
            seam_iou=policy.get("fusion_seam_iou", args.seam_iou),
            overlap_iou=policy.get("fusion_overlap_iou", args.overlap_iou),
            max_gap=args.max_gap,
            min_quality=args.min_quality,
            min_score=args.min_score,
            min_mask_quality=args.min_mask_quality,
            no_gate_fallback=args.no_gate_fallback or verified,
            verified=verified,
            match_margin=policy.get("fusion_match_margin", getattr(args, "match_margin", 0.10)),
        )
        if finder is not None:
            manifest["georeferencing"] = finder.group_metadata.get(group, {})
        if verified:
            manifest["quality_semantics"] = "Endpoint MQC is evidence only; fused masks have no re-estimated MQC."
            for item in manifest["instances"]:
                item["quality_valid"] = item["member_count"] == 1 and np.isfinite(item["quality_mean"])
                item["needs_review"] = bool(item["boundary_pending"] or not item["quality_valid"])
        manifest = _finite_json(manifest)
        name = _safe_group_name(group)
        np.savez_compressed(output_root / f"{name}.npz", **arrays)
        (output_root / f"{name}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        group_summaries.append(
            {
                "group": group,
                "tiles": manifest["tile_count"],
                "tiles_with_instances": manifest["tiles_with_instances"],
                "empty_tiles": manifest["empty_tile_count"],
                "neighbor_tile_pairs": manifest["neighbor_tile_pair_count"],
                "input_instances": manifest["input_instance_count"],
                "fused_instances": manifest["fused_instance_count"],
                "candidate_instance_pairs": manifest["candidate_instance_pair_count"],
                "eligible_candidate_pairs": manifest["eligible_candidate_pair_count"],
                "positive_score_pairs": manifest["positive_score_pair_count"],
                "threshold_passed_pairs": manifest["threshold_passed_pair_count"],
                "one_to_one_conflicts": manifest["one_to_one_conflict_count"],
                "gate_rejections": manifest["gate_rejection_counts"],
                "max_candidate_score": manifest["max_candidate_score"],
                "matched_pairs": manifest["matched_pair_count"],
                "boundary_pending": manifest["boundary_pending_count"],
            }
        )
    summary = {
        "input": str(input_root),
        "output": str(output_root),
        "coordinate_mode": args.coordinate_mode,
        "parameters": {
            "seam_band": args.seam_band,
            "seam_iou": args.seam_iou,
            "overlap_iou": args.overlap_iou,
            "max_gap": args.max_gap,
            "min_quality": args.min_quality,
            "min_score": args.min_score,
            "min_mask_quality": args.min_mask_quality,
            "no_gate_fallback": args.no_gate_fallback or verified,
        },
        "tile_count": len(tiles),
        "instance_count": len(instances),
        "skipped": skipped,
        "verified": verified,
        "review": finder.issues if finder is not None else [],
        "group_count": len(tile_groups),
        "groups": group_summaries,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="FieldBQC-Agent output directory")
    parser.add_argument("--output", default=None, help="Directory for fused groups")
    parser.add_argument(
        "--coordinate-mode",
        choices=("auto", "grid", "pixel"),
        default="auto",
        help="Interpret the two filename coordinates as grid indices or pixel origins.",
    )
    parser.add_argument("--seam-band", type=int, default=8)
    parser.add_argument("--seam-iou", type=float, default=0.35)
    parser.add_argument("--overlap-iou", type=float, default=0.08)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-quality", type=float, default=0.45)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum detection confidence required at both link endpoints.",
    )
    parser.add_argument(
        "--min-mask-quality",
        type=float,
        default=0.0,
        help="Minimum MQC mask-quality score required at both link endpoints.",
    )
    parser.add_argument(
        "--no-gate-fallback",
        action="store_true",
        help=(
            "Lock one-to-one geometric assignments before quality gates; "
            "do not promote weaker candidates when the best pair fails a gate."
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verified", action="store_true", help="Use GeoTIFF metadata, mutual-best matching and component verification")
    parser.add_argument("--match-margin", type=float, default=0.10)
    return parser.parse_args()


def run_self_test() -> None:
    cq = parse_tile_meta("ddfhapd_CQ_0000000017_0_1_0_0_256_256.png")
    assert cq is not None and cq.group == "ddfhapd_CQ_0000000017"
    assert (cq.origin_x, cq.origin_y) == (256, 0)
    xj = parse_tile_meta("ddfhapd_XJ_XJ1_000383_005887_0_0_256_256.png")
    assert xj is not None and xj.group == "ddfhapd_XJ_XJ1"
    assert (xj.origin_x, xj.origin_y) == (383, 5887)

    left_tile = TileMeta("left.png", "left", "synthetic", 0, 0, 16, 16)
    right_tile = TileMeta("right.png", "right", "synthetic", 16, 0, 16, 16)
    left_mask = np.zeros((16, 16), dtype=bool)
    right_mask = np.zeros((16, 16), dtype=bool)
    left_mask[4:12, 12:16] = True
    right_mask[4:12, :4] = True
    left = TileInstance(left_tile, 0, left_mask, 0.8, 0.8, 0.8, 0.8)
    right = TileInstance(right_tile, 0, right_mask, 0.8, 0.8, 0.8, 0.8)
    arrays, manifest = fuse_group(
        "synthetic", [left_tile, right_tile], [left, right], 4, 0.35, 0.08, 2, 0.45
    )
    assert manifest["matched_pair_count"] == 1
    assert manifest["fused_instance_count"] == 1
    assert manifest["tile_count"] == 2
    assert manifest["neighbor_tile_pair_count"] == 1
    assert arrays["mask_000000"].sum() == 64

    _, score_gated_manifest = fuse_group(
        "synthetic",
        [left_tile, right_tile],
        [left, right],
        4,
        0.35,
        0.08,
        2,
        0.45,
        min_score=0.85,
    )
    assert score_gated_manifest["matched_pair_count"] == 0
    assert score_gated_manifest["gate_rejection_counts"]["low_detection_score"] == 1

    _, mask_quality_gated_manifest = fuse_group(
        "synthetic",
        [left_tile, right_tile],
        [left, right],
        4,
        0.35,
        0.08,
        2,
        0.45,
        min_mask_quality=0.85,
    )
    assert mask_quality_gated_manifest["matched_pair_count"] == 0
    assert mask_quality_gated_manifest["gate_rejection_counts"]["low_mask_quality"] == 1

    low_score_right = TileInstance(
        right_tile, 0, right_mask, 0.4, 0.8, 0.8, 0.8
    )
    alternative_mask = np.zeros((16, 16), dtype=bool)
    alternative_mask[4:8, :4] = True
    alternative_right = TileInstance(
        right_tile, 1, alternative_mask, 0.8, 0.8, 0.8, 0.8
    )
    _, fallback_manifest = fuse_group(
        "synthetic",
        [left_tile, right_tile],
        [left, low_score_right, alternative_right],
        4,
        0.35,
        0.08,
        2,
        0.45,
        min_score=0.5,
    )
    assert fallback_manifest["matched_pair_count"] == 1
    _, locked_manifest = fuse_group(
        "synthetic",
        [left_tile, right_tile],
        [left, low_score_right, alternative_right],
        4,
        0.35,
        0.08,
        2,
        0.45,
        min_score=0.5,
        no_gate_fallback=True,
    )
    assert locked_manifest["matched_pair_count"] == 0
    assert locked_manifest["gate_rejection_counts"]["low_detection_score"] == 1

    duplicate_right = TileInstance(right_tile, 1, right_mask, 0.7, 0.7, 0.7, 0.7)
    _, conflict_manifest = fuse_group(
        "synthetic",
        [left_tile, right_tile],
        [left, right, duplicate_right],
        4,
        0.35,
        0.08,
        2,
        0.45,
    )
    assert conflict_manifest["threshold_passed_pair_count"] == 2
    assert conflict_manifest["matched_pair_count"] == 1
    assert conflict_manifest["one_to_one_conflict_count"] == 1
    assert conflict_manifest["fused_instance_count"] == 2

    third_tile = TileMeta("third.png", "third", "synthetic", 32, 0, 16, 16)
    third_mask = np.zeros((16, 16), dtype=bool)
    middle_mask = np.zeros((16, 16), dtype=bool)
    middle_mask[4:12, :] = True
    third_mask[4:12, :4] = True
    middle = TileInstance(right_tile, 0, middle_mask, 0.8, 0.8, 0.8, 0.8)
    third = TileInstance(third_tile, 0, third_mask, 0.8, 0.8, 0.8, 0.8)
    _, chain_manifest = fuse_group(
        "synthetic",
        [left_tile, right_tile, third_tile],
        [left, middle, third],
        4,
        0.35,
        0.08,
        2,
        0.45,
    )
    assert chain_manifest["matched_pair_count"] == 2
    assert chain_manifest["one_to_one_conflict_count"] == 0
    assert chain_manifest["fused_instance_count"] == 1
    print("FieldBQC tile-fusion self-test passed.")


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.self_test:
        run_self_test()
    else:
        if not cli_args.input or not cli_args.output:
            raise SystemExit("--input and --output are required unless --self-test is used")
        summary = run(cli_args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
