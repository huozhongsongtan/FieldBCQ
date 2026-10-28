"""Instance-scoped tools, domain routing and transactional verification for FieldBQC.

The model supplies observations; tools only propose changes. No labels or test
metrics are available to the policy. Defaults need development-set calibration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from fieldbqc_policy import DomainPolicy, DomainRouter
from fieldbqc_recipes import RepairRecipeLibrary

try:
    import cv2
except ImportError:
    cv2 = None

Array = np.ndarray
RerunTool = Callable[[Array, Sequence["Parcel"], str, Mapping[str, Any]], Sequence["Parcel"]]
CropPredictor = Callable[[Array, int, float], Sequence["Parcel"]]


@dataclass
class Parcel:
    mask: Array
    score: float
    cls: int = 0
    quality: Optional[float] = None
    mask_quality: Optional[float] = None
    boundary_quality: Optional[float] = None
    parcel_id: Optional[str] = None
    quality_valid: bool = True
    quality_source: str = "model"

    def copy(self) -> "Parcel":
        return Parcel(np.asarray(self.mask, dtype=bool).copy(), float(self.score), int(self.cls),
                      self.quality, self.mask_quality, self.boundary_quality, self.parcel_id,
                      self.quality_valid, self.quality_source)

    def invalidate_quality(self):
        self.quality = self.mask_quality = self.boundary_quality = None
        self.quality_valid = False
        self.quality_source = "geometry_modified"


@dataclass(frozen=True)
class AgentConfig:
    accept_quality: float = 0.72
    review_quality: float = 0.45
    min_edge_alignment: float = 0.18
    max_tta_disagreement: float = 0.12
    border_margin: int = 2
    min_area: int = 6
    min_action_gain: float = 0.005
    max_iterations: int = 2
    min_peak_distance: int = 12
    min_peak_value: float = 4.0
    split_merge_risk: float = 0.5
    enable_boundary_tool: bool = True
    enable_split_tool: bool = True
    enable_merge_tool: bool = True
    enable_simplify_tool: bool = True
    max_contour_vertices: int = 80
    max_split_parts: int = 6
    max_local_calls: int = 2
    max_seconds: float = 30.0
    min_reinfer_iou: float = 0.50
    reinfer_match_margin: float = 0.10
    max_edge_regression: float = 0.03

    def __post_init__(self):
        for name in ("max_iterations", "max_local_calls"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("max_contour_vertices", "max_split_parts"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2")
        for name in ("accept_quality", "review_quality", "min_edge_alignment", "min_action_gain",
                     "split_merge_risk",
                     "min_reinfer_iou", "reinfer_match_margin", "max_edge_regression"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"Invalid {name}")
        if not np.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")


@dataclass
class ParcelObservation:
    index: int
    quality: Optional[float]
    edge_alignment: float
    border_risk: float
    merge_risk: float
    fragment_risk: float
    topology_valid: bool
    issue_type: str
    severity: float
    reasons: List[str] = field(default_factory=list)
    issue_types: List[str] = field(default_factory=list)
    parcel_id: str = ""
    area_fraction: float = 0.0
    component_count: int = 1
    contour_vertices: int = 0
    merge_partner: Optional[str] = None
    distance_peak_count: int = 0


@dataclass
class AgentState:
    mean_quality: Optional[float]
    mean_edge_alignment: float
    tta_disagreement: float
    high_risk_count: int
    topology_valid: bool
    observations: List[ParcelObservation]


@dataclass
class AgentResult:
    parcels: List[Parcel]
    accepted: bool
    needs_review: bool
    action_trace: List[Dict[str, Any]]
    final_state: AgentState
    review_ids: List[str] = field(default_factory=list)
    domain_policy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolProposal:
    action: str
    target_ids: List[str]
    parcels: List[Parcel]
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationDecision:
    accept: bool
    reasons: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)


def _as_bool_mask(mask: Array, shape: Optional[Tuple[int, int]] = None) -> Array:
    value = np.asarray(mask)
    if value.ndim != 2 or (shape is not None and value.shape != shape):
        raise ValueError(f"Invalid mask shape {value.shape}, expected {shape or 'HxW'}")
    return value.astype(bool)


def _bbox(mask: Array) -> Tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0, 0, 0, 0
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _boundary(mask: Array) -> Array:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    eroded = np.ones(mask.shape, dtype=bool)
    for row in range(3):
        for col in range(3):
            eroded &= padded[row:row + mask.shape[0], col:col + mask.shape[1]]
    return np.asarray(mask, dtype=bool) & ~eroded


def _boundary_band(mask: Array, width: int) -> Array:
    if cv2 is None:
        raise RuntimeError("BoundaryRefiner requires OpenCV")
    kernel = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    source = mask.astype(np.uint8)
    outer = cv2.dilate(source, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    inner = cv2.erode(source, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return outer.astype(bool) & ~inner.astype(bool)


def _edge_map(image: Array) -> Array:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if cv2 is not None else image.mean(axis=2)
    else:
        gray = image
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    if cv2 is not None:
        return cv2.dilate((cv2.Canny(gray, 50, 150) > 0).astype(np.uint8),
                          np.ones((3, 3), np.uint8)).astype(bool)
    dx = np.abs(np.diff(gray.astype(float), axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray.astype(float), axis=0, prepend=gray[:1]))
    return dx + dy > 12


def _edge_alignment(image: Array, mask: Array, edges: Array | None = None) -> float:
    border = _boundary(mask)
    return float(np.count_nonzero(border & (edges if edges is not None else _edge_map(image))) /
                 max(1, np.count_nonzero(border)))


def _quality(parcel: Parcel) -> Optional[float]:
    value = parcel.quality
    if not parcel.quality_valid or value is None or not np.isfinite(value) or not 0 <= value <= 1:
        return None
    return float(value)


def _peak_count(mask: Array, min_distance: int, min_value: float) -> int:
    return len(_distance_peaks(mask, min_distance, min_value)[1])


def _distance_peaks(mask: Array, min_distance: int, min_value: float) -> Tuple[Array, List[Tuple[int, int]]]:
    """Return a distance transform and separated local-maximum seed points."""
    if cv2 is None or not mask.any():
        return np.zeros(mask.shape, dtype=np.float32), []
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    kernel = np.ones((2 * max(1, int(min_distance)) + 1,) * 2, np.uint8)
    peaks = (distance >= cv2.dilate(distance, kernel) - 1e-6) & (distance >= min_value)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        peaks.astype(np.uint8), connectivity=8
    )
    points = []
    for label in range(1, count):
        x, y = centroids[label]
        points.append((int(round(y)), int(round(x))))
    return distance, points


def _component_count(mask: Array) -> int:
    if cv2 is None:
        return int(bool(np.asarray(mask).any()))
    return max(0, int(cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)[0] - 1))


def _contour_vertices(mask: Array) -> int:
    if cv2 is None or not np.asarray(mask).any():
        return 0
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return int(sum(len(contour) for contour in contours))


def _topology_valid(parcels: Sequence[Parcel], shape: Tuple[int, int]) -> bool:
    occupied = np.zeros(shape, dtype=bool)
    for parcel in parcels:
        mask = _as_bool_mask(parcel.mask, shape)
        if not mask.any() or np.count_nonzero(occupied & mask) > max(2, int(mask.sum() * 0.01)):
            return False
        occupied |= mask
    return True


def _mask_iou(left: Array, right: Array) -> float:
    return float(np.count_nonzero(left & right) / max(1, np.count_nonzero(left | right)))


class BoundaryRefiner:
    """Propose morphology candidates in a narrow band, selected using image edges."""

    def propose(self, image: Array, parcels: Sequence[Parcel], target_id: str,
                policy: DomainPolicy) -> ToolProposal:
        if cv2 is None:
            raise RuntimeError("BoundaryRefiner requires OpenCV")
        result = [parcel.copy() for parcel in parcels]
        target = next(parcel for parcel in result if parcel.parcel_id == target_id)
        original = target.mask.copy()
        band = _boundary_band(original, policy.boundary_band)
        edges = _edge_map(image)
        best, best_alignment = original, _edge_alignment(image, original, edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for operation in (cv2.MORPH_CLOSE, cv2.MORPH_OPEN):
            changed = cv2.morphologyEx(original.astype(np.uint8), operation, kernel).astype(bool)
            changed = np.where(band, changed, original)
            alignment = _edge_alignment(image, changed, edges)
            if changed.any() and alignment > best_alignment:
                best, best_alignment = changed, alignment
        target.mask = best.copy()
        if not np.array_equal(best, original):
            target.invalidate_quality()
        return ToolProposal("boundary_cleanup", [target_id], result,
                            {"band_width": policy.boundary_band, "edge_alignment": best_alignment})


class AdhesionSplitter:
    """Split one merged mask using distance-transform seeds and watershed."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def propose(self, image: Array, parcels: Sequence[Parcel], target_id: str,
                policy: DomainPolicy) -> ToolProposal:
        if cv2 is None:
            raise RuntimeError("AdhesionSplitter requires OpenCV")
        result = [parcel.copy() for parcel in parcels]
        target = next(parcel for parcel in result if parcel.parcel_id == target_id)
        mask = _as_bool_mask(target.mask, tuple(image.shape[:2]))
        distance, points = _distance_peaks(mask, self.config.min_peak_distance, self.config.min_peak_value)
        if len(points) < 2:
            raise ValueError("Insufficient interior seeds for adhesion split")
        if len(points) > self.config.max_split_parts:
            points = sorted(points, key=lambda point: distance[point], reverse=True)[:self.config.max_split_parts]

        markers = np.zeros(mask.shape, dtype=np.int32)
        for label, (row, col) in enumerate(points, start=1):
            markers[row, col] = label
        # Watershed on the inverse distance surface keeps the cut in the
        # narrowest part of the connected prediction instead of expanding it.
        max_distance = max(float(distance.max()), 1.0)
        surface = np.clip((1.0 - distance / max_distance) * 255.0, 0, 255).astype(np.uint8)
        surface = cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)
        markers[~mask] = -1
        cv2.watershed(surface, markers)

        parts = []
        for label in range(1, len(points) + 1):
            part = (markers == label) & mask
            if int(part.sum()) >= policy.split_min_area:
                parts.append(part)
        # Watershed marks the separating ridge as -1. Assign those pixels to
        # the nearest surviving part so the split preserves the source mask.
        if parts:
            union = np.logical_or.reduce(parts)
            missing = mask & ~union
            if missing.any():
                distances = np.stack([
                    cv2.distanceTransform((~part).astype(np.uint8), cv2.DIST_L2, 5)
                    for part in parts
                ])
                nearest = np.argmin(distances, axis=0)
                parts = [part | (missing & (nearest == index)) for index, part in enumerate(parts)]
        if len(parts) < 2:
            raise ValueError("Watershed did not produce two valid split parts")
        union = np.logical_or.reduce(parts)
        if not np.array_equal(union, mask):
            raise ValueError("Split parts do not cover the original mask")
        if any(np.count_nonzero(left & right) for index, left in enumerate(parts) for right in parts[index + 1:]):
            raise ValueError("Split parts overlap")

        replacement = []
        for index, part in enumerate(parts):
            replacement.append(Parcel(
                part, target.score, target.cls, None, None, None,
                f"{target_id}:split{index}", False, "split_geometry"
            ))
        result = [parcel for parcel in result if parcel.parcel_id != target_id] + replacement
        return ToolProposal(
            "split_instances", [target_id], result,
            {"seed_count": len(points), "part_count": len(parts),
             "part_areas": [int(part.sum()) for part in parts]},
        )


class FragmentMerger:
    """Connect nearby components inside one prediction using image-edge evidence."""

    def propose(self, image: Array, parcels: Sequence[Parcel], target_id: str,
                policy: DomainPolicy) -> ToolProposal:
        if cv2 is None:
            raise RuntimeError("FragmentMerger requires OpenCV")
        result = [parcel.copy() for parcel in parcels]
        ids = target_id.split("|", 1)
        target = next(parcel for parcel in result if parcel.parcel_id == ids[0])
        original = _as_bool_mask(target.mask, tuple(image.shape[:2]))
        if len(ids) == 2:
            partner = next(parcel for parcel in result if parcel.parcel_id == ids[1])
            if partner.cls != target.cls or np.count_nonzero(original & partner.mask):
                raise ValueError("Fragment pair is not a disjoint same-class pair")
            original = original | _as_bool_mask(partner.mask, tuple(image.shape[:2]))
        components_before = _component_count(original)
        if components_before < 2:
            raise ValueError("Target has no disconnected fragments")
        edges = _edge_map(image)
        edge_before = _edge_alignment(image, original, edges)
        best = original
        best_score = (-1, edge_before)
        for gap in range(1, policy.fragment_gap + 1):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
            candidate = cv2.morphologyEx(original.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
            components_after = _component_count(candidate)
            edge_after = _edge_alignment(image, candidate, edges)
            if components_after < components_before and edge_after >= edge_before - 0.03:
                score = (components_before - components_after, edge_after)
                if score > best_score:
                    best, best_score = candidate, score
        if np.array_equal(best, original):
            raise ValueError("No adjacent fragment merge supported by image evidence")
        target.mask = best
        target.invalidate_quality()
        if len(ids) == 2:
            result = [parcel for parcel in result if parcel.parcel_id != ids[1]]
        return ToolProposal(
            "fragment_merge", ids, result,
            {"components_before": components_before, "components_after": _component_count(best),
             "edge_before": edge_before, "edge_after": _edge_alignment(image, best, edges)},
        )


class ContourSimplifier:
    """Reduce polygon vertices while preserving the raster mask within a fixed IoU."""

    def propose(self, image: Array, parcels: Sequence[Parcel], target_id: str,
                policy: DomainPolicy) -> ToolProposal:
        if cv2 is None:
            raise RuntimeError("ContourSimplifier requires OpenCV")
        result = [parcel.copy() for parcel in parcels]
        target = next(parcel for parcel in result if parcel.parcel_id == target_id)
        original = _as_bool_mask(target.mask, tuple(image.shape[:2]))
        contours, _ = cv2.findContours(original.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Cannot simplify an empty mask")
        vertex_before = int(sum(len(contour) for contour in contours))
        if vertex_before < 4:
            raise ValueError("Contour is already simple")
        best = original
        best_vertices = vertex_before
        best_iou = 1.0
        for multiplier in (0.5, 1.0, 1.5, 2.0, 3.0):
            candidate = np.zeros_like(original, dtype=np.uint8)
            for contour in contours:
                epsilon = max(0.5, policy.simplify_epsilon * multiplier * cv2.arcLength(contour, True))
                polygon = cv2.approxPolyDP(contour, epsilon, True)
                cv2.fillPoly(candidate, [polygon], 1)
            candidate = candidate.astype(bool)
            vertices = _contour_vertices(candidate)
            iou = _mask_iou(original, candidate)
            if vertices < best_vertices and iou >= policy.simplify_min_iou:
                best, best_vertices, best_iou = candidate, vertices, iou
        if np.array_equal(best, original):
            raise ValueError("No contour simplification satisfies the IoU constraint")
        target.mask = best
        target.invalidate_quality()
        return ToolProposal(
            "contour_simplify", [target_id], result,
            {"vertices_before": vertex_before, "vertices_after": best_vertices,
             "mask_iou": best_iou, "simplify_epsilon": policy.simplify_epsilon},
        )


class LocalReinfer:
    """Re-run the same model on a crop and replace one complete matched instance."""

    def __init__(self, predict_crop: CropPredictor, config: AgentConfig):
        self.predict_crop, self.config = predict_crop, config

    def propose(self, image: Array, parcels: Sequence[Parcel], target_id: str,
                policy: DomainPolicy) -> ToolProposal:
        target = next(parcel for parcel in parcels if parcel.parcel_id == target_id)
        height, width = image.shape[:2]
        x0, y0, x1, y1 = _bbox(target.mask)
        pad = policy.roi_padding
        x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        predicted = self.predict_crop(crop, policy.reinfer_imgsz, policy.reinfer_conf)
        matches = []
        for parcel in predicted:
            if parcel.cls != target.cls:
                continue
            mask = _as_bool_mask(parcel.mask, crop.shape[:2])
            if ((x0 > 0 and mask[:, 0].any()) or (x1 < width and mask[:, -1].any()) or
                    (y0 > 0 and mask[0].any()) or (y1 < height and mask[-1].any())):
                continue
            mapped = parcel.copy()
            mapped.mask = np.zeros((height, width), dtype=bool)
            mapped.mask[y0:y1, x0:x1] = mask
            matches.append((_mask_iou(target.mask, mapped.mask), mapped))
        matches.sort(key=lambda item: item[0], reverse=True)
        if not matches or matches[0][0] < self.config.min_reinfer_iou:
            raise ValueError("No complete ROI prediction matches the target")
        second = matches[1][0] if len(matches) > 1 else 0.0
        if matches[0][0] - second < self.config.reinfer_match_margin:
            raise ValueError("Ambiguous ROI instance match")
        replacement = matches[0][1]
        replacement.parcel_id = target_id
        result = [replacement if parcel.parcel_id == target_id else parcel.copy() for parcel in parcels]
        return ToolProposal("local_rerun", [target_id], result,
                            {"roi_xyxy": [x0, y0, x1, y1], "imgsz": policy.reinfer_imgsz,
                             "conf": policy.reinfer_conf, "match_iou": matches[0][0],
                             "match_margin": matches[0][0] - second})


class QualityVerifier:
    def __init__(self, config: AgentConfig):
        self.config = config

    @staticmethod
    def _unchanged(before: Parcel, after: Parcel) -> bool:
        return (np.array_equal(before.mask, after.mask) and before.score == after.score and
                before.cls == after.cls and before.quality == after.quality and
                before.mask_quality == after.mask_quality and
                before.boundary_quality == after.boundary_quality and
                before.quality_valid == after.quality_valid and
                before.quality_source == after.quality_source)

    def _evaluate_split(self, image: Array, old: Dict[str, Parcel], new: Dict[str, Parcel],
                        targets: set, policy: DomainPolicy) -> VerificationDecision:
        if len(targets) != 1:
            return VerificationDecision(False, ["split_requires_one_target"])
        target_id = next(iter(targets))
        if target_id not in old or target_id in new:
            return VerificationDecision(False, ["split_target_identity_changed"])
        old_other_ids = set(old) - {target_id}
        new_other_ids = set(new) & old_other_ids
        if new_other_ids != old_other_ids or len(new) - len(new_other_ids) < 2:
            return VerificationDecision(False, ["split_instance_identity_changed"])
        old_target = old[target_id]
        new_ids = sorted(set(new) - old_other_ids)
        parts = []
        for key in new_ids:
            parcel = new[key]
            mask = _as_bool_mask(parcel.mask, tuple(image.shape[:2]))
            if parcel.cls != old_target.cls or not mask.any() or not np.isfinite(parcel.score) or not 0 <= parcel.score <= 1:
                return VerificationDecision(False, ["invalid_split_part"])
            if int(mask.sum()) < policy.split_min_area or _component_count(mask) != 1:
                return VerificationDecision(False, ["invalid_split_part_geometry"])
            parts.append(mask)
        union = np.logical_or.reduce(parts)
        if not np.array_equal(union, old_target.mask):
            return VerificationDecision(False, ["split_does_not_cover_target"])
        if any(np.count_nonzero(left & right) for index, left in enumerate(parts) for right in parts[index + 1:]):
            return VerificationDecision(False, ["split_parts_overlap"])
        for key in old_other_ids:
            other = old[key]
            if any(np.count_nonzero(part & other.mask) > max(2, int(part.sum() * 0.01)) for part in parts):
                return VerificationDecision(False, ["split_overlaps_other_instance"])
            if not self._unchanged(other, new[key]):
                return VerificationDecision(False, ["untargeted_instance_changed"])
        edges = _edge_map(image)
        edge_before = _edge_alignment(image, old_target.mask, edges)
        total_area = max(1, sum(int(part.sum()) for part in parts))
        edge_after = sum(_edge_alignment(image, part, edges) * int(part.sum()) for part in parts) / total_area
        if edge_after < edge_before - self.config.max_edge_regression:
            return VerificationDecision(False, ["split_boundary_regression"], {
                "edge_before": edge_before, "edge_after": edge_after,
                "parts": len(parts),
            })
        return VerificationDecision(True, ["verified_split_candidate"], {
            "parts": len(parts), "areas": [int(part.sum()) for part in parts],
            "original_area": int(old_target.mask.sum()), "edge_before": edge_before,
            "edge_after": edge_after,
        })

    def _evaluate_merge(self, image: Array, old: Dict[str, Parcel], new: Dict[str, Parcel],
                        target_ids: Sequence[str], policy: DomainPolicy) -> VerificationDecision:
        if len(target_ids) != 2 or target_ids[0] == target_ids[1]:
            return VerificationDecision(False, ["merge_requires_two_targets"])
        left_id, right_id = target_ids
        if left_id not in old or right_id not in old or left_id not in new or right_id in new:
            return VerificationDecision(False, ["merge_target_identity_changed"])
        if set(new) != set(old) - {right_id}:
            return VerificationDecision(False, ["merge_instance_identity_changed"])
        for key in set(old) - {left_id, right_id}:
            if not self._unchanged(old[key], new[key]):
                return VerificationDecision(False, ["untargeted_instance_changed"])
        before_left, before_right = old[left_id], old[right_id]
        candidate = new[left_id]
        mask = _as_bool_mask(candidate.mask, tuple(image.shape[:2]))
        source = _as_bool_mask(before_left.mask, tuple(image.shape[:2])) | _as_bool_mask(before_right.mask, tuple(image.shape[:2]))
        if candidate.cls != before_left.cls or not mask.any() or not np.isfinite(candidate.score) or not 0 <= candidate.score <= 1:
            return VerificationDecision(False, ["invalid_merge_target"])
        if np.any(source & ~mask):
            return VerificationDecision(False, ["merge_dropped_source_pixels"])
        if _component_count(mask) != 1:
            return VerificationDecision(False, ["merge_not_connected"])
        added_fraction = (int(mask.sum()) - int(source.sum())) / max(1, int(source.sum()))
        if added_fraction > policy.max_area_change:
            return VerificationDecision(False, ["excessive_merge_bridge"], {"added_fraction": added_fraction})
        for key, other in new.items():
            if key == left_id:
                continue
            if np.count_nonzero(mask & other.mask) > max(2, int(mask.sum() * 0.01)):
                return VerificationDecision(False, ["new_instance_overlap"])
        if candidate.quality_valid or _quality(candidate) is not None:
            return VerificationDecision(False, ["stale_quality_after_geometry_change"])
        edges = _edge_map(image)
        edge_before = _edge_alignment(image, source, edges)
        edge_after = _edge_alignment(image, mask, edges)
        if edge_after < edge_before - self.config.max_edge_regression:
            return VerificationDecision(False, ["boundary_regression"], {
                "edge_before": edge_before, "edge_after": edge_after,
            })
        return VerificationDecision(True, ["verified_fragment_merge"], {
            "source_area": int(source.sum()), "candidate_area": int(mask.sum()),
            "added_fraction": added_fraction, "edge_before": edge_before,
            "edge_after": edge_after,
        })

    def evaluate(self, image: Array, before: Sequence[Parcel], proposal: ToolProposal,
                 policy: DomainPolicy) -> VerificationDecision:
        old = {parcel.parcel_id: parcel for parcel in before}
        new = {parcel.parcel_id: parcel for parcel in proposal.parcels}
        targets = set(proposal.target_ids)
        if not targets or not targets <= old.keys() or len(new) != len(proposal.parcels):
            return VerificationDecision(False, ["instance_identity_changed"])
        if proposal.action == "split_instances":
            return self._evaluate_split(image, old, new, targets, policy)
        if proposal.action == "fragment_merge" and len(proposal.target_ids) == 2:
            return self._evaluate_merge(image, old, new, proposal.target_ids, policy)
        if old.keys() != new.keys():
            return VerificationDecision(False, ["instance_identity_changed"])
        for key in old.keys() - targets:
            a, b = old[key], new[key]
            if not self._unchanged(a, b):
                return VerificationDecision(False, ["untargeted_instance_changed"])
        metrics = {}
        for key in targets:
            a, b = old[key], new[key]
            mask = _as_bool_mask(b.mask, tuple(image.shape[:2]))
            if a.cls != b.cls or not mask.any() or not np.isfinite(b.score) or not 0 <= b.score <= 1:
                return VerificationDecision(False, ["invalid_target"], metrics)
            area_change = abs(int(mask.sum()) - int(a.mask.sum())) / max(1, int(a.mask.sum()))
            edges = _edge_map(image)
            edge_before, edge_after = _edge_alignment(image, a.mask, edges), _edge_alignment(image, mask, edges)
            q_before, q_after = _quality(a), _quality(b)
            metrics[str(key)] = {"area_change": area_change, "edge_before": edge_before,
                                 "edge_after": edge_after, "quality_before": q_before, "quality_after": q_after}
            if area_change > policy.max_area_change:
                return VerificationDecision(False, ["excessive_area_change"], metrics)
            for other_key, other in new.items():
                if other_key == key:
                    continue
                previous = int(np.count_nonzero(a.mask & old[other_key].mask))
                following = int(np.count_nonzero(mask & other.mask))
                if following > max(previous, 2, int(mask.sum() * 0.01)):
                    return VerificationDecision(False, ["new_instance_overlap"], metrics)
            if cv2 is not None:
                count_before = cv2.connectedComponents(a.mask.astype(np.uint8), connectivity=8)[0]
                count_after = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0]
                if count_after > count_before:
                    return VerificationDecision(False, ["new_disconnected_fragments"], metrics)
            if proposal.action == "boundary_cleanup":
                if np.any((a.mask ^ mask) & ~_boundary_band(a.mask, policy.boundary_band)):
                    return VerificationDecision(False, ["change_outside_boundary_band"], metrics)
                if np.array_equal(a.mask, mask):
                    return VerificationDecision(False, ["no_change"], metrics)
                if q_after is not None or b.quality_valid:
                    return VerificationDecision(False, ["stale_quality_after_geometry_change"], metrics)
                if edge_after - edge_before < policy.min_edge_gain:
                    return VerificationDecision(False, ["insufficient_boundary_evidence"], metrics)
            elif proposal.action == "local_rerun":
                if edge_after < edge_before - self.config.max_edge_regression:
                    return VerificationDecision(False, ["boundary_regression"], metrics)
                if q_after is None:
                    return VerificationDecision(False, ["candidate_quality_unavailable"], metrics)
                required = self.config.accept_quality if q_before is None else q_before + self.config.min_action_gain
                if q_after < required:
                    return VerificationDecision(False, ["insufficient_quality_gain"], metrics)
            elif proposal.action in {"fragment_merge", "contour_simplify"}:
                if np.array_equal(a.mask, mask):
                    return VerificationDecision(False, ["no_change"], metrics)
                if q_after is not None or b.quality_valid:
                    return VerificationDecision(False, ["stale_quality_after_geometry_change"], metrics)
                if edge_after < edge_before - self.config.max_edge_regression:
                    return VerificationDecision(False, ["boundary_regression"], metrics)
                if proposal.action == "fragment_merge":
                    if _component_count(mask) >= _component_count(a.mask):
                        return VerificationDecision(False, ["fragments_not_merged"], metrics)
                else:
                    before_vertices = _contour_vertices(a.mask)
                    after_vertices = _contour_vertices(mask)
                    if after_vertices >= before_vertices:
                        return VerificationDecision(False, ["contour_not_simplified"], metrics)
                    if _mask_iou(a.mask, mask) < policy.simplify_min_iou:
                        return VerificationDecision(False, ["simplification_iou_regression"], metrics)
            else:
                return VerificationDecision(False, ["unsupported_tool_action"], metrics)
        return VerificationDecision(True, ["verified_candidate"], metrics)


class FieldBqcAgent:
    def __init__(self, config: AgentConfig | None = None, rerun_tool: RerunTool | None = None,
                 local_tool: LocalReinfer | None = None, router: DomainRouter | None = None,
                 planner: Any = None, routing_mode: str = "rule", verify: bool = True,
                 proposal_observer: Optional[Callable] = None,
                 recipes: RepairRecipeLibrary | Mapping[str, Sequence[str]] | None = None):
        if routing_mode not in {"rule", "fixed", "blind"}:
            raise ValueError("routing_mode must be rule, fixed, or blind")
        self.config = config or AgentConfig()
        self.planner, self.routing_mode, self.verify = planner, routing_mode, verify
        self.proposal_observer = proposal_observer
        self.rerun_tool, self.local_tool = rerun_tool, local_tool
        self.router = router or DomainRouter()
        self.recipes = recipes if isinstance(recipes, RepairRecipeLibrary) else RepairRecipeLibrary(recipes)
        self.boundary_tool = BoundaryRefiner()
        self.split_tool = AdhesionSplitter(self.config)
        self.merge_tool = FragmentMerger()
        self.simplify_tool = ContourSimplifier()
        self.verifier = QualityVerifier(self.config)

    def inspect(self, image: Array, parcels: Sequence[Parcel], metadata: Mapping[str, Any] | None = None,
                tta_disagreement: float = 0.0) -> AgentState:
        if image.ndim not in (2, 3):
            raise ValueError("Image must be HxW or HxWxC")
        policy = self.router.route(metadata or {})
        edges = _edge_map(image)
        observations = []
        masks_by_index = {}
        for index, parcel in enumerate(parcels):
            mask = _as_bool_mask(parcel.mask, tuple(image.shape[:2]))
            masks_by_index[index] = mask
            quality = _quality(parcel)
            alignment = _edge_alignment(image, mask, edges)
            x0, y0, x1, y1 = _bbox(mask)
            margin = self.config.border_margin
            border = bool(mask.any() and (x0 < margin or y0 < margin or
                          x1 > mask.shape[1] - margin or y1 > mask.shape[0] - margin))
            area_fraction = float(mask.sum() / mask.size)
            peaks = _peak_count(mask, self.config.min_peak_distance, self.config.min_peak_value)
            merge_risk = min(1.0, max(0, peaks - 1) / 2.0)
            component_count = _component_count(mask)
            fragment_risk = min(1.0, max(0, component_count - 1) / 2.0)
            contour_vertices = _contour_vertices(mask)
            issues, reasons = [], []
            for condition, issue, reason in (
                (not mask.any(), "invalid_mask", "empty_instance_mask"),
                (border, "border_truncation", "touches_image_border"),
                (mask.any() and area_fraction < policy.small_area_fraction, "small_instance", "small_predicted_area"),
                (quality is None, "quality_unavailable", "mqc_missing_or_invalidated"),
                (quality is not None and quality < self.config.review_quality, "low_quality", "low_mqc_quality"),
                (alignment < self.config.min_edge_alignment or
                 (parcel.quality_valid and parcel.boundary_quality is not None and
                  parcel.boundary_quality < self.config.review_quality), "boundary_shift", "weak_boundary_evidence"),
                (merge_risk >= self.config.split_merge_risk, "merge_warning", "multiple_distance_peaks"),
                (fragment_risk > 0, "fragmentation", "disconnected_mask_components"),
                (contour_vertices > self.config.max_contour_vertices, "contour_complexity", "redundant_contour_vertices"),
            ):
                if condition:
                    issues.append(issue)
                    reasons.append(reason)
            severity = 0.75 if any(i in issues for i in ("invalid_mask", "border_truncation", "low_quality")) else 0.5 if issues else 0.0
            observations.append(ParcelObservation(index, quality, alignment, float(border), merge_risk, fragment_risk,
                                bool(mask.any()), issues[0] if issues else "normal", severity,
                                reasons, issues, str(parcel.parcel_id if parcel.parcel_id is not None else index), area_fraction,
                                component_count, contour_vertices))
            observations[-1].distance_peak_count = peaks
        if cv2 is not None:
            small_area_limit = policy.small_area_fraction * 8.0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (2 * policy.fragment_gap + 1, 2 * policy.fragment_gap + 1))
            for left_index, left in enumerate(observations):
                for right_index in range(left_index + 1, len(observations)):
                    right = observations[right_index]
                    if parcels[left_index].cls != parcels[right_index].cls:
                        continue
                    if min(left.area_fraction, right.area_fraction) > small_area_limit:
                        continue
                    left_mask, right_mask = masks_by_index[left_index], masks_by_index[right_index]
                    if np.count_nonzero(left_mask & right_mask):
                        continue
                    if not np.any(cv2.dilate(left_mask.astype(np.uint8), kernel) & right_mask.astype(np.uint8)):
                        continue
                    left.merge_partner = right.parcel_id
                    right.merge_partner = left.parcel_id
                    for item in (left, right):
                        if "fragmentation" not in item.issue_types:
                            item.issue_types.append("fragmentation")
                            item.reasons.append("adjacent_small_fragment")
                            if item.issue_type == "normal":
                                item.issue_type = "fragmentation"
        known = [obs.quality for obs in observations if obs.quality is not None]
        return AgentState(float(np.mean(known)) if known else None,
                          float(np.mean([obs.edge_alignment for obs in observations])) if observations else 0.0,
                          float(np.clip(tta_disagreement, 0, 1)), sum(obs.severity >= 0.65 for obs in observations),
                          _topology_valid(parcels, tuple(image.shape[:2])), observations)

    def _next_action(self, state: AgentState, attempted: set, local_calls: int):
        for obs in sorted(state.observations, key=lambda item: item.severity, reverse=True):
            issues = set(obs.issue_types)
            if "invalid_mask" in issues:
                continue
            allowed = set(self.recipes.actions_for(obs.issue_types))
            unstable = state.tta_disagreement > self.config.max_tta_disagreement
            if (self.config.enable_split_tool and cv2 is not None and "split_instances" in allowed and
                    (obs.parcel_id, "split_instances") not in attempted):
                return "split_instances", obs.parcel_id
            if (self.config.enable_merge_tool and cv2 is not None and "fragment_merge" in allowed and
                    (obs.parcel_id, "fragment_merge") not in attempted):
                target_ref = obs.parcel_id
                if obs.merge_partner is not None:
                    target_ref = "|".join(sorted((obs.parcel_id, obs.merge_partner)))
                if (target_ref, "fragment_merge") not in attempted:
                    return "fragment_merge", target_ref
            if (self.config.enable_simplify_tool and cv2 is not None and "contour_simplify" in allowed and
                    (obs.parcel_id, "contour_simplify") not in attempted):
                return "contour_simplify", obs.parcel_id
            if ((self.local_tool is not None or self.rerun_tool is not None) and
                    local_calls < self.config.max_local_calls and
                    (unstable or "local_rerun" in allowed) and
                    (obs.parcel_id, "local_rerun") not in attempted):
                return "local_rerun", obs.parcel_id
            if (self.config.enable_boundary_tool and cv2 is not None and "boundary_cleanup" in allowed and
                    (obs.parcel_id, "boundary_cleanup") not in attempted):
                return "boundary_cleanup", obs.parcel_id
        return None

    def _next_fixed_action(self, state: AgentState, attempted: set, local_calls: int):
        """Choose the first eligible action in a fixed global order.

        Unlike the legacy blind route, each tool is still gated by the issue
        it is meant to address. This gives a meaningful fixed-policy control:
        the order is fixed, while the diagnosis remains observable-only.
        """
        ordered = (
            ("split_instances", {"merge_warning"}),
            ("fragment_merge", {"fragmentation"}),
            ("contour_simplify", {"contour_complexity"}),
            ("local_rerun", {"small_instance", "low_quality"}),
            ("boundary_cleanup", {"boundary_shift"}),
        )
        unstable = state.tta_disagreement > self.config.max_tta_disagreement
        observations = sorted(state.observations, key=lambda item: item.severity, reverse=True)
        for action, issue_types in ordered:
            for obs in observations:
                issues = set(obs.issue_types)
                allowed = set(self.recipes.actions_for(obs.issue_types))
                if "invalid_mask" in issues:
                    continue
                if action not in allowed:
                    continue
                if action == "split_instances" and not self.config.enable_split_tool:
                    continue
                if action == "fragment_merge" and not self.config.enable_merge_tool:
                    continue
                if action == "contour_simplify" and not self.config.enable_simplify_tool:
                    continue
                if action == "boundary_cleanup" and not self.config.enable_boundary_tool:
                    continue
                if action == "local_rerun":
                    if local_calls >= self.config.max_local_calls:
                        continue
                    if not (unstable or issues & issue_types):
                        continue
                elif not issues & issue_types:
                    continue
                target_ref = obs.parcel_id
                if action == "fragment_merge" and obs.merge_partner:
                    target_ref = "|".join(sorted((obs.parcel_id, obs.merge_partner)))
                if (target_ref, action) not in attempted:
                    return action, target_ref
        return None

    def choose_action(self, state: AgentState) -> str:
        action = self._next_action(state, set(), 0)
        return action[0] if action else "review" if any(obs.issue_types for obs in state.observations) else "accept"

    def available_actions(self, state: AgentState, attempted: set, local_calls: int):
        """Enumerate enabled tools without using labels or quality thresholds."""
        actions = []
        for obs in state.observations:
            if "invalid_mask" in obs.issue_types:
                continue
            allowed = set(self.recipes.actions_for(obs.issue_types))
            for enabled, action in (
                (self.config.enable_split_tool and cv2 is not None, "split_instances"),
                (self.config.enable_merge_tool and cv2 is not None, "fragment_merge"),
                (self.config.enable_simplify_tool and cv2 is not None, "contour_simplify"),
                ((self.local_tool is not None or self.rerun_tool is not None) and
                 local_calls < self.config.max_local_calls, "local_rerun"),
                (self.config.enable_boundary_tool and cv2 is not None, "boundary_cleanup"),
            ):
                if action not in allowed:
                    continue
                refs = [obs.parcel_id]
                if action == "fragment_merge" and obs.merge_partner:
                    refs.append("|".join(sorted((obs.parcel_id, obs.merge_partner))))
                for ref in refs:
                    if enabled and (ref, action) not in attempted:
                        entry = {"action": action, "target_ids": ref.split("|")}
                        if entry not in actions:
                            actions.append(entry)
        return actions

    def run(self, image: Array, parcels: Sequence[Parcel], metadata: Mapping[str, Any] | None = None,
            tta_disagreement: float = 0.0) -> AgentResult:
        metadata = dict(metadata or {})
        policy = self.router.route(metadata)
        current = [parcel.copy() for parcel in parcels]
        for index, parcel in enumerate(current):
            parcel.parcel_id = str(parcel.parcel_id if parcel.parcel_id is not None else index)
            _as_bool_mask(parcel.mask, tuple(image.shape[:2]))
        if len({parcel.parcel_id for parcel in current}) != len(current):
            raise ValueError("Instance ids must be unique")
        state = self.inspect(image, current, metadata, tta_disagreement)
        attempted, trace, review = set(), [], set()
        started, local_calls = time.monotonic(), 0
        stop_reason = "action_budget"
        for iteration in range(self.config.max_iterations):
            if time.monotonic() - started >= self.config.max_seconds:
                stop_reason = "time_budget"
                trace.append({"action": "stop", "reason": "time_budget", "accepted": False})
                break
            next_action = None
            if self.planner is not None:
                available = self.available_actions(state, attempted, local_calls)
                if available:
                    try:
                        from fieldbqc_planner import build_context, validate_goal

                        context = build_context(current, state, policy, available, trace,
                                                self.config.max_iterations - iteration,
                                                self.config.max_local_calls - local_calls)
                        plan = self.planner.plan(context, image, current,
                                                 self.config.max_seconds - (time.monotonic() - started))
                        goal = validate_goal(plan["goal"], context)
                        trace.append({"action": "plan", "iteration": iteration,
                                      "context": context, **plan})
                        if goal["decision"] != "repair":
                            stop_reason = "planner_" + goal["decision"]
                            if goal["decision"] == "review":
                                review.update(goal["target_ids"])
                            break
                        next_action = goal["action"], "|".join(goal["target_ids"])
                    except Exception as error:
                        trace.append({**getattr(error, "record", {}), "action": "plan", "iteration": iteration,
                                      "fallback": "rule", "error": f"{type(error).__name__}: {error}"})
                        next_action = self._next_action(state, attempted, local_calls)
            elif self.routing_mode == "fixed":
                next_action = self._next_fixed_action(state, attempted, local_calls)
            elif self.routing_mode == "blind":
                available = self.available_actions(state, attempted, local_calls)
                if available:
                    first = available[0]
                    next_action = first["action"], "|".join(first["target_ids"])
            else:
                next_action = self._next_action(state, attempted, local_calls)
            if self.planner is not None and time.monotonic() - started >= self.config.max_seconds:
                stop_reason = "time_budget"
                break
            if next_action is None:
                stop_reason = "no_eligible_tool"
                break
            action, target_ref = next_action
            target_ids = target_ref.split("|") if action == "fragment_merge" else [target_ref]
            attempted.add((target_ref, action))
            action_started = time.monotonic()
            if action == "local_rerun":
                local_calls += 1
            try:
                snapshot = [parcel.copy() for parcel in current]
                if action == "split_instances":
                    proposal = self.split_tool.propose(image, snapshot, target_ref, policy)
                elif action == "fragment_merge":
                    proposal = self.merge_tool.propose(image, snapshot, target_ref, policy)
                elif action == "contour_simplify":
                    proposal = self.simplify_tool.propose(image, snapshot, target_ref, policy)
                elif action == "boundary_cleanup":
                    proposal = self.boundary_tool.propose(image, snapshot, target_ref, policy)
                elif self.local_tool is not None:
                    proposal = self.local_tool.propose(image, snapshot, target_ref, policy)
                else:
                    callback_meta = dict(metadata, target_ids=target_ids, domain_policy=policy.to_dict())
                    candidate = list(self.rerun_tool(image, snapshot, action, callback_meta))
                    proposal = ToolProposal(action, target_ids, candidate)
                decision = (self.verifier.evaluate(image, current, proposal, policy) if self.verify else
                            VerificationDecision(True, ["verifier_disabled_for_ablation"]))
                if self.proposal_observer is not None:
                    self.proposal_observer(iteration, current, proposal, decision)
                if decision.accept:
                    current = [parcel.copy() for parcel in proposal.parcels]
                    review.difference_update(target_ids)
                else:
                    review.update(target_ids)
                trace.append({"iteration": iteration, "action": action, "target_ids": target_ids,
                              "accepted": decision.accept, "rolled_back": not decision.accept,
                              "reasons": decision.reasons, "metrics": decision.metrics,
                              "evidence": proposal.evidence, "elapsed_seconds": time.monotonic() - action_started})
            except Exception as error:
                review.update(target_ids)
                trace.append({"iteration": iteration, "action": action, "target_ids": target_ids,
                              "accepted": False, "rolled_back": True,
                              "reasons": ["tool_error"], "error": f"{type(error).__name__}: {error}",
                              "elapsed_seconds": time.monotonic() - action_started})
            state = self.inspect(image, current, metadata, tta_disagreement)
        for obs in state.observations:
            if obs.issue_types or obs.quality is None or obs.quality < self.config.accept_quality:
                review.add(obs.parcel_id)
        if not state.topology_valid or state.tta_disagreement > self.config.max_tta_disagreement:
            review.update(parcel.parcel_id for parcel in current)
        needs_review = bool(review) or not current
        accepted = not needs_review
        trace.append({"action": "accept" if accepted else "review", "accepted": accepted,
                      "review_ids": sorted(review), "local_calls": local_calls,
                      "stop_reason": stop_reason,
                      "empty_scene_unverified": not current, "elapsed_seconds": time.monotonic() - started})
        return AgentResult(current, accepted, needs_review, trace, state, sorted(review), policy.to_dict())


def parcels_from_ultralytics(result: Any) -> List[Parcel]:
    if getattr(result, "boxes", None) is None or getattr(result, "masks", None) is None:
        return []
    masks = result.masks.data.detach().cpu().numpy() > 0.5
    if masks.shape[1:] != tuple(result.orig_shape):
        raise ValueError("Agent requires original-size masks; enable retina_masks=True")
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    def quality_values(name):
        values = getattr(result, name, None)
        if values is None:
            return [None] * len(masks)
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        values = np.asarray(values).reshape(-1)
        if len(values) != len(masks):
            raise ValueError(f"{name} does not align with retained instances")
        return [float(value) if np.isfinite(value) and 0 <= value <= 1 else None for value in values]

    quality = quality_values("mqc_combined_scores")
    mask_quality = quality_values("mqc_mask_iou_scores")
    boundary_quality = quality_values("mqc_boundary_f1_scores")
    return [Parcel(mask, float(score), int(cls), quality[i], mask_quality[i], boundary_quality[i], str(i),
                   quality[i] is not None, "mqc" if quality[i] is not None else "unavailable")
            for i, (mask, score, cls) in enumerate(zip(masks, scores, classes))]


def run_self_test():
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[20:70, 20:60] = True
    result = FieldBqcAgent(AgentConfig(min_edge_alignment=0)).run(image, [Parcel(mask, 0.9, quality=0.9)])
    assert result.accepted and not result.needs_review
    assert np.array_equal(result.parcels[0].mask, mask)
    print("FieldBQC-Agent self-test passed.")


if __name__ == "__main__":
    run_self_test()
