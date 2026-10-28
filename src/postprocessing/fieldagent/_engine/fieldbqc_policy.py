"""Inference-only domain policies and image metadata (never reads labels)."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DomainPolicy:
    # Starting points for development, not calibrated accuracy guarantees.
    domain: str = "unknown"
    small_area_fraction: float = 0.002
    roi_padding: int = 24
    reinfer_imgsz: int = 960
    reinfer_conf: float = 0.15
    boundary_band: int = 2
    fragment_gap: int = 4
    split_min_area: int = 20
    simplify_epsilon: float = 0.01
    simplify_min_iou: float = 0.95
    max_area_change: float = 0.20
    min_edge_gain: float = 0.01
    fusion_seam_iou: float = 0.70
    fusion_overlap_iou: float = 0.50
    fusion_match_margin: float = 0.10
    roi_padding_m: float | None = None
    boundary_band_m: float | None = None

    def __post_init__(self):
        for name in ("small_area_fraction", "reinfer_conf", "simplify_epsilon", "simplify_min_iou",
                     "max_area_change", "min_edge_gain",
                     "fusion_seam_iou", "fusion_overlap_iou", "fusion_match_margin"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid domain policy {name}: {value}")
        for name in ("roi_padding", "reinfer_imgsz", "boundary_band", "fragment_gap", "split_min_area"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Invalid domain policy {name}: {value}")
        for name in ("roi_padding_m", "boundary_band_m"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"Invalid physical policy {name}: {value}")

    def to_dict(self):
        return asdict(self)


def resolution_m(value: Any) -> float | None:
    if value is None:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(m|cm)?\s*", str(value), re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1)) * (0.01 if (match.group(2) or "").lower() == "cm" else 1)
    return number if math.isfinite(number) and number > 0 else None


class DomainRouter:
    def __init__(self, overrides: Mapping[str, Mapping[str, Any]] | None = None):
        base = DomainPolicy()
        self.policies = {
            "unknown": base,
            "ddfhapd": replace(base, domain="ddfhapd"),
            "ftw": replace(base, domain="ftw", small_area_fraction=0.004, roi_padding=16),
            "s2": replace(base, domain="s2", roi_padding=32, boundary_band=1,
                          fusion_seam_iou=0.80, roi_padding_m=320, boundary_band_m=10),
            "planet": replace(base, domain="planet", roi_padding=24, roi_padding_m=72, boundary_band_m=6),
            "ortho": replace(base, domain="ortho", roi_padding=24, roi_padding_m=24, boundary_band_m=2),
            "maxar": replace(base, domain="maxar", small_area_fraction=0.001,
                             fusion_seam_iou=0.85, fusion_match_margin=0.15,
                             roi_padding_m=12, boundary_band_m=0.5),
        }
        allowed = {item.name for item in fields(DomainPolicy)} - {"domain"}
        for domain, values in (overrides or {}).items():
            if domain not in self.policies or set(values) - allowed:
                raise ValueError(f"Unknown domain or policy fields: {domain}: {set(values) - allowed}")
            values = dict(values)
            for pixels, meters in (("roi_padding", "roi_padding_m"), ("boundary_band", "boundary_band_m")):
                if pixels in values and meters not in values:
                    values[meters] = None
            self.policies[domain] = replace(self.policies[domain], **values)

    @staticmethod
    def _resolve_gsd(policy, metadata):
        gsd = resolution_m(metadata.get("resolution_m") or metadata.get("resolution"))
        changes = {}
        if gsd is not None:
            if policy.roi_padding_m is not None:
                changes["roi_padding"] = max(8, min(256, math.ceil(policy.roi_padding_m / gsd)))
            if policy.boundary_band_m is not None:
                changes["boundary_band"] = max(1, min(8, math.ceil(policy.boundary_band_m / gsd)))
        return replace(policy, **changes)

    def route(self, metadata: Mapping[str, Any]) -> DomainPolicy:
        explicit = str(metadata.get("domain", "auto")).lower()
        if explicit not in ("", "auto"):
            if explicit not in self.policies:
                raise ValueError(f"Unknown domain: {explicit}")
            return self._resolve_gsd(self.policies[explicit], metadata)
        source = f"{metadata.get('dataset', '')} {metadata.get('source', '')}".lower()
        sensor = str(metadata.get("sensor", "")).lower()
        text = f"{source} {sensor}"
        if "ddfhapd" in source:
            domain = "ddfhapd"
        elif "ftw" in source:
            domain = "ftw"
        elif re.search(r"sentinel.?2|(?:^|[_\W])s2(?:[_\W]|$)", text):
            domain = "s2"
        elif "planet" in text:
            domain = "planet"
        elif "ortho" in text:
            domain = "ortho"
        elif any(token in text for token in ("maxar", "worldview", "quickbird", "geoeye")):
            domain = "maxar"
        else:
            # Resolution alone does not identify a sensor or a data source.
            domain = "unknown"
        return self._resolve_gsd(self.policies[domain], metadata)


def load_metadata_manifest(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    else:
        with path.open(encoding="utf-8-sig") as stream:
            document = json.load(stream)
        records = document.get("images", []) if isinstance(document, dict) else document
    index = {}
    for record in records:
        key = record.get("image_id") or record.get("image_path") or record.get("source")
        if not key:
            raise ValueError("Manifest record needs image_id, image_path or source")
        key = Path(str(key)).stem
        if key in index:
            raise ValueError(f"Ambiguous manifest image id: {key}")
        index[key] = dict(record)
    return index


def image_metadata(path: str, width: int, height: int, supplied: Mapping[str, Any] | None = None) -> dict:
    metadata = {"source": path, "width": width, "height": height}
    metadata.update({key: value for key, value in (supplied or {}).items() if value not in (None, "")})
    metadata["source"] = path
    metadata["width"], metadata["height"] = width, height
    gsd = resolution_m(metadata.get("resolution_m") or metadata.get("resolution"))
    metadata["resolution_m"] = gsd
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as dataset:
                if dataset.crs is not None:
                    metadata["crs"] = dataset.crs.to_string()
                    metadata["transform"] = list(dataset.transform)[:6]
                    metadata["bounds"] = list(dataset.bounds)
                    metadata["geo_source"] = "geotiff"
                    if dataset.crs.is_projected:
                        factor = dataset.crs.linear_units_factor[1]
                        metadata["resolution_m"] = math.hypot(dataset.transform.a, dataset.transform.d) * factor
        except (ImportError, OSError, ValueError) as error:
            metadata["geo_warning"] = f"{type(error).__name__}: {error}"
    return metadata
