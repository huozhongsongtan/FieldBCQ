"""Dataset/grid helpers for Tables 4--10."""

import csv
import math
import os
from pathlib import Path

from project_paths import project_path


def resolve_images(data, split="test"):
    """Resolve once, then pass this exact image list to the validator."""
    import yaml
    data_path = project_path(data)
    config = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    root = Path(os.path.expandvars(str(config.get("path") or "."))).expanduser()
    root = (data_path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    entries = config[split]
    images = []
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    for entry in entries if isinstance(entries, list) else [entries]:
        path = Path(os.path.expandvars(str(entry))).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if path.is_dir():
            images.extend(sorted(p.resolve() for p in path.rglob("*") if p.suffix.lower() in suffixes))
        elif path.suffix.lower() == ".txt" and path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                image = Path(os.path.expandvars(line.strip())).expanduser()
                if not image.is_absolute():
                    local = path.parent / image
                    image = local if local.exists() else root / image
                images.append(image.resolve())
        elif path.is_file() and path.suffix.lower() in suffixes:
            images.append(path)
        else:
            raise FileNotFoundError(f"Missing test entry: {path}")
    if not images or len(set(images)) != len(images):
        raise ValueError("Test image list is empty or contains duplicates")
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(image)
    if len(config.get("names", ["field"])) != 1:
        raise ValueError("The manuscript protocol requires one field class")
    return data_path, config, root, images


def load_metadata(path, root, images):
    """Accept relative/absolute image paths or unambiguous basenames; require full coverage."""
    with project_path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"image", "source", "resolution_m"}.issubset(rows[0]):
        raise ValueError("Metadata requires image,source,resolution_m columns")
    if len(rows) != len(images):
        raise ValueError(f"Test set must match the metadata exactly: {len(images)} images vs {len(rows)} rows")
    by_path, by_name = {}, {}
    for row in rows:
        value = Path(row["image"].replace("\\", "/"))
        key = (root / value).resolve() if not value.is_absolute() else value.resolve()
        if key in by_path:
            raise ValueError(f"Duplicate metadata image: {value}")
        by_path[key] = row
        by_name.setdefault(value.name, []).append(row)
    image_names = {}
    for image in images:
        image_names[image.name] = image_names.get(image.name, 0) + 1
    result = {}
    for image in images:
        row = by_path.get(image)
        if row is None:
            matches = by_name.get(image.name, [])
            if len(matches) != 1 or image_names[image.name] != 1:
                raise ValueError(f"Missing or ambiguous metadata: {image}")
            row = matches[0]
        source = row["source"].strip().upper().replace("DDFHAPD", "FHAPD")
        resolution = float(row["resolution_m"])
        if source not in {"FBIS", "FHAPD", "FTW"} or not math.isfinite(resolution) or resolution <= 0:
            raise ValueError(f"Invalid source/resolution: {image}")
        result[image] = {**row, "source": source, "resolution_m": format(resolution, "g")}
    return result

def evaluation_content_shape(original_shape, size, ratio_pad=None):
    """Use recorded loader resize ratios (height, width), including ceil rounding."""
    height, width = original_shape
    if size <= 0 or min(height, width) <= 0:
        raise ValueError("Image and evaluation grid dimensions must be positive")
    if ratio_pad is None:
        gain = min(size / height, size / width)
        content_h, content_w = round(height * gain), round(width * gain)
    else:
        (gain_h, gain_w), _ = ratio_pad
        if not all(math.isfinite(float(g)) and g > 0 for g in (gain_h, gain_w)):
            raise ValueError("Invalid recorded image resize ratios")
        content_h, content_w = round(height * gain_h), round(width * gain_w)
    if not (0 < content_h <= size and 0 < content_w <= size):
        raise ValueError(f"Resized content {(content_h, content_w)} cannot fit the {size} square grid")
    return content_h, content_w


def load_targets(label, original_shape, size, ratio_pad=None):
    import cv2
    import numpy as np
    height, width = original_shape
    scale = min(size / height, size / width)
    content_h, content_w = evaluation_content_shape(original_shape, size, ratio_pad)
    left = round((size - content_w) / 2 - 0.1)
    top = round((size - content_h) / 2 - 0.1)
    # Retain the original standalone/predictor convention when no loader metadata is supplied.
    polygon_scale = [width * scale, height * scale] if ratio_pad is None else [content_w, content_h]
    classes, masks = [], []
    for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = np.asarray([float(item) for item in line.split()])
        if len(values) < 7 or (len(values) - 1) % 2 or not np.isfinite(values).all():
            raise ValueError(f"Malformed polygon: {label}:{line_number}")
        if values[0] != 0 or (values[1:] < 0).any() or (values[1:] > 1).any():
            raise ValueError(f"Require class 0 and normalized polygon coordinates: {label}:{line_number}")
        points = values[1:].reshape(-1, 2) * polygon_scale + [left, top]
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        if not mask.any():
            raise ValueError(f"Polygon has no rasterized area: {label}:{line_number}")
        classes.append(int(values[0]))
        masks.append(mask.astype(bool))
    return np.asarray(masks, dtype=bool).reshape(-1, size, size), np.asarray(classes, dtype=int), scale


def square_evaluation_masks(masks, original_shape, size, ratio_pad=None):
    """Translate loader/predictor masks to the square GT grid without resampling.

    Rect validation can add stride padding beyond imgsz (e.g. 672 for imgsz=640).
    Crop only portions outside the destination grid, or add missing padding.
    Recorded loader ratios/padding account for non-square image resize rounding.
    """
    import numpy as np
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or size <= 0 or min(original_shape) <= 0:
        raise ValueError("Expected N,H,W masks and positive original/image sizes")
    content_h, content_w = evaluation_content_shape(original_shape, size, ratio_pad)
    mask_h, mask_w = masks.shape[1:]
    if mask_h < content_h or mask_w < content_w:
        raise ValueError(f"Mask grid {masks.shape[1:]} is incompatible with imgsz={size}, original={original_shape}")
    square_top = round((size - content_h) / 2 - 0.1)
    square_left = round((size - content_w) / 2 - 0.1)
    if ratio_pad is None:
        rect_top = round((mask_h - content_h) / 2 - 0.1)
        rect_left = round((mask_w - content_w) / 2 - 0.1)
    else:
        rect_left, rect_top = ratio_pad[1]
        if any(not math.isfinite(float(v)) or int(v) != v for v in (rect_top, rect_left)):
            raise ValueError("Loader padding must be integer pixel offsets")
        rect_left, rect_top = int(rect_left), int(rect_top)
    if min(rect_top, rect_left) < 0 or rect_top + content_h > mask_h or rect_left + content_w > mask_w:
        raise ValueError("Recorded padding/content does not fit the mask grid")
    top, left = square_top - rect_top, square_left - rect_left
    if (mask_h, mask_w) == (size, size) and top == left == 0:
        return masks
    output = np.zeros((len(masks), size, size), dtype=bool)
    source_top, source_left = max(0, -top), max(0, -left)
    dest_top, dest_left = max(0, top), max(0, left)
    copy_h = min(mask_h - source_top, size - dest_top)
    copy_w = min(mask_w - source_left, size - dest_left)
    output[:, dest_top:dest_top + copy_h, dest_left:dest_left + copy_w] = masks[
        :, source_top:source_top + copy_h, source_left:source_left + copy_w]
    return output
