"""Validated input manifests and reproducible, credential-free experiment files."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_inputs(path):
    """Resolve paths relative to this file, never relative to the shell cwd."""
    path = Path(path).resolve()
    document = load(path)
    if document.get("coordinate_space") != "image_pixels":
        raise ValueError("Declare coordinate_space=image_pixels; geographic coordinates require conversion")
    entries = document.get("inputs", [])
    if not entries:
        raise ValueError("The input manifest is empty")
    result, seen = [], set()
    for source in entries:
        scene = source.get("scene") or Path(source["image"]).stem
        if not re.fullmatch(r"[\w.-]+", scene) or scene in {".", ".."} or scene.casefold() in seen:
            raise ValueError("Unsafe or duplicate scene identifier: " + scene)
        seen.add(scene.casefold())
        entry = {"scene": scene, "domain": source.get("domain", scene.split("_")[0])}
        for name in ("image", "prediction_vector", "label"):
            if not source.get(name):
                if name != "label":
                    raise ValueError("Missing input " + name)
                continue
            value = Path(source[name])
            value = (path.parent / value).resolve() if not value.is_absolute() else value.resolve()
            if not value.is_file():
                raise FileNotFoundError(value)
            actual = digest(value)
            expected = source.get(name + "_sha256")
            if expected is not None and expected != actual:
                raise ValueError("Input changed: " + str(value))
            entry[name] = str(value)
            entry[name + "_sha256"] = actual
        result.append(entry)
    return result


def import_cache(cache, output, *, limit=0):
    """Reference frozen prediction files; do not copy images, weights or run data."""
    cache, output = Path(cache).resolve(), Path(output).resolve()
    parent = load(cache / "manifest.json")
    entries = parent["entries"][:limit or None]
    if not entries or limit < 0:
        raise ValueError("No input entries or invalid limit")
    rows = []
    for entry in entries:
        scene = entry["scene"]
        folder = cache / "predictions" / scene
        vector = folder / "prediction.pixel.geojson"
        completed = load(folder / "completed.json")
        if completed["image_sha256"] != entry["image_sha256"] or digest(vector) != completed["vector_sha256"]:
            raise ValueError("Unverified prediction cache: " + scene)
        row = {"scene": scene, "domain": entry["domain"], "image": entry["image"],
               "image_sha256": entry["image_sha256"], "prediction_vector": str(vector),
               "prediction_vector_sha256": completed["vector_sha256"]}
        if entry.get("label"):
            row.update(label=entry["label"], label_sha256=entry["label_sha256"])
        rows.append(row)
    document = {"coordinate_space": "image_pixels", "inputs": rows,
                "source_cache_manifest_sha256": digest(cache / "manifest.json"),
                "labels_offline_only": True, "subset_limit": limit}
    if output.exists() and load(output) != document:
        raise ValueError("Input manifest already exists with different contents")
    save(output, document)
    read_inputs(output)
    return len(rows)


PLANNER_KEYS = {"api", "base_url", "model", "api_key_env", "vision", "timeout_seconds",
                "max_output_tokens", "max_instances", "request_overrides"}


def planner_config(path):
    config = load(path)
    if set(config) - PLANNER_KEYS:
        raise ValueError("Unknown planner setting; store credentials in environment variables only")
    if config.get("api") not in {"responses", "chat_completions"}:
        raise ValueError("api must be responses or chat_completions")
    if not config.get("model") or not config.get("base_url", "").startswith("https://"):
        raise ValueError("A model name and HTTPS base_url including /v1 are required")
    if config.get("vision") is not True:
        raise ValueError("The main visual-agent comparison requires vision=true")
    from urllib.parse import urlsplit
    parsed = urlsplit(config["base_url"])
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Do not put credentials in base_url")
    overrides = json.dumps(config.get("request_overrides", {})).lower()
    if any(word in overrides for word in ("api_key", "authorization", "bearer ")):
        raise ValueError("Do not put credentials in request_overrides")
    return config

