"""Audited removal of empty polygon placeholders, common to all experiment arms.

Original predictions and frozen manifests are never edited. Nonempty invalid
geometry remains an error; this module does not perform polygon repair.
"""
import os
from pathlib import Path

from fieldagent.io import digest, load, now, save

ENV = "FIELDAGENT_SCENE_INPUT_OUTPUT"
_INSTALLED = False


def prepare(entry, out):
    from fieldagent.experiment import check_inputs
    from cvr_agent import read_items, save_items, vertices
    check_inputs(entry)
    raw = read_items(entry["prediction_vector"])
    removed = [p for p in raw if p.geometry.is_empty and p.geometry.geom_type in {"Polygon", "MultiPolygon"}]
    if not removed:
        return entry, None
    if len({p.identifier for p in raw}) != len(raw):
        raise ValueError("Duplicate instance IDs in scene " + entry["scene"])
    folder = Path(out) / "input_preprocessing" / entry["scene"]
    path = folder / "prediction.nonempty.pixel.geojson"
    record_path = folder / "record.json"
    removed_ids = {p.identifier for p in removed}
    filtered = [p for p in raw if p.identifier not in removed_ids]
    evidence = dict(scene=entry["scene"], policy="exclude_empty_polygon_placeholders_all_arms_v1",
        source_prediction=entry["prediction_vector"], source_sha256=entry["prediction_vector_sha256"],
        original_instances=len(raw), usable_instances=len(filtered),
        original_vertices=sum(vertices(p.geometry) for p in raw),
        usable_vertices=sum(vertices(p.geometry) for p in filtered),
        excluded=[dict(instance_id=p.identifier, reason="empty_polygon_no_boundary", vertices=0, area=0.) for p in removed],
        extension_source_sha256=digest(__file__), runtime_GT_access=False)
    if record_path.exists():
        prior = load(record_path)
        if any(prior.get(k) != v for k, v in evidence.items()):
            raise ValueError("Input preprocessing evidence changed: " + entry["scene"])
        if digest(path) != prior["derived_sha256"]:
            raise ValueError("Derived input changed: " + entry["scene"])
    else:
        if path.exists():
            raise ValueError("Derived vector lacks preprocessing record: " + entry["scene"])
        save_items(path, filtered)
        evidence.update(derived_sha256=digest(path), derived_prediction=str(path), created_utc=now())
        save(record_path, evidence)
    return dict(entry, prediction_vector=str(path), prediction_vector_sha256=digest(path)), str(record_path)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import scene_run
    from fieldagent import experiment
    original_read = experiment.read_scene
    original_persist = experiment.persist_method
    original_freeze = scene_run.freeze
    original_import = scene_run.import_controls

    def freeze(inputs, out, *args, **kwargs):
        manifest = original_freeze(inputs, out, *args, **kwargs)
        os.environ[ENV] = str(Path(out).resolve())
        return manifest

    def read(entry):
        if ENV not in os.environ:
            return original_read(entry)
        derived, _ = prepare(entry, os.environ[ENV])
        try:
            return original_read(derived)
        except ValueError as error:
            raise ValueError(f"Scene {entry['scene']}: {error}") from error

    def persist(out, entry, *args, **kwargs):
        result = original_persist(out, entry, *args, **kwargs)
        record = Path(out) / "input_preprocessing" / entry["scene"] / "record.json"
        if record.exists():
            result["input_preprocessing"] = dict(record=str(record), sha256=digest(record),
                policy="exclude_empty_polygon_placeholders_all_arms_v1")
            method = args[0] if args else kwargs["method"]
            save(Path(out) / "scenes" / entry["scene"] / method / "completed.json", result)
        return result

    def import_controls(manifest, out):
        from fieldagent.experiment import completed_method, run_local
        os.environ[ENV] = str(Path(out).resolve())
        affected, regular = [], []
        for entry in manifest["entries"]:
            _, record = prepare(entry, out)
            (affected if record else regular).append(entry)
        # Legacy cached evaluation includes the empty instance. Recompute this
        # scene's controls/evaluation with exactly the same usable raw as GPT.
        count = original_import(dict(manifest, entries=regular), out)
        if affected:
            need = [e for e in affected if not all(completed_method(out, e, m)
                    for m in ("raw", "public_edge", "local_adaptive_strict"))]
            if need:
                print("INPUT PREPROCESSING: recomputing local controls for "
                      + ", ".join(e["scene"] for e in need), flush=True)
                run_local(dict(manifest, entries=need), out)
                count += len(need)
        return count

    experiment.read_scene = read
    experiment.persist_method = persist
    scene_run.freeze = freeze
    scene_run.import_controls = import_controls
    _INSTALLED = True
