"""Prechecked choices plus bounded, explicit coordinate evidence for scene v2.

No candidate is selected or committed by the local preparer. Sampled coordinates
are observation only; direct edits retain original ring indices and geometry.
"""
import base64
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import time

from fieldagent.io import digest, load, save

MAX_POINTS = 32
MAX_CHOICES = 3


def schema():
    import boundary_autonomous as ba
    from boundary_tools import REGISTRY
    commands = []
    for original in ba.COMMANDS:
        spec = deepcopy(original)
        kind = spec["properties"]["kind"]["enum"]
        if kind == ["direct_edit"]:
            spec["properties"]["task"]["enum"] = ["simplify"]
            spec["properties"]["edits"]["items"]["anyOf"] = [
                e for e in spec["properties"]["edits"]["items"]["anyOf"]
                if e["properties"]["kind"]["enum"] in (["retain_vertices"], ["replace_boundary_arc"])]
        elif kind == ["use_tool"]:
            spec["properties"]["tool_id"]["enum"] = [k for k, v in REGISTRY.items() if v.task == "simplify"]
        elif "keep" in kind:
            spec["properties"]["kind"]["enum"] = ["keep"]
        else:
            continue
        commands.append(spec)
    commands.append(ba.obj(dict(kind=dict(type="string", enum=["select_candidate"]),
        **ba.COMMON, candidate_id=ba.STRING)))
    return ba.obj(dict(state_id=ba.STRING, diagnosis=ba.STRING,
        action=dict(type="string", enum=["propose", "commit", "finish"]),
        commands=dict(type="array", items=dict(anyOf=commands)), batch_id=dict(type=["string", "null"]),
        done=dict(type="boolean")))


def pool_path(executor):
    # root may be a transaction/recovery folder. The immutable state namespace
    # prevents candidates for an older committed geometry being selected.
    root = getattr(executor, "compact_pool_root", executor.root)
    return Path(root) / "local_candidates" / executor.state_id / "pool.json"


def candidate_pool(executor):
    import compact_scene as cs
    from cvr_agent import read_items, save_items, vertices
    from boundary_comparison import features
    from boundary_autonomous import shared_boundary_context
    path = pool_path(executor)
    if path.exists():
        pool = load(path)
        if pool["base_state"] != executor.state_id or pool["original_state"] != __import__('cvr_agent').state_key(executor.original):
            raise ValueError("Stale local candidate pool")
        for row in pool["candidates"]:
            ref = row["candidate_file"]
            if digest(ref["path"]) != ref["sha256"]:
                raise ValueError("Local candidate file changed")
        return pool
    started = time.monotonic()
    records, attempts = [], []
    for parcel in executor.current:
        if parcel.identifier not in cs.unresolved(executor):
            continue
        shared = any(r["shared_edges"] for r in shared_boundary_context(parcel, executor.current))
        preferred = "contour_shared_dp" if shared else "contour_dp"
        other = "contour_dp" if shared else "contour_shared_dp"
        specs = [(preferred, t) for t in (2., 1., .5, .25)] + [(other, 1.)]
        passing, seen = [], set()
        for tool, tolerance in specs:
            if time.monotonic() - executor.started >= executor.config.seconds:
                raise TimeoutError("Scene time budget exhausted during local candidate preparation")
            row = dict(target_id=parcel.identifier, tool=tool, parameters={"tolerance": tolerance},
                       base_state=executor.state_id)
            before = time.monotonic()
            with executor.audit.call("compact_local_candidate_precheck", row,
                    ["local_candidate_preparation_not_GPT", "common_typed_verifier"]) as event:
                try:
                    candidate, _ = executor.box.propose(tool, executor.current, [parcel.identifier], row["parameters"])
                    verdict = executor.verify(candidate, [parcel.identifier], "simplify")
                    target = next(p for p in candidate if p.identifier == parcel.identifier)
                    row["verdict"] = verdict
                    if verdict["accept"] and vertices(target.geometry) < vertices(parcel.geometry):
                        key = target.geometry.normalize().wkb_hex
                        row["duplicate"] = key in seen
                        if key not in seen:
                            seen.add(key)
                            ref = save_items(path.parent / f"candidate-{len(attempts):05d}.geojson", [target])
                            row["candidate_file"] = ref
                            passing.append(dict(candidate_id=f"c{len(attempts):05d}", target_id=parcel.identifier,
                                base_state=executor.state_id, candidate_file=ref, verdict=verdict,
                                chain=[dict(tool=tool, parameters=row["parameters"])],
                                features=features(executor.box, parcel, target)))
                except Exception as error:
                    row["verdict"] = dict(accept=False, reasons=["local_candidate_error"], error=type(error).__name__)
                row["seconds"] = time.monotonic() - before
                event.update(verdict=row["verdict"])
            attempts.append(row)
        # Offer a small range of compression levels rather than auto-selecting.
        passing.sort(key=lambda r: (r["features"]["vertices"], r["features"]["shift"], r["candidate_id"]))
        selected = sorted({0, len(passing) // 2, len(passing) - 1}) if passing else []
        records.extend(passing[i] for i in selected[:MAX_CHOICES])
    joint_options, seen_options = [], set()
    for name, end in (("fewest_points", 0), ("more_detail", -1)):
        mapping = {}
        selected_ids = []
        for parcel in executor.current:
            choices = [r for r in records if r["target_id"] == parcel.identifier]
            if choices:
                chosen = choices[end]
                mapping[parcel.identifier] = read_items(chosen["candidate_file"]["path"])[0]
                selected_ids.append(chosen["candidate_id"])
        fingerprint = tuple(selected_ids)
        if not mapping or fingerprint in seen_options:
            continue
        seen_options.add(fingerprint)
        candidate = [mapping.get(p.identifier, p) for p in executor.current]
        with executor.audit.call("compact_joint_option_precheck", dict(name=name, candidate_ids=selected_ids),
                ["preparation_only_not_automatic_selection", "common_joint_verifier"]) as event:
            verdict = executor.verify(candidate, list(mapping), "simplify")
            event.update(verdict=verdict)
        joint_options.append(dict(name=name, candidate_ids=selected_ids, target_ids=list(mapping),
            verdict={k: verdict[k] for k in ("accept", "reasons")},
            total_vertices=sum(vertices(p.geometry) for p in candidate)))
    pool = dict(base_state=executor.state_id, original_state=__import__('cvr_agent').state_key(executor.original),
        candidates=records, attempts=attempts, seconds=time.monotonic() - started,
        joint_options=joint_options, automatic_selection=False, runtime_gt_access=False)
    save(path, pool)
    return pool


def indexed_points(parcel, pool):
    """At most 32 shown original indices per ring, including structural bends.

    This does not simplify the input. A model may select only the indices it has
    actually observed. Other local tools remain available for unsampled arcs.
    """
    import numpy as np
    from shapely.geometry import LineString
    from fieldbqc_direct_edit import ring_records
    from cvr_agent import read_items
    suggestions = []
    for row in pool["candidates"]:
        if row["target_id"] == parcel.identifier:
            target = read_items(row["candidate_file"]["path"])[0]
            suggestions.extend(tuple(p) for r in ring_records(target.geometry) for p in r["coordinates"])
    result = []
    for ring in ring_records(parcel.geometry):
        coords = ring["coordinates"]
        lookup = {tuple(p): i for i, p in enumerate(coords)}
        indices = {lookup[p] for p in suggestions if p in lookup}
        # DP anchors are observation sampling only, never replacement geometry.
        simplified = LineString(coords + coords[:1]).simplify(.5, preserve_topology=False)
        anchors = [lookup[tuple(p)] for p in simplified.coords if tuple(p) in lookup]
        indices.update(anchors)
        n = len(coords)
        if len(indices) > MAX_POINTS:
            chosen = sorted(indices)
            indices = {chosen[int(i)] for i in np.linspace(0, len(chosen) - 1, MAX_POINTS)}
        spare = MAX_POINTS - len(indices)
        if spare > 0:
            indices.update(int(i) for i in np.linspace(0, n - 1, min(spare, n)))
        indices = sorted(indices)[:MAX_POINTS]
        result.append(dict(ring_id=ring["ring_id"], original_vertex_count=n,
            indexed_points=[[i, *coords[i]] for i in indices],
            sampling="Original indices; omitted points still exist and are checked by the verifier"))
    return result


def build_context(executor, evidence_ids, images=None):
    import compact_scene as cs
    from cvr_agent import vertices
    from boundary_tools import catalog, REGISTRY
    from boundary_autonomous import shared_boundary_context
    from fieldbqc_direct_edit import ring_records
    pool = candidate_pool(executor)
    unresolved = cs.unresolved(executor)
    pending = next((b for b in reversed(executor.batches) if b["status"] == "previewed"
                    and b["base_state"] == executor.state_id), None)
    pending_ids = set(pending["target_ids"]) if pending and pending["verdict"]["accept"] else set()
    detailed = [uid for uid in unresolved if uid not in pending_ids]
    instances = []
    for parcel in executor.current:
        if parcel.identifier not in unresolved:
            continue
        row = dict(target_id=parcel.identifier, vertices=vertices(parcel.geometry),
                   bounds=list(parcel.geometry.bounds), in_passing_pending_batch=parcel.identifier in pending_ids)
        if parcel.identifier in detailed:
            row.update(rings=indexed_points(parcel, pool), shared_boundary_protection=[
                dict(ring_id=r["ring_id"], protected_endpoint_indices=r["protected_endpoint_indices"])
                for r in shared_boundary_context(parcel, executor.current)])
        instances.append(row)
    choices = []
    for row in pool["candidates"]:
        if row["target_id"] not in detailed:
            continue
        target = __import__('cvr_agent').read_items(row["candidate_file"]["path"])[0]
        choices.append(dict(candidate_id=row["candidate_id"], target_id=row["target_id"],
            chain=row["chain"], features={k: round(row["features"][k], 4)
                for k in ("vertices", "reduction", "shift", "area_change", "support_gain")}, individually_passed=True,
            # Only a truly small candidate's full coordinates are sent.
            rings=ring_records(target.geometry) if vertices(target.geometry) <= 24 else [],
            coordinate_detail="all" if vertices(target.geometry) <= 24 else "summary_only_use_preview"))
    # Keep only the latest verdict per unresolved target, not all commands,
    # failed geometries, candidate files, scores or growing history.
    latest = {}
    for p in executor.pending_proposals.values():
        if p["target_id"] in unresolved:
            latest[p["target_id"]] = dict(target_id=p["target_id"], source=p["source"],
                accept=p["verdict"]["accept"], reasons=p["verdict"]["reasons"],
                vertices=p["features"]["vertices"] if p["features"] else None)
    return dict(version=cs.VERSION, protocol="compact_whole_image_v2", state_id=executor.state_id,
        raw_input_state=__import__('cvr_agent').state_key(executor.original),
        goal_mode="minimize_vertices", image_shape=list(executor.box.shape), runtime_gt_access=False,
        coordinate_system="Original-image pixel coordinates; [index,x,y] uses original zero-based ring indices",
        allowed_target_ids=unresolved, unresolved_target_ids=unresolved,
        detailed_target_ids=detailed, allow_direct=True, direct_only=False,
        remaining_image_calls=executor.max_calls - executor.model_calls + 1,
        instances=instances, pool_candidates=choices,
        scene_inventory=[dict(target_id=p.identifier, vertices=vertices(p.geometry), bounds=list(p.geometry.bounds),
            resolved=p.identifier in executor.resolved_ids) for p in executor.current],
        batches=[dict(batch_id=pending["batch_id"], base_state=pending["base_state"], status=pending["status"],
            target_ids=pending["target_ids"], verdict={k: pending["verdict"][k] for k in ("accept", "reasons")})] if pending else [],
        prechecked_joint_options=pool.get("joint_options", []) if detailed else [],
        proposals=[], latest_feedback=list(latest.values()), evidence_ids=evidence_ids,
        images=[{k: v for k, v in row.items() if k not in {"path", "sha256", "atlas_tiles"}} for row in images or []],
        available_tools={k: dict(parameters=v["parameters"], description=v["description"])
                         for k, v in catalog().items() if REGISTRY[k].task == "simplify"},
        limits=asdict(executor.config),
        local_preparation=dict(attempts=len(pool["attempts"]), seconds=pool["seconds"], automatic_selection=False),
        direct_geometry_policy="Only use shown original indices/coordinates. Unsampled detail is not implicitly simplified. No movement/holes/overlap tools.")


def validate_packet(packet, context):
    import boundary_autonomous as ba
    if not isinstance(packet, dict) or set(packet) != set(schema()["required"]):
        raise ValueError("Invalid compact response fields")
    if packet["state_id"] != context["state_id"] or not isinstance(packet["diagnosis"], str) or not packet["diagnosis"].strip() or type(packet["done"]) is not bool:
        raise ValueError("Stale state or missing diagnosis/done")
    commands, action = packet["commands"], packet["action"]
    if not isinstance(commands, list):
        raise ValueError("Commands must be a list")
    ids = [c.get("target_id") for c in commands if isinstance(c, dict)]
    if len(ids) != len(commands) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate or malformed target command")
    if action == "commit":
        batch = next((b for b in context["batches"] if b["batch_id"] == packet["batch_id"]), None)
        if not batch or batch["status"] != "previewed" or batch["base_state"] != context["state_id"] or not batch["verdict"]["accept"]:
            raise ValueError("Cannot commit stale, unobserved or failed joint batch")
        remaining = set(context["unresolved_target_ids"]) - set(batch["target_ids"])
        if not set(ids) <= remaining or any(c["kind"] != "keep" for c in commands):
            raise ValueError("Commit commands may explicitly keep only targets outside the passing batch")
        if packet["done"] and set(ids) != remaining:
            raise ValueError("Finishing commit must explicitly decide all remaining targets")
    elif action in {"propose", "finish"}:
        if packet["batch_id"] is not None or packet["done"] != (action == "finish") or set(ids) != set(context["unresolved_target_ids"]):
            raise ValueError("Explicit command required for every unresolved target")
        if action == "finish" and any(c["kind"] != "keep" for c in commands):
            raise ValueError("Finish must explicitly keep every unresolved target")
    else:
        raise ValueError("Unknown action")
    choices = {r["candidate_id"]: r for r in context["pool_candidates"]}
    instances = {r["target_id"]: r for r in context["instances"]}
    for command in commands:
        mapped = dict(command)
        kind = command.get("kind")
        if kind == "select_candidate":
            row = choices.get(command.get("candidate_id"))
            if not row or row["target_id"] != command.get("target_id"):
                raise ValueError("Unknown, stale or unobserved candidate")
            if set(command) != {"kind", "target_id", "reason", "evidence_refs", "candidate_id"}:
                raise ValueError("Incorrect candidate selection fields")
            # Common target/reason/evidence validation through an existing keep
            # command avoids reliance on legacy globally restricted pool schema.
            mapped = {k: v for k, v in command.items() if k != "candidate_id"}
            mapped["kind"] = "keep"
        elif kind not in {"direct_edit", "use_tool", "keep"}:
            raise ValueError("Unknown compact command")
        ba.validate_packet(dict(state_id=packet["state_id"], diagnosis=packet["diagnosis"], commands=[mapped], done=False), context)
        if kind == "direct_edit":
            if command["task"] != "simplify" or command["target_id"] not in context["detailed_target_ids"]:
                raise ValueError("Direct editing requires current detailed evidence")
            rings = {r["ring_id"]: r for r in instances[command["target_id"]].get("rings", [])}
            for edit in command["edits"]:
                ring = rings.get(edit.get("ring_id"))
                if not ring:
                    raise ValueError("Unknown observed ring")
                shown = {p[0]: p[1:] for p in ring["indexed_points"]}
                if edit["kind"] == "retain_vertices":
                    if not set(edit["keep_indices"]) <= set(shown):
                        raise ValueError("Cannot use unobserved vertex indices")
                elif edit["kind"] == "replace_boundary_arc":
                    if edit.get("start_index") not in shown or edit.get("end_index") not in shown:
                        raise ValueError("Arc endpoints must be observed original indices")
                else:
                    raise ValueError("Only literal vertex reduction edits are allowed")
        elif kind == "use_tool":
            from boundary_tools import REGISTRY
            if REGISTRY[command["tool_id"]].task != "simplify":
                raise ValueError("Only simplification tools are allowed")
    return packet


def image_bundle(executor, folder):
    import cv2
    import numpy as np
    from boundary_comparison import overlay
    from cvr_agent import read_items
    folder.mkdir(parents=True, exist_ok=True)
    pool = candidate_pool(executor)
    pending = next((b for b in reversed(executor.batches) if b["status"] == "previewed"
                    and b["base_state"] == executor.state_id), None)
    already_previewed = set(pending["target_ids"]) if pending and pending["verdict"]["accept"] else set()
    frames = [("raw_scene", overlay(executor.image, [], title="Raw image", side=512)),
              ("current_scene_all_parcels", overlay(executor.image, executor.current, title="All parcel IDs", side=512))]
    if already_previewed:
        # Requests are stateless: retain the full-image pixels in the joint
        # overlay below, but avoid resending two additional identical views.
        frames = []
    if pending:
        ref = pending["candidate_file"]
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Joint preview changed")
        frames.append((pending["batch_id"] + " joint preview accept=" + str(pending["verdict"]["accept"]),
                       overlay(executor.image, read_items(ref["path"]), title="Joint preview " + pending["batch_id"], side=512)))
    tiles, details = [], []
    for parcel in executor.current:
        if parcel.identifier in executor.resolved_ids or parcel.identifier in already_previewed:
            continue
        x0, y0, x1, y1 = parcel.geometry.bounds
        bounds = [max(0, x0 - 6), max(0, y0 - 6), min(executor.box.shape[1], x1 + 6), min(executor.box.shape[0], y1 + 6)]
        bx0, by0, bx1, by1 = int(bounds[0]), int(bounds[1]), math.ceil(bounds[2]), math.ceil(bounds[3])
        scale = min(192 / max(bx1 - bx0, 1), 192 / max(by1 - by0, 1))
        tile = overlay(executor.image, [parcel], bounds, "ID " + parcel.identifier, side=192)
        for ring in indexed_points(parcel, pool):
            shown = ring["indexed_points"]
            for index, x, y in shown[::max(1, math.ceil(len(shown) / 16))]:
                point = (round((x - bx0) * scale), 32 + round((y - by0) * scale))
                cv2.circle(tile, point, 1, (0, 230, 255), -1)
                cv2.putText(tile, str(index), point, cv2.FONT_HERSHEY_SIMPLEX, .26, (0, 230, 255), 1)
        tiles.append(tile)
        details.append(parcel.identifier)
    for start in range(0, len(tiles), 16):
        group = tiles[start:start + 16]
        atlas = np.full((math.ceil(len(group) / 4) * 224, 768, 3), 245, np.uint8)
        for index, tile in enumerate(group):
            row, col = divmod(index, 4)
            atlas[row * 224:(row + 1) * 224, col * 192:(col + 1) * 192] = tile
        frames.append(("unresolved_indices_" + ",".join(details[start:start + 16]), atlas))
    urls, records = [], []
    for index, (name, frame) in enumerate(frames):
        eid = f"image_{index:02d}"
        cv2.putText(frame, eid, (max(1, frame.shape[1] - 95), 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            raise ValueError("Image encoding failed")
        path = folder / (eid + ".png")
        path.write_bytes(png.tobytes())
        urls.append("data:image/png;base64," + base64.b64encode(png).decode("ascii"))
        records.append(dict(evidence_id=eid, name=name, path=str(path), sha256=digest(path)))
    save(folder / "images.json", records)
    return urls, records
