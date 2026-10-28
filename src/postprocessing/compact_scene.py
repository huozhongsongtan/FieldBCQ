"""Whole-image GPT proposals and atomic joint commits, with durable call state.

No labels enter this module. Every target is offered in each whole-image
observation. Existing per-target and combined guards remain authoritative.
"""
import base64
from copy import copy, deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from fieldagent import activate_engine
from fieldagent.io import digest, load, save

PROMPT_PATH = Path(__file__).with_name("COMPACT_SCENE_PROMPT.md")
VERSION = "compact-scene-vertex-v2"


def schema():
    import boundary_autonomous as ba
    from boundary_tools import REGISTRY
    commands = []
    for original in ba.COMMANDS:
        spec = deepcopy(original)
        kinds = spec["properties"]["kind"]["enum"]
        if "direct_edit" in kinds:
            spec["properties"]["task"]["enum"] = ["simplify"]
            spec["properties"]["edits"]["items"]["anyOf"] = [
                e for e in spec["properties"]["edits"]["items"]["anyOf"]
                if e["properties"]["kind"]["enum"] in (["retain_vertices"], ["replace_boundary_arc"])]
        elif "use_tool" in kinds:
            spec["properties"]["tool_id"]["enum"] = [k for k, v in REGISTRY.items() if v.task == "simplify"]
        elif "keep" in kinds:
            spec["properties"]["kind"]["enum"] = ["keep"]
        else:
            continue
        commands.append(spec)
    return ba.obj(dict(state_id=ba.STRING, diagnosis=ba.STRING,
        action=dict(type="string", enum=["propose", "commit", "finish"]),
        commands=dict(type="array", items=dict(anyOf=commands)), batch_id=dict(type=["string", "null"]),
        done=dict(type="boolean")))


def unresolved(executor):
    return [p.identifier for p in executor.current if p.identifier not in executor.resolved_ids]


def validate_packet(packet, context):
    import boundary_autonomous as ba
    from boundary_tools import REGISTRY
    if not isinstance(packet, dict) or set(packet) != set(schema()["required"]):
        raise ValueError("Invalid whole-image response fields")
    if packet["state_id"] != context["state_id"]:
        raise ValueError("Stale whole-image state")
    if not isinstance(packet["diagnosis"], str) or not packet["diagnosis"].strip() or type(packet["done"]) is not bool:
        raise ValueError("Missing diagnosis or invalid done")
    commands, action = packet["commands"], packet["action"]
    if not isinstance(commands, list):
        raise ValueError("Commands must be an array")
    if action == "commit":
        batches = {b["batch_id"]: b for b in context["batches"]}
        batch = batches.get(packet["batch_id"])
        if commands or not batch or batch["status"] != "previewed" or batch["base_state"] != context["state_id"]:
            raise ValueError("Unknown, stale or unobserved batch")
        if not batch["verdict"]["accept"]:
            raise ValueError("Combined batch did not pass")
        if packet["done"] and set(context["unresolved_target_ids"]) - set(batch["target_ids"]):
            raise ValueError("Cannot finish while targets remain unresolved")
        return packet
    if action not in {"propose", "finish"} or packet["batch_id"] is not None:
        raise ValueError("Invalid proposal/finish action")
    if packet["done"] != (action == "finish"):
        raise ValueError("A proposal must be observed in a later response")
    seen = []
    for command in commands:
        ba.validate_packet(dict(state_id=packet["state_id"], diagnosis=packet["diagnosis"],
            commands=[command], done=False), context)
        kind = command["kind"]
        if kind not in {"direct_edit", "use_tool", "keep"} or (action == "finish" and kind != "keep"):
            raise ValueError("Only literal simplification, simplifier or explicit keep allowed")
        if kind == "direct_edit" and (command["task"] != "simplify" or any(
                e.get("kind") not in {"retain_vertices", "replace_boundary_arc"} for e in command["edits"])):
            raise ValueError("Only vertex reduction edits are allowed")
        if kind == "use_tool" and REGISTRY[command["tool_id"]].task != "simplify":
            raise ValueError("Only simplification tools are allowed")
        seen.append(command["target_id"])
    if len(seen) != len(set(seen)) or set(seen) != set(context["unresolved_target_ids"]):
        raise ValueError("One explicit command per unresolved target is required")
    return packet


def image_bundle(executor, folder):
    """Two scene images plus compact atlases; never two API calls per parcel."""
    import cv2
    import numpy as np
    from boundary_comparison import overlay
    from fieldbqc_direct_edit import ring_records
    from cvr_agent import read_items
    frames = [("raw_scene", overlay(executor.image, [], title="Raw image", side=512), None),
              ("current_scene", overlay(executor.image, executor.current, title="Current parcel IDs", side=512), None)]
    tiles, tile_records = [], []
    for parcel in executor.current:
        x0, y0, x1, y1 = parcel.geometry.bounds
        bounds = [max(0, x0 - 6), max(0, y0 - 6), min(executor.box.shape[1], x1 + 6), min(executor.box.shape[0], y1 + 6)]
        bx0, by0, bx1, by1 = int(bounds[0]), int(bounds[1]), math.ceil(bounds[2]), math.ceil(bounds[3])
        side = 192
        scale = min(side / max(bx1 - bx0, 1), side / max(by1 - by0, 1))
        tile = overlay(executor.image, [parcel], bounds, "ID " + parcel.identifier, side=side)
        for ring in ring_records(parcel.geometry):
            coords = ring["coordinates"]
            for index in range(0, len(coords), max(1, math.ceil(len(coords) / 12))):
                x, y = coords[index]
                point = (round((x - bx0) * scale), 32 + round((y - by0) * scale))
                cv2.circle(tile, point, 1, (0, 230, 255), -1)
                cv2.putText(tile, str(index), (point[0] + 1, point[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, .26, (0, 230, 255), 1)
        tiles.append(tile)
        tile_records.append({"target_id": parcel.identifier, "original_bounds": [bx0, by0, bx1, by1],
                             "scale": scale, "tile_image_offset_y": 32})
    for start in range(0, len(tiles), 16):
        group = tiles[start:start + 16]
        atlas = np.full((math.ceil(len(group) / 4) * 224, 4 * 192, 3), 245, np.uint8)
        mapping = []
        for index, tile in enumerate(group):
            row, col = divmod(index, 4)
            atlas[row * 224:(row + 1) * 224, col * 192:(col + 1) * 192] = tile
            mapping.append({**tile_records[start + index], "atlas_offset_xy": [col * 192, row * 224]})
        frames.append(("vertex_atlas_" + str(start // 16), atlas, mapping))
    for batch in executor.batches[-1:]:
        if batch["status"] == "previewed" and batch.get("candidate_file"):
            ref = batch["candidate_file"]
            if digest(ref["path"]) != ref["sha256"]:
                raise ValueError("Combined preview changed")
            frames.append((batch["batch_id"] + " combined preview accept=" + str(batch["verdict"]["accept"]),
                overlay(executor.image, read_items(ref["path"]), title=batch["batch_id"] + " combined preview", side=512), None))
    folder.mkdir(parents=True, exist_ok=True)
    urls, images = [], []
    for index, (name, frame, mapping) in enumerate(frames):
        eid = f"image_{index:02d}"
        cv2.putText(frame, eid, (max(2, frame.shape[1] - 95), 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
        ok, blob = cv2.imencode(".png", frame)
        if not ok:
            raise ValueError("Image encoding failed")
        path = folder / (eid + ".png")
        path.write_bytes(blob.tobytes())
        urls.append("data:image/png;base64," + base64.b64encode(blob).decode("ascii"))
        images.append({"evidence_id": eid, "name": name, "path": str(path), "sha256": digest(path), "atlas_tiles": mapping})
    save(folder / "images.json", images)
    return urls, images


def build_context(executor, evidence_ids, images=None):
    import boundary_autonomous as ba
    from boundary_tools import REGISTRY
    ids = [p.identifier for p in executor.current]
    context = ba.build_context(executor, None, ids, True, evidence_ids)
    context.update(version=VERSION, protocol="Whole-image proposals; combined validation; later atomic batch commit",
        unresolved_target_ids=unresolved(executor), remaining_image_calls=executor.max_calls - executor.model_calls + 1,
        available_tools={k: v for k, v in context["available_tools"].items() if REGISTRY[k].task == "simplify"},
        batches=[{k: v for k, v in b.items() if k != "candidate_file"} for b in executor.batches[-2:]],
        proposals=[{k: p[k] for k in ("proposal_id", "target_id", "source", "command", "verdict", "features", "status")}
                   for p in executor.pending_proposals.values() if p["base_state"] == executor.state_id],
        images=images or [], research_objective="minimum necessary boundary vertices only")
    return context


def request_packet(executor, context, urls, settings, folder):
    from fieldbqc_planner import APIPlanner
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    payload = APIPlanner(settings, schema=schema(), system_prompt=prompt)._payload(context, urls)
    save(folder / "context.json", context)
    save(folder / "request.json", payload)
    (folder / "system_prompt.md").write_text(prompt, encoding="utf-8")
    endpoint = settings["base_url"].rstrip("/") + ("/responses" if settings["api"] == "responses" else "/chat/completions")
    meta = {"requested_model": settings["model"], "endpoint": endpoint,
            "request_sha256": digest(folder / "request.json"), "retries": 0, "protocol": VERSION}
    started = time.monotonic()
    try:
        with executor.audit.call("whole_scene_GPT_request", meta, ["HTTP POST", "joint_visual_geometry_decision"]) as event:
            key = os.environ.get(settings.get("api_key_env", "OPENAI_API_KEY"))
            if not key:
                raise RuntimeError("Configured API key unavailable")
            request = urllib.request.Request(endpoint, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
            remaining = executor.config.seconds - (time.monotonic() - executor.started)
            if remaining <= 0:
                raise TimeoutError("Scene time budget exhausted")
            with urllib.request.urlopen(request, timeout=min(settings.get("timeout_seconds", 90), remaining)) as response:
                data = json.load(response)
            save(folder / "response.json", data)
            if settings["api"] == "responses":
                raw = "".join(c.get("text", "") for m in data.get("output", []) if m.get("type") == "message"
                    for c in m.get("content", []) if c.get("type") == "output_text")
                complete = data.get("status") == "completed"
            else:
                choice = data["choices"][0]
                raw, complete = choice["message"]["content"], choice.get("finish_reason") == "stop"
            meta.update(seconds=time.monotonic() - started, usage=data.get("usage", {}), response_id=data.get("id"),
                        response_model=data.get("model"), completed=complete)
            if not complete:
                raise ValueError("Incomplete whole-image model response")
            packet = validate_packet(json.loads(raw), context)
            save(folder / "decision.json", packet)
            event.update(decision=packet)
        save(folder / "metadata.json", meta)
        return packet, meta
    except Exception as error:
        meta.update(seconds=time.monotonic() - started,
            error=f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError) else type(error).__name__ + ": " + str(error))
        save(folder / "error.json", meta)
        raise RuntimeError(meta["error"]) from None


def initialize_executor(image, raw, run, config, max_calls):
    import boundary_autonomous as ba
    executor = ba.AutonomousExecutor(image, raw, run, config)
    executor.goal_mode, executor.direct_only = "minimize_vertices", False
    executor.model_calls, executor.max_calls = 0, max_calls
    executor.batches, executor.resolved_ids, executor.model_records = [], set(), []
    executor.phase, executor.gpt_error, executor.ready_packet = "ready", None, None
    executor.decisions = []
    return executor


def checkpoint(executor):
    from cvr_agent import save_items, state_key
    vector = executor.root / "state" / f"call-{executor.model_calls:03d}-{executor.phase}.pixel.geojson"
    ref = save_items(vector, executor.current)
    record = {"version": VERSION, "config": asdict(executor.config), "original_state": state_key(executor.original),
        "current_state": executor.state_id, "current_vector": ref, "max_calls": executor.max_calls,
        "model_calls": executor.model_calls, "phase": executor.phase, "ready_packet": executor.ready_packet,
        "gpt_error": executor.gpt_error, "history": executor.history, "proposals": list(executor.pending_proposals.values()),
        "proposal_counter": executor.proposal_counter, "preview_attempts": executor.preview_attempts,
        "batches": executor.batches, "resolved_ids": sorted(executor.resolved_ids), "decisions": executor.decisions,
        "model_records": executor.model_records, "seconds": time.monotonic() - executor.started}
    save(executor.root / "scene_checkpoint.json", record)


def restore_executor(image, raw, run, config, max_calls):
    from cvr_agent import Audit, read_items, state_key
    record = load(run / "scene_checkpoint.json")
    if record["version"] != VERSION or record["config"] != asdict(config) or record["max_calls"] != max_calls:
        raise ValueError("Whole-image protocol changed")
    if type(record["model_calls"]) is not int or not 0 <= record["model_calls"] <= max_calls:
        raise ValueError("Invalid saved image attempt budget")
    if record["phase"] not in {"ready", "inflight", "response_ready", "external_error"}:
        raise ValueError("Unknown saved execution phase")
    if not set(record["resolved_ids"]) <= {p.identifier for p in raw}:
        raise ValueError("Unknown resolved target")
    if record["original_state"] != state_key(raw):
        raise ValueError("Raw anchor changed")
    ref = record["current_vector"]
    if digest(ref["path"]) != ref["sha256"]:
        raise ValueError("Saved committed state changed")
    executor = initialize_executor(image, raw, run / "recovery_initialization" / str(time.time_ns()), config, max_calls)
    executor.root, executor.audit = run, Audit(run)
    executor.current = read_items(ref["path"])
    if executor.state_id != record["current_state"]:
        raise ValueError("Saved state fingerprint changed")
    for key in ("model_calls", "phase", "ready_packet", "gpt_error", "history", "proposal_counter", "preview_attempts",
                "batches", "decisions", "model_records"):
        setattr(executor, key, record[key])
    executor.resolved_ids = set(record["resolved_ids"])
    executor.pending_proposals = {p["proposal_id"]: p for p in record["proposals"]}
    for proposal in record["proposals"]:
        ref = proposal.get("candidate_file")
        if ref:
            if digest(ref["path"]) != ref["sha256"]:
                raise ValueError("Stored proposal changed")
            target = next(p for p in read_items(ref["path"]) if p.identifier == proposal["target_id"])
            executor.tried.add((proposal["base_state"], proposal["target_id"], proposal["task"], target.geometry.normalize().wkb_hex))
    for batch in executor.batches:
        ref = batch.get("candidate_file")
        if ref and digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Stored combined candidate changed")
    executor.started = time.monotonic() - record["seconds"]
    audit = run / "calls.jsonl"
    if audit.exists():
        for line in audit.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            executor.audit.counter = max(executor.audit.counter, int(row["call_id"][1:]))
    return executor


def apply_packet(executor, packet):
    """Apply into a new transaction directory; promote only through checkpoint.

    A crash during execution can replay the already-saved response locally into
    a new transaction without another model call or overwriting prior evidence.
    """
    from cvr_agent import Audit, read_items, save_items, scene_metrics
    tx = executor.root / "transactions" / f"call-{executor.model_calls:03d}-{time.time_ns()}"
    tx.mkdir(parents=True)
    work = copy(executor)
    work.root, work.audit = tx, Audit(tx)
    for name in ("current", "history", "pending_proposals", "batches", "resolved_ids", "tried", "decisions"):
        setattr(work, name, deepcopy(getattr(executor, name)))
    base = work.state_id
    action = packet["action"]
    if action in {"propose", "finish"}:
        for batch in work.batches:
            if batch["status"] == "previewed":
                batch["status"] = "superseded"
        selected = []
        for command in packet["commands"]:
            if command["kind"] == "keep":
                work.resolved_ids.add(command["target_id"])
                with work.audit.call("explicit_model_keep", command, ["model_visual_judgment"]):
                    pass
            else:
                if command["kind"] == "select_candidate":
                    from compact_observation import candidate_pool
                    pool = candidate_pool(executor)
                    mapped = dict(command, kind="use_pool")
                    selected.append(work.propose(mapped, expected_state=base, pool=pool))
                else:
                    selected.append(work.propose(command, expected_state=base))
        if action == "propose" and selected:
            replacement = {}
            for proposal in selected:
                if proposal["verdict"]["accept"]:
                    ref = proposal["candidate_file"]
                    if digest(ref["path"]) != ref["sha256"]:
                        raise ValueError("Individual candidate changed")
                    target = next(p for p in read_items(ref["path"]) if p.identifier == proposal["target_id"])
                    replacement[target.identifier] = target
            candidate = [replacement.get(p.identifier, p) for p in work.current]
            ids = list(replacement)
            with work.audit.call("joint_scene_verification", {"target_ids": ids}, ["common_scene_safety", "cumulative_origin_guard", "combined_neighbor_check"]) as event:
                verdict = work.verify(candidate, ids, "simplify") if ids else {"accept": False, "reasons": ["no_individual_candidate_passed"], "metrics": {}}
                event.update(verdict=verdict)
            batch_id = f"b{len(work.batches) + 1:05d}"
            batch = {"batch_id": batch_id, "base_state": base, "target_ids": ids,
                "proposal_ids": [p["proposal_id"] for p in selected], "status": "previewed", "verdict": verdict,
                "candidate_file": save_items(tx / "joint" / (batch_id + ".pixel.geojson"), candidate),
                "created_by_call": executor.model_calls}
            work.batches.append(batch)
    else:
        batch = next(b for b in work.batches if b["batch_id"] == packet["batch_id"])
        if batch["base_state"] != base or batch["status"] != "previewed" or not batch["verdict"]["accept"]:
            raise ValueError("Stale or failed atomic batch")
        ref = batch["candidate_file"]
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Combined candidate changed before commit")
        candidate = read_items(ref["path"])
        with work.audit.call("atomic_scene_commit", {"batch_id": batch["batch_id"], "target_ids": batch["target_ids"]},
                             ["explicit_observed_feedback_commit", "joint_commit_reverification"]) as event:
            verdict = work.verify(candidate, batch["target_ids"], "simplify")
            if not verdict["accept"]:
                batch["status"] = "commit_rejected"
            else:
                work.current = candidate
                work.resolved_ids.update(batch["target_ids"])
                for command in packet["commands"]:
                    work.resolved_ids.add(command["target_id"])
                    with work.audit.call("explicit_model_keep_with_joint_commit", command, ["model_visual_judgment"]):
                        pass
                batch["status"] = "committed"
                for pid in batch["proposal_ids"]:
                    proposal = work.pending_proposals[pid]
                    if proposal["target_id"] in batch["target_ids"]:
                        proposal["status"] = "committed"
            work.history.append({"trial_id": f"t{len(work.history):05d}", "batch_id": batch["batch_id"],
                "base_state": base, "output_state": work.state_id, "target_ids": batch["target_ids"],
                "committed": verdict["accept"], "verdict": verdict, "source": "GPT_whole_scene",
                "candidate_file": ref, "reason": packet["diagnosis"]})
            event.update(committed=verdict["accept"], verdict=verdict)
    work.decisions.append({"call": executor.model_calls, "packet": packet, "state_before": base,
                           "state_after": work.state_id, "transaction": str(tx)})
    save(tx / "applied.json", {"packet": packet, "state_after": work.state_id, "metrics": scene_metrics(work.current)})
    for key in ("current", "history", "pending_proposals", "proposal_counter", "preview_attempts", "batches", "resolved_ids", "tried", "decisions"):
        setattr(executor, key, getattr(work, key))
    executor.phase, executor.ready_packet = "ready", None
    checkpoint(executor)
    with executor.audit.call("whole_scene_transaction_promoted", {"transaction": str(tx), "state_id": executor.state_id}, ["durable_atomic_checkpoint"]):
        pass


def export(executor, stop):
    trace = executor.finish(stop)
    direct = [p for p in executor.pending_proposals.values() if p["source"] == "direct_edit"]
    trace.update(version=VERSION, protocol="compact_whole_image_v2", goal_mode="minimize_vertices",
        model_calls=executor.model_calls, max_calls_per_scene=executor.max_calls, model_records=executor.model_records,
        gpt_error=executor.gpt_error, target_ids=[p.identifier for p in executor.current],
        processed_targets=len(executor.resolved_ids), unfinished_target_ids=unresolved(executor),
        commits=sum(len(h["target_ids"]) for h in executor.history if h["committed"]),
        batch_commits=sum(h["committed"] for h in executor.history), direct_proposals=len(direct),
        direct_committed=sum(p["status"] == "committed" for p in direct), decisions=executor.decisions,
        batches=executor.batches, proposals=list(executor.pending_proposals.values()),
        preview_attempts=executor.preview_attempts, runtime_gt_access=False,
        limitations=["Fixed common gates; fewer points do not guarantee improved GT boundary accuracy",
                     "Whole-image call budget differs from the prior per-parcel protocol; compare separately"])
    pools = [load(p) for p in (executor.root / "local_candidates").glob("*/pool.json")]
    trace.update(local_candidate_attempts=sum(len(p["attempts"]) for p in pools),
        local_preparation_seconds=sum(p["seconds"] for p in pools),
        selected_local_candidates=sum(p["source"] == "use_pool" for p in executor.pending_proposals.values()),
        selected_local_candidates_committed=sum(p["source"] == "use_pool" and p["status"] == "committed"
            for p in executor.pending_proposals.values()),
        automatic_candidate_selection=False)
    save(executor.root / "trace.json", trace)
    return trace


def run_scene(image, raw, run, config, settings, *, max_calls=4, resume_after_error=False, planner=None):
    activate_engine()
    run = Path(run)
    if max_calls < 2:
        raise ValueError("Reserve at least a proposal and a later commit call")
    if not raw:
        return run_empty_scene(image, run, settings, max_calls, resume_after_error)
    if (run / "scene_checkpoint.json").exists():
        executor = restore_executor(image, raw, run, config, max_calls)
    else:
        if run.exists() and any(run.iterdir()):
            raise RuntimeError("Nonempty output lacks whole-image checkpoint; refusing paid replay")
        executor = initialize_executor(image, raw, run, config, max_calls)
        checkpoint(executor)
    if executor.phase in {"inflight", "external_error"}:
        if not resume_after_error:
            raise RuntimeError("Explicit --resume-after-error required for the charged interrupted request")
        folder = run / "gpt_calls" / f"{executor.model_calls:03d}"
        if executor.phase == "inflight" and (folder / "decision.json").exists() and (folder / "metadata.json").exists():
            validate_packet(load(folder / "decision.json"), load(folder / "context.json"))
            executor.ready_packet = load(folder / "decision.json")
            executor.model_records.append(load(folder / "metadata.json"))
            executor.phase = "response_ready"
        else:
            if executor.phase == "inflight" and not (folder / "error.json").exists():
                save(folder / "error.json", {"error": "Interrupted request; response/billing unknown; attempt remains charged",
                     "seconds": None, "requested_model": settings.get("model"), "billing": "unknown"})
            executor.phase, executor.gpt_error = "ready", None
        checkpoint(executor)
    if executor.phase == "response_ready":
        # Re-execute only a saved model decision, in a new local transaction.
        if time.monotonic() - executor.started >= executor.config.seconds:
            return export(executor, "scene_time_budget_response_retained_without_commit")
        apply_packet(executor, executor.ready_packet)
    while unresolved(executor) and executor.model_calls < max_calls:
        if time.monotonic() - executor.started >= executor.config.seconds:
            return export(executor, "scene_time_budget")
        executor.model_calls += 1
        folder = run / "gpt_calls" / f"{executor.model_calls:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        executor.phase, executor.gpt_error = "inflight", None
        checkpoint(executor)  # Charge the attempt BEFORE any external request.
        urls, images = image_bundle(executor, folder)
        context = build_context(executor, [r["evidence_id"] for r in images], images)
        save(folder / "context.json", context)
        try:
            if planner is None:
                packet, meta = request_packet(executor, context, urls, settings, folder)
            else:
                packet, meta = planner(executor, context, executor.model_calls)
                validate_packet(packet, context)
                save(folder / "decision.json", packet)
                save(folder / "metadata.json", meta)
        except RuntimeError as error:
            executor.phase, executor.gpt_error = "external_error", str(error)
            checkpoint(executor)
            return export(executor, "external_api_or_response_error")
        executor.model_records.append(meta)
        executor.phase, executor.ready_packet = "response_ready", packet
        checkpoint(executor)
        if time.monotonic() - executor.started >= executor.config.seconds:
            return export(executor, "scene_time_budget_response_retained_without_commit")
        apply_packet(executor, packet)
    return export(executor, "all_targets_decided" if not unresolved(executor) else "image_call_budget_exhausted")


def run_empty_scene(image, run, settings, max_calls, resume_after_error):
    """Real visual inspection even with zero predictions; interrupted calls charged."""
    from fieldagent.gpt import empty_prediction_call
    state_path = run / "empty_checkpoint.json"
    if (run / "trace.json").exists():
        prior_trace = load(run / "trace.json")
        if not prior_trace.get("gpt_error"):
            return prior_trace
    prior = load(state_path) if state_path.exists() else {"attempts": 0, "phase": "ready"}
    if type(prior["attempts"]) is not int or not 0 <= prior["attempts"] <= max_calls:
        raise ValueError("Invalid saved empty-scene attempt budget")
    if prior["phase"] in {"inflight", "external_error"} and not resume_after_error:
        raise RuntimeError("Explicit continuation required for the charged empty-scene inspection")
    confirmed, error = False, None
    if prior["phase"] == "inflight":
        folder = run / "gpt_calls" / f"{prior['attempts']:03d}"
        confirmed = (folder / "metadata.json").exists() and (folder / "decision.json").exists()
        if not confirmed and not (folder / "error.json").exists():
            save(folder / "error.json", {"error": "Interrupted empty-scene request; response/billing unknown",
                "seconds": None, "requested_model": settings.get("model"), "billing": "unknown"})
    started = time.monotonic()
    if not confirmed and prior["attempts"] < max_calls:
        prior = {"attempts": prior["attempts"] + 1, "phase": "inflight"}
        save(state_path, prior)
        try:
            empty_prediction_call(image, run / "gpt_calls" / f"{prior['attempts']:03d}", settings)
            confirmed = True
        except Exception as exc:
            error = str(exc)
    save(state_path, {"attempts": prior["attempts"], "phase": "complete" if confirmed else "external_error" if error else "budget_exhausted"})
    trace = {"protocol": "whole_image_batch", "max_calls_per_scene": max_calls, "model_calls": prior["attempts"],
        "gpt_error": error, "commits": 0, "batch_commits": 0, "direct_proposals": 0, "direct_committed": 0,
        "processed_targets": 0, "unfinished_target_ids": [], "runtime_gt_access": False,
        "seconds": time.monotonic() - started, "empty_inspection_confirmed": confirmed,
        "stop_reason": "empty_scene_inspected" if confirmed else "external_error" if error else "image_call_budget_exhausted"}
    save(run / "trace.json", trace)
    return trace

# Compact observation/validation replace only the transport contract, while
# original-state guards and the durable proposal/commit engine above are reused.
from compact_observation import schema, validate_packet, image_bundle, build_context
