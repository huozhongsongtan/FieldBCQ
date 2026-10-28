"""Explicit continuation: preserve raw anchors, paid attempts and audit IDs."""
import copy
from pathlib import Path
import shutil
import time
from .io import digest, load, save, now

def restore_executor(ba, image, original, run, config, prior):
    from cvr_agent import Audit, read_items, state_key
    raw_path = run / "baseline.pixel.geojson"
    if state_key(read_items(raw_path)) != state_key(original):
        raise ValueError("Checkpoint baseline differs from frozen raw predictions")
    final_path = run / "final" / "prediction.repaired.pixel.geojson"
    current = read_items(final_path)
    expected = prior["history"][-1]["output_state"] if prior["history"] else state_key(original)
    if state_key(current) != expected:
        raise ValueError("Last committed checkpoint was changed")
    # Initialize common gates against RAW predictions, not the repaired geometry.
    init_root = run / "recovery_initialization" / str(time.time_ns())
    e = ba.AutonomousExecutor(image, original, init_root, config)
    e.root = run
    e.current = current
    e.history = copy.deepcopy(prior["history"])
    e.decisions = copy.deepcopy(prior["decisions"])
    e.pending_proposals = {p["proposal_id"]: copy.deepcopy(p) for p in prior["proposals"]}
    e.preview_attempts = prior["preview_attempts"]
    e.proposal_counter = max((int(pid[1:]) for pid in e.pending_proposals), default=0)
    e.goal_mode = prior["goal_mode"]
    e.direct_only = False
    e.started = time.monotonic() - prior["seconds"]
    e.audit = Audit(run)
    log = run / "calls.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            record = __import__("json").loads(line)
            e.audit.counter = max(e.audit.counter, int(record["call_id"][1:]))
            if record["status"] == "completed":
                e.audit.methods.update(record.get("methods", []))
    for proposal in e.pending_proposals.values():
        ref = proposal.get("candidate_file")
        if ref is None:
            continue
        if digest(ref["path"]) != ref["sha256"]:
            raise ValueError("Stored proposal was changed: " + proposal["proposal_id"])
        target = next(p for p in read_items(ref["path"]) if p.identifier == proposal["target_id"])
        e.tried.add((proposal["base_state"], proposal["target_id"], proposal["task"], target.geometry.normalize().wkb_hex))
    with e.audit.call("restore_verified_GPT_checkpoint", {
        "prior_trace_sha256": digest(run / "trace.json"), "restored_state": e.state_id,
        "completed_targets_will_be_skipped": True, "failed_attempts_use_original_budget": True,
        "original_anchor_state": state_key(e.original)}, ["checkpoint_hash_check", "preserve_raw_anchors"]) as event:
        event["restored"] = True
    return e


def continue_scene(ba, image, original, run, config, settings, calls_per_target, *, planner=None):
    """Reuse the frozen observation/proposal/commit loop one unfinished target at a time."""
    prior = load(run / "trace.json")
    if prior["goal_mode"] != "minimize_vertices":
        raise ValueError("Unexpected checkpoint objective")
    all_ids = [p.identifier for p in original]
    if prior["target_ids"] != all_ids:
        raise ValueError("Checkpoint targets differ from frozen prediction order")
    counts = dict(prior["per_target_calls"])
    if any(type(n) is not int or not 0 <= n <= calls_per_target for n in counts.values()):
        raise ValueError("Invalid saved call budget")
    unfinished = set(prior["unfinished_target_ids"])
    remaining = [uid for uid in all_ids if uid in unfinished and counts.get(uid, 0) < calls_per_target]
    if not remaining:
        raise ValueError("No remaining decision budget in failed scene")
    snapshot = run / "recovery_snapshots" / str(time.time_ns())
    snapshot.mkdir(parents=True)
    for name in ("trace.json", "progress.json", "methods.json", "tool_catalog.json"):
        if (run / name).exists():
            shutil.copyfile(run / name, snapshot / name)
    e = restore_executor(ba, image, original, run, config, prior)
    request = ba.request_packet if planner is None else planner
    original_constructor = ba.AutonomousExecutor
    records = list(prior["model_records"])
    attempted = prior["model_calls"]
    prior_failures = list(prior.get("recovered_external_errors", []))
    if prior.get("gpt_error"):
        prior_failures.append({"error": prior["gpt_error"], "attempted_calls": attempted,
            "retained_snapshot": str(snapshot / "trace.json")})
    merged = prior
    try:
        # The constructor is replaced only while reusing this EXISTING executor.
        # It preserves original anchors, prior proposals, candidate IDs and audit IDs.
        ba.AutonomousExecutor = lambda *args, **kwargs: e
        for uid in remaining:
            offset = attempted
            before_decisions = len(e.decisions)
            remaining_budget = calls_per_target - counts[uid]

            def numbered_planner(executor, pool, ids, cfg, number, allow_direct):
                return request(executor, pool, ids, cfg, offset + number, allow_direct)

            trace = ba.run_autonomous(image, original, run, config, settings, ids=[uid],
                max_calls=remaining_budget, calls_per_target=remaining_budget,
                allow_direct=True, goal_mode="minimize_vertices", planner=numbered_planner)
            for decision in e.decisions[before_decisions:]:
                decision["call"] += offset
            attempted += trace["model_calls"]
            counts[uid] += trace["per_target_calls"][uid]
            records.extend(trace["model_records"])
            if uid not in trace["unfinished_target_ids"]:
                unfinished.discard(uid)
            merged = dict(trace, target_ids=all_ids, per_target_calls=dict(counts), model_calls=attempted,
                model_records=list(records), processed_targets=len(all_ids)-len(unfinished),
                unfinished_target_ids=[u for u in all_ids if u in unfinished],
                decisions=copy.deepcopy(e.decisions), recovered_external_errors=prior_failures,
                recovery_protocol="preserve_raw_anchors_and_per_target_attempt_caps_v1",
                recovery_snapshot=str(snapshot))
            if not trace["gpt_error"]:
                merged["stop_reason"] = "all_targets_decided" if not unfinished else "remaining_targets_or_budget_incomplete"
            save(run / "trace.json", merged)
            save(run / "progress.json", {"model_calls": attempted, "target_ids": all_ids,
                "processed_targets": merged["processed_targets"], "per_target_calls": dict(counts),
                "unfinished_target_ids": merged["unfinished_target_ids"], "current": merged["final"],
                "commits": merged["commits"], "last_target": uid, "updated_utc": now()})
            if trace["gpt_error"]:
                break
    finally:
        ba.AutonomousExecutor = original_constructor
    return merged

