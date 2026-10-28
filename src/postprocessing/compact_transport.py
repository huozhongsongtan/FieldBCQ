"""Audited transient transport retries and CLI for compact scene v2."""
import argparse
import json
import os
from pathlib import Path
import time

from fieldagent.io import load, save
from scene_retry import use_transport, validate_policy, augment_report

ENV = "FIELDAGENT_COMPACT_TRANSPORT"
_INSTALLED = False


def policy():
    return validate_policy(json.loads(os.environ.get(ENV, json.dumps(dict(
        timeout_seconds=300., max_retries=5, backoff_seconds=10., max_backoff_seconds=60.)))))


def install_worker_transport():
    global _INSTALLED
    if _INSTALLED:
        return
    import compact_scene as cs
    from fieldagent import gpt
    base_request, base_empty = cs.request_packet, gpt.empty_prediction_call

    def request(executor, context, urls, settings, folder):
        remaining = lambda: executor.config.seconds - (time.monotonic() - executor.started)
        with use_transport(policy(), folder, remaining) as transport:
            packet, meta = base_request(executor, context, urls, settings, folder)
        meta.update(retries=transport.summary()["retries"], transport=transport.summary())
        return packet, meta

    def empty(image, folder, settings):
        deadline = time.monotonic() + 7200.
        with use_transport(policy(), folder, lambda: deadline - time.monotonic()):
            return base_empty(image, folder, settings)

    cs.request_packet, gpt.empty_prediction_call = request, empty
    _INSTALLED = True


def detailed_target_count(context, path):
    """Empty-scene visual inspections use a separate, explicit input contract."""
    if "detailed_target_ids" in context:
        targets = context["detailed_target_ids"]
        if not isinstance(targets, list):
            raise ValueError("Invalid detailed_target_ids in " + str(path))
        return len(targets)
    if (context.get("predicted_instances") == []
            and context.get("goal") == "boundary vertex reduction only"
            and context.get("runtime_gt_access") is False):
        return 0
    raise ValueError("Unrecognized compact context contract in " + str(path))


def compact_report(manifest, out, result):
    rows = []
    for entry in manifest["entries"]:
        run = Path(out) / "scenes" / entry["scene"] / "gpt_autonomous/run"
        context_paths = sorted((run / "gpt_calls").glob("*/context.json"))
        contexts = [load(p) for p in context_paths]
        pools = [load(p) for p in (run / "local_candidates").glob("*/pool.json")]
        if not contexts:
            continue
        rows.append(dict(scene=entry["scene"], calls=len(contexts),
            context_characters=sum(len(json.dumps(c, ensure_ascii=True)) for c in contexts),
            detailed_targets_by_call=[detailed_target_count(c, p) for p, c in zip(context_paths, contexts)],
            local_preparation_seconds=sum(p["seconds"] for p in pools),
            local_candidate_attempts=sum(len(p["attempts"]) for p in pools),
            automatic_candidate_selection=False))
    result.update(protocol="compact_whole_image_v2", interaction_metrics=rows,
        speedup_measured_with_real_GPT=False,
        note="Context compression is measured separately from API token/latency savings; candidate generation adds local computation.")
    augment_report(out, result)
    save(Path(out) / "COMPACT_INTERACTION.json", rows)


def main(argv=None):
    import compact_run
    import compact_inputs
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request-timeout", type=float, default=300.)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=10.)
    parser.add_argument("--retry-max-delay", type=float, default=60.)
    args, rest = parser.parse_known_args(argv)
    cfg = validate_policy(dict(timeout_seconds=args.request_timeout, max_retries=args.max_retries,
        backoff_seconds=args.retry_delay, max_backoff_seconds=args.retry_max_delay))
    os.environ[ENV] = json.dumps(cfg)
    compact_inputs.install()
    install_worker_transport()
    return compact_run.main(rest)
