"""Audited transport-only extension of the frozen whole-image experiment.

Retry the same immutable HTTP payload, never a geometry transaction. Original
sources/manifest remain intact; effective policy and every POST are separate
evidence. This module does not send requests until the existing runner does.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import errno
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import scene_run
import scene_batch
from fieldagent.io import digest, load, now, save

ENV = "FIELDAGENT_SCENE_TRANSPORT_POLICY"
_BASE_INITIALIZE = scene_run.initialize_worker
_INSTALLED = False
RETRY_CODES = {408, 429, 500, 502, 503, 504}


def validate_policy(policy):
    if type(policy["max_retries"]) is not int or not 0 <= policy["max_retries"] <= 20:
        raise ValueError("max_retries must be an integer in 0..20")
    for name in ("timeout_seconds", "backoff_seconds", "max_backoff_seconds"):
        if not isinstance(policy[name], (int, float)) or not math.isfinite(policy[name]) or policy[name] <= 0:
            raise ValueError(name + " must be finite and positive")
    return policy


def retryable(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRY_CODES
    if isinstance(error, ssl.SSLError):
        return False  # Certificate/configuration failures require inspection.
    if isinstance(error, urllib.error.URLError):
        return retryable(error.reason)
    if isinstance(error, (TimeoutError, ConnectionError, socket.gaierror,
                          http.client.IncompleteRead, http.client.RemoteDisconnected)):
        return True
    return isinstance(error, OSError) and (error.errno in {
        errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED,
        errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPIPE
    } or getattr(error, "winerror", None) in {10053, 10054, 10060, 10061, 10065})


def retry_after(error):
    if not isinstance(error, urllib.error.HTTPError) or not error.headers:
        return 0.
    value = error.headers.get("Retry-After")
    if not value:
        return 0.
    try:
        number = float(value)
        return max(0., number) if math.isfinite(number) else 0.
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0., (date - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0.


class Transport:
    def __init__(self, policy, folder, opener, remaining, *, sleep=time.sleep, clock=time.monotonic):
        self.policy = validate_policy(policy)
        self.folder = Path(folder) / "transport_attempts"
        self.opener, self.remaining, self.sleep, self.clock = opener, remaining, sleep, clock
        self.records = []

    def open(self, request, *args, **kwargs):
        if args or request.get_method() != "POST":
            raise ValueError("Retry adapter accepts only the existing POST request")
        endpoint = urlsplit(request.full_url)
        endpoint = endpoint.scheme + "://" + endpoint.hostname + endpoint.path
        payload_hash = hashlib.sha256(request.data or b"").hexdigest()
        for index in range(self.policy["max_retries"] + 1):
            remaining = self.remaining()
            if remaining <= 0:
                raise TimeoutError("Scene time budget exhausted before HTTP attempt")
            timeout = min(self.policy["timeout_seconds"], remaining)
            record = {"attempt": index + 1, "started_utc": now(), "status": "inflight",
                "endpoint": endpoint, "payload_sha256": payload_hash, "timeout_seconds": timeout,
                "billing": "unknown_until_provider_confirmation"}
            self.folder.mkdir(parents=True, exist_ok=True)
            path = self.folder / f"{index + 1:03d}.json"
            if path.exists():
                raise RuntimeError("Existing HTTP attempt evidence; refusing to overwrite or replay")
            save(path, record)  # Durable before POST, including interrupts.
            self.records.append(record)
            started = self.clock()
            try:
                options = dict(kwargs, timeout=timeout)
                with self.opener(request, **options) as response:
                    body = response.read()  # Also retry timeouts/truncated reads after headers.
            except Exception as error:
                transient = retryable(error)
                record.update(status="failed", seconds=self.clock() - started,
                    error_type=type(error).__name__, http_status=getattr(error, "code", None),
                    retryable=transient)
                requested_wait = max(min(self.policy["max_backoff_seconds"],
                    self.policy["backoff_seconds"] * 2 ** index), retry_after(error))
                again = transient and index < self.policy["max_retries"] and requested_wait < self.remaining()
                record.update(will_retry=again, retry_delay_seconds=requested_wait if again else None)
                save(path, record)
                if isinstance(error, urllib.error.HTTPError):
                    error.close()
                if not again:
                    raise
                print(f"HTTP RETRY {self.folder.parents[4].name} "
                      f"call={self.folder.parent.name} {index + 1}/{self.policy['max_retries']} "
                      f"{type(error).__name__} status={getattr(error, 'code', '-')} "
                      f"wait={requested_wait:g}s", flush=True)
                until = self.clock() + requested_wait
                while self.clock() < until:
                    self.sleep(min(.5, until - self.clock()))
                continue
            # A persistence error after receiving a response must not resend it.
            record.update(status="response_received", seconds=self.clock() - started,
                          response_sha256=hashlib.sha256(body).hexdigest())
            (self.folder / f"{index + 1:03d}.response.bin").write_bytes(body)
            save(path, record)
            return io.BytesIO(body)

    def summary(self):
        return {"http_attempts": len(self.records), "retries": max(0, len(self.records) - 1),
            "effective_timeout_seconds": self.policy["timeout_seconds"],
            "policy": self.policy, "evidence": str(self.folder),
            "transport_failures": sum(r["status"] == "failed" for r in self.records),
            "billing_on_missing_responses": "unknown; retry may be billed separately"}


@contextmanager
def use_transport(policy, folder, remaining):
    original = urllib.request.urlopen
    transport = Transport(policy, folder, original, remaining)
    urllib.request.urlopen = transport.open
    try:
        yield transport
    finally:
        urllib.request.urlopen = original
        details = transport.summary()
        save(Path(folder) / "transport.json", details)
        for name in ("metadata.json", "error.json"):
            path = Path(folder) / name
            if path.exists():
                meta = load(path)
                meta.update(retries=details["retries"], transport=details)
                save(path, meta)


def install(policy):
    global _INSTALLED
    if _INSTALLED:
        return
    validate_policy(policy)
    import scene_inputs
    scene_inputs.install()
    _INSTALLED = True
    original_request = scene_batch.request_packet

    def request(executor, context, urls, settings, folder):
        remaining = lambda: executor.config.seconds - (time.monotonic() - executor.started)
        with use_transport(policy, folder, remaining) as transport:
            packet, meta = original_request(executor, context, urls, settings, folder)
        meta.update(retries=transport.summary()["retries"], transport=transport.summary())
        return packet, meta

    scene_batch.request_packet = request
    from fieldagent import gpt
    original_empty = gpt.empty_prediction_call

    def empty(image, folder, settings):
        # Same finite 7200-second ceiling used by nonempty scene workers.
        deadline = time.monotonic() + 7200.
        with use_transport(policy, folder, lambda: deadline - time.monotonic()):
            return original_empty(image, folder, settings)

    gpt.empty_prediction_call = empty
    original_report = scene_run.build_report

    def report(manifest, out):
        result = original_report(manifest, out)
        augment_report(out, result)
        return result

    scene_run.build_report = report
    scene_run.initialize_worker = initialize_worker
    original_run = scene_run.run_gpt

    def run(manifest, out, workers, resume_after_error):
        session = Path(out) / "transport_sessions" / str(time.time_ns())
        save(session / "policy.json", {"created_utc": now(), "policy": policy,
            "extension_source_sha256": digest(__file__),
            "input_preprocessing_source_sha256": digest(Path(__file__).with_name("scene_inputs.py")),
            "launcher_sha256": digest(Path(__file__).parent / "scripts/run_gpt_scene_retry.py"),
            "base_manifest_sha256": digest(Path(out) / "manifest.json"),
            "logical_budget_unchanged": True,
            "note": "Effective transport override; frozen manifest describes original no-retry transport"})
        print(f"TRANSPORT timeout={policy['timeout_seconds']:g}s max_retries={policy['max_retries']} "
              "(HTTP attempts counted separately from decision slots)", flush=True)
        return original_run(manifest, out, workers, resume_after_error)

    scene_run.run_gpt = run


def initialize_worker():
    _BASE_INITIALIZE()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    install(json.loads(os.environ[ENV]))


def augment_report(out, result):
    out = Path(out)
    preprocessing = [load(path) for path in out.glob("input_preprocessing/*/record.json")]
    result["input_preprocessing"] = {
        "policy": "exclude_empty_polygon_placeholders_all_arms_v1",
        "affected_images": len(preprocessing),
        "excluded_empty_instances": sum(len(r["excluded"]) for r in preprocessing),
        "excluded_vertices": 0,
        "records": [{k: r[k] for k in ("scene", "original_instances", "usable_instances", "excluded")}
                    for r in preprocessing],
        "note": "Empty polygon placeholders have no editable boundary. Local controls/evaluation recomputed for affected images."}
    attempt_paths = list(out.glob("scenes/*/gpt_autonomous/run/gpt_calls/*/transport_attempts/*.json"))
    attempts = [load(path) for path in attempt_paths]
    retry_folders = {path.parent for path in attempt_paths}
    # Old calls have one HTTP attempt per charged slot and no transport ledger.
    old = sum(1 for path in out.glob("scenes/*/gpt_autonomous/run/gpt_calls/*")
              if path.is_dir() and not (path / "transport_attempts").exists())
    result["transport_accounting"] = {
        "http_attempts_recorded": len(attempts), "legacy_charged_slots_without_transport_ledger": old,
        "total_http_attempts_including_legacy_estimate": len(attempts) + old,
        "extra_retry_attempts": sum(max(0, len(list(p.glob("*.json"))) - 1) for p in retry_folders),
        "transport_failures": sum(r["status"] == "failed" for r in attempts),
        "interrupted_attempts_with_unknown_response": sum(r["status"] == "inflight" for r in attempts),
        "cost": None,
        "note": "model_calls remains charged decision slots; HTTP POST attempts include retries; missing-response billing unknown",
        "effective_policy_sessions": "transport_sessions/*/policy.json"}
    save(out / "summary.json", result)
    save(out / "TRANSPORT_SUMMARY.json", result["transport_accounting"])
    report = out / "RESULTS.md"
    if report.exists():
        report.write_text(f"输入预处理：{len(preprocessing)} 张图共排除 {sum(len(r['excluded']) for r in preprocessing)} 个空轮廓实例，四组一致；详见 input_preprocessing。\n\n"
            + "通信重试扩展：model_calls 为决策槽位，实际 HTTP 请求（含重试）见 TRANSPORT_SUMMARY.json。"
            "有效超时/重试策略见 transport_sessions；原 manifest 保留最初无重试配置。\n\n"
            + report.read_text(encoding="utf-8"), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--request-timeout", type=float, default=300.)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=10.)
    parser.add_argument("--retry-max-delay", type=float, default=60.)
    args, rest = parser.parse_known_args(argv)
    policy = validate_policy({"timeout_seconds": args.request_timeout,
        "max_retries": args.max_retries, "backoff_seconds": args.retry_delay,
        "max_backoff_seconds": args.retry_max_delay})
    os.environ[ENV] = json.dumps(policy)
    install(policy)
    return scene_run.main(rest)
