"""Bounded RepairGoal planning through Responses or compatible Chat APIs.

Only prediction evidence enters this module. Ground truth belongs in the offline
evaluator. No model-generated code or policy overrides are executed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

PROMPT_VERSION = "field-repair-v1"
SYSTEM_PROMPT = """You plan conservative repairs of agricultural parcel instances.
Return one JSON RepairGoal matching the supplied schema. Use only available_actions.
Diagnostic flags, MQC, distance peaks and edges are uncertain evidence, not labels.
Choose repair only when evidence supports the operation; otherwise keep or review.
The raw image and overlay, when provided, show the same pixel coordinates; overlay
labels identify prediction IDs. No ground truth is available. Explain the decision
briefly using observable evidence. Never invent IDs, tools, thresholds or commands.
split_instances separates adhesion; fragment_merge joins disconnected fragments;
contour_simplify reduces vertices; local_rerun requests a crop prediction;
boundary_cleanup is optional boundary refinement. A deterministic verifier decides
whether candidates can be committed. Keep preserves predictions and does not certify
their correctness. Review preserves predictions for later inspection.
"""
GOAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["repair", "keep", "review"]},
        "issue": {"type": "string", "enum": ["adhesion", "fragmentation", "contour_complexity",
                                                  "low_confidence", "boundary_shift", "none"]},
        "target_ids": {"type": "array", "items": {"type": "string"}},
        "action": {"type": ["string", "null"], "enum": [None, "split_instances", "fragment_merge",
                                                           "contour_simplify", "local_rerun", "boundary_cleanup"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "issue", "target_ids", "action", "reason"],
}


class PlanningError(ValueError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def build_context(parcels, state, policy, available, history, iterations, local_calls):
    instances = []
    for parcel, observation in zip(parcels, state.observations):
        rows, columns = np.nonzero(parcel.mask)
        box = ([int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1]
               if len(rows) else [0, 0, 0, 0])
        instances.append(dict(asdict(observation), bbox=box, class_id=parcel.cls,
                              confidence=parcel.score,
                              mask_quality=parcel.mask_quality if parcel.quality_valid else None,
                              boundary_quality=parcel.boundary_quality if parcel.quality_valid else None))
    return {
        "instances": instances, "available_actions": available, "policy": policy.to_dict(),
        "remaining_budget": {"iterations": iterations, "local_calls": local_calls},
        "history": [{key: item[key] for key in ("action", "target_ids", "accepted", "reasons") if key in item}
                    for item in history if item.get("action") not in {"plan", "stop"}][-8:],
    }


def validate_goal(goal, context):
    if not isinstance(goal, dict) or set(goal) != set(GOAL_SCHEMA["required"]):
        raise ValueError("RepairGoal fields do not match the schema")
    for name in ("decision", "issue", "action"):
        if goal[name] not in GOAL_SCHEMA["properties"][name]["enum"]:
            raise ValueError(f"Invalid RepairGoal {name}")
    ids = goal["target_ids"]
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("RepairGoal target_ids must be unique strings")
    known = {item["parcel_id"] for item in context["instances"]}
    if not set(ids) <= known or any("|" in item for item in ids):
        raise ValueError("RepairGoal contains an unknown instance ID")
    if not isinstance(goal["reason"], str) or not 1 <= len(goal["reason"]) <= 2000:
        raise ValueError("RepairGoal reason must be a short string")
    if goal["decision"] == "repair":
        if {"action": goal["action"], "target_ids": ids} not in context["available_actions"]:
            raise ValueError("RepairGoal requests an unavailable action or exhausted budget")
    elif goal["action"] is not None:
        raise ValueError("keep/review must not specify an action")
    return goal


def visual_inputs(image, parcels, max_side=768):
    raw = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = raw.copy()
    scale = min(1.0, max_side / max(raw.shape[:2]))
    if scale < 1:
        size = (max(1, round(raw.shape[1] * scale)), max(1, round(raw.shape[0] * scale)))
        raw = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
        overlay = raw.copy()
    for parcel in parcels:
        mask = cv2.resize(parcel.mask.astype(np.uint8), (raw.shape[1], raw.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        if contours:
            x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.putText(overlay, str(parcel.parcel_id), (x, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1, cv2.LINE_AA)
    urls = []
    for frame in (raw, overlay):
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise ValueError("Could not encode planner image")
        urls.append("data:image/png;base64," + base64.b64encode(encoded).decode("ascii"))
    return urls


class APIPlanner:
    def __init__(self, config, cache_dir=None, *, schema=None, system_prompt=None,
                 validator=None, prompt_version=None, visualizer=None):
        self.schema = GOAL_SCHEMA if schema is None else schema
        self.system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.validator = validate_goal if validator is None else validator
        self.prompt_version = PROMPT_VERSION if prompt_version is None else prompt_version
        self.visualizer = visual_inputs if visualizer is None else visualizer
        allowed = {"api", "base_url", "model", "api_key_env", "vision", "timeout_seconds",
                   "max_output_tokens", "request_overrides", "response_format", "max_instances"}
        if set(config) - allowed:
            raise ValueError(f"Unknown planner settings: {sorted(set(config) - allowed)}")
        self.config = dict(config)
        if config.get("api") not in {"responses", "chat"} or not config.get("model"):
            raise ValueError("Planner needs api=responses|chat and an explicit model ID")
        url = config.get("base_url", "")
        if "REPLACE_" in url or not url.startswith(("https://", "http://localhost:", "http://127.0.0.1:")):
            raise ValueError("Use HTTPS or an explicit localhost inference endpoint")
        if not isinstance(config.get("vision", False), bool):
            raise ValueError("vision must be a boolean")
        for key, default in (("timeout_seconds", 45), ("max_output_tokens", 2048), ("max_instances", 24)):
            value = config.get(key, default)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid planner setting: {key}")
        if config.get("response_format", "json_schema") not in {"json_schema", "json_object"}:
            raise ValueError("response_format must be json_schema or json_object")
        overrides = config.get("request_overrides", {})
        if set(overrides) - {"reasoning", "temperature", "enable_thinking", "seed"}:
            raise ValueError("Only reasoning, temperature, enable_thinking and seed overrides are allowed")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _payload(self, context, urls):
        description = json.dumps({"context": context, "RepairGoal_schema": self.schema},
                                 ensure_ascii=True, allow_nan=False)
        if self.config["api"] == "responses":
            content = [{"type": "input_text", "text": description}]
            content.extend({"type": "input_image", "image_url": url} for url in urls)
            payload = {
                "model": self.config["model"], "instructions": self.system_prompt, "store": False,
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": int(self.config.get("max_output_tokens", 2048)),
                "text": {"format": {"type": "json_schema", "name": "RepairGoal",
                                    "strict": True, "schema": self.schema}},
            }
        else:
            content = [{"type": "text", "text": description}]
            content.extend({"type": "image_url", "image_url": {"url": url}} for url in urls)
            mode = self.config.get("response_format", "json_schema")
            response_format = {"type": mode}
            if mode == "json_schema":
                response_format["json_schema"] = {"name": "RepairGoal", "strict": True, "schema": self.schema}
            payload = {
                "model": self.config["model"], "messages": [
                    {"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}],
                "max_tokens": int(self.config.get("max_output_tokens", 2048)),
                "response_format": response_format,
            }
        payload.update(self.config.get("request_overrides", {}))
        return payload

    def plan(self, context, image, parcels, remaining_seconds):
        started = time.monotonic()
        # Bound context size while keeping the rule fallback independent of it.
        limit = int(self.config.get("max_instances", 24))
        ordered = sorted(context["instances"], key=lambda item: (-item["severity"], item["index"]))[:limit]
        ids = {item["parcel_id"] for item in ordered}
        limited = dict(context, instances=ordered,
                       available_actions=[a for a in context["available_actions"] if set(a["target_ids"]) <= ids])
        selected = [p for p in parcels if p.parcel_id in ids]
        urls = self.visualizer(image, selected) if self.config.get("vision", False) else []
        payload = self._payload(limited, urls)
        key_data = {"prompt_version": self.prompt_version, "endpoint": self.config["base_url"], "payload": payload}
        cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json" if self.cache_dir else None
        if cache_path and cache_path.exists():
            result = json.loads(cache_path.read_text(encoding="utf-8"))
            self.validator(result["goal"], limited)
            return dict(result, cache_hit=True, billed_usage={},
                        elapsed_seconds=time.monotonic() - started, input_context=limited)
        key_name = self.config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(key_name)
        if not api_key:
            raise RuntimeError(f"Set environment variable {key_name} before live planning")
        timeout = min(float(self.config.get("timeout_seconds", 45)),
                      remaining_seconds - (time.monotonic() - started))
        if timeout <= 0:
            raise TimeoutError("No time remains for planner request")
        endpoint = self.config["base_url"].rstrip("/") + (
            "/responses" if self.config["api"] == "responses" else "/chat/completions")
        request = urllib.request.Request(
            endpoint, data=json.dumps(payload, allow_nan=False).encode("utf-8"), method="POST",
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Planner HTTP {error.code}; check model ID, endpoint and API permissions") from None
        usage = data.get("usage", {})
        result = {"raw_output": "", "model": data.get("model", self.config["model"]),
                  "request_id": data.get("id"), "usage": usage, "prompt_version": self.prompt_version,
                  "cache_key": cache_key, "cache_hit": False, "billed_usage": usage,
                  "elapsed_seconds": time.monotonic() - started}
        try:
            if self.config["api"] == "responses":
                raw = "".join(part.get("text", "") for item in data.get("output", [])
                              if item.get("type") == "message" for part in item.get("content", [])
                              if part.get("type") == "output_text")
                result["raw_output"] = raw
                if data.get("status") != "completed":
                    raise ValueError("Planner response is incomplete")
            else:
                choice = data["choices"][0]
                raw = choice["message"]["content"]
                result["raw_output"] = raw
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Planner completion was truncated or refused")
            result["goal"] = self.validator(json.loads(raw), limited)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            raise PlanningError(str(error), dict(result, input_context=limited)) from None
        if cache_path:
            temporary = cache_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(cache_path)
        return dict(result, input_context=limited)
