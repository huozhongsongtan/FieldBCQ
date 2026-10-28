"""Vertex-only GPT capabilities, inherited from the frozen experiment."""
import base64
import copy
import json
import os
from pathlib import Path
import time
from .io import digest, load, save
HERE = Path(__file__).resolve().parent

def boundary_only_capabilities():
    """Narrow the existing model schema, discovery and validation consistently."""
    import boundary_autonomous as ba
    if getattr(ba, "_fieldagent_vertex_only", False):
        return ba, list(ba._fieldagent_allowed_tools)
    from boundary_tools import REGISTRY
    allowed = {name for name, spec in REGISTRY.items() if spec.task == "simplify"}
    original_context, original_validate = ba.build_context, ba.validate_packet
    commands = copy.deepcopy(ba.COMMANDS)
    commands = [s for s in commands if s["properties"]["kind"]["enum"] != ["use_pool"]]
    for spec in commands:
        kind = spec["properties"]["kind"]["enum"][0]
        if kind == "direct_edit":
            spec["properties"]["task"]["enum"] = ["simplify"]
            edits = spec["properties"]["edits"]["items"]["anyOf"]
            spec["properties"]["edits"]["items"]["anyOf"] = [
                s for s in edits if s["properties"]["kind"]["enum"][0] != "move_vertices"]
        elif kind == "use_tool":
            spec["properties"]["tool_id"]["enum"] = sorted(allowed)
    ba.COMMANDS = commands
    ba.SCHEMA = copy.deepcopy(ba.SCHEMA)
    ba.SCHEMA["properties"]["commands"]["items"]["anyOf"] = commands
    ba.PROMPT = (HERE / "GPT_VERTEX_ONLY_PROMPT.md").read_text(encoding="utf-8")
    ba.MINIMUM_VERTICES_PROMPT = ""

    def context(*args, **kwargs):
        ctx = original_context(*args, **kwargs)
        ctx["available_tools"] = {k: v for k, v in ctx["available_tools"].items() if k in allowed}
        ctx["pool_candidates"] = []
        ctx["research_objective"] = "Minimize necessary boundary vertices only"
        return ctx

    def validate(packet, ctx):
        result = original_validate(packet, ctx)
        for command in packet["commands"]:
            if command["kind"] == "use_tool" and command["tool_id"] not in allowed:
                raise ValueError("Vertex-only run forbids non-simplification tools")
            if command["kind"] == "direct_edit":
                if command["task"] != "simplify" or any(
                    e.get("kind") not in {"retain_vertices", "replace_boundary_arc"} for e in command["edits"]):
                    raise ValueError("Vertex-only run forbids movement/repair edits")
        return result
    ba.build_context, ba.validate_packet = context, validate
    ba._fieldagent_vertex_only = True
    ba._fieldagent_allowed_tools = sorted(allowed)
    return ba, sorted(allowed)


def empty_prediction_call(image, folder, settings):
    """Even an empty prediction gets a real visual GPT inspection, no hallucination."""
    import urllib.request
    import cv2
    from fieldbqc_planner import APIPlanner
    folder.mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Cannot encode empty-scene evidence")
    (folder / "original.png").write_bytes(png.tobytes())
    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    schema = {"type": "object", "additionalProperties": False,
        "properties": {"diagnosis": {"type": "string"}, "done": {"type": "boolean"}},
        "required": ["diagnosis", "done"]}
    prompt = "Inspect the image. There are zero predicted parcel instances to simplify. Report no editable targets, done=true. Do not detect new instances or invent repairs."
    ctx = {"predicted_instances": [], "goal": "boundary vertex reduction only", "runtime_gt_access": False}
    payload = APIPlanner(settings, schema=schema, system_prompt=prompt)._payload(ctx, [url])
    save(folder / "context.json", ctx); save(folder / "request.json", payload)
    (folder / "system_prompt.md").write_text(prompt, encoding="utf-8")
    endpoint = settings["base_url"].rstrip("/") + ("/responses" if settings["api"] == "responses" else "/chat/completions")
    meta = {"endpoint": endpoint, "requested_model": settings["model"],
        "request_sha256": digest(folder / "request.json"), "retries": 0}
    started = time.monotonic()
    try:
        key = os.environ.get(settings.get("api_key_env", "OPENAI_API_KEY"))
        if not key:
            raise RuntimeError("Configured API key unavailable")
        request = urllib.request.Request(endpoint, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=settings.get("timeout_seconds", 90)) as response:
            data = json.load(response)
        save(folder / "response.json", data)
        if settings["api"] == "responses":
            raw = "".join(c.get("text", "") for m in data.get("output", []) if m.get("type") == "message"
                for c in m.get("content", []) if c.get("type") == "output_text")
            complete = data.get("status") == "completed"
        else:
            choice = data["choices"][0]
            raw, complete = choice["message"]["content"], choice.get("finish_reason") == "stop"
        decision = json.loads(raw)
        if not complete or set(decision) != {"diagnosis", "done"} or decision["done"] is not True or not decision["diagnosis"]:
            raise ValueError("Invalid empty-scene GPT decision")
        meta.update(seconds=time.monotonic()-started, usage=data.get("usage", {}), response_model=data.get("model"), completed=True)
        save(folder / "decision.json", decision); save(folder / "metadata.json", meta)
        return {"model_calls": 1, "gpt_error": None, "commits": 0, "direct_proposals": 0,
            "direct_committed": 0, "unfinished_target_ids": [], "per_target_calls": {}}
    except Exception as exc:
        meta.update(error=type(exc).__name__ + ": " + str(exc), seconds=time.monotonic()-started)
        save(folder / "error.json", meta)
        raise

