"""Geometry goals over the existing compatible Responses/Chat transport."""
from __future__ import annotations

from dataclasses import replace

from fieldbqc_geometry import ACTIONS, PROFILES
from fieldbqc_planner import APIPlanner, visual_inputs

VERSION = "field-geometry-target-v3"
PROMPT = """You coordinate agricultural parcel geometry repair after visual segmentation.
Return a JSON GeometryGoal matching the supplied schema. Do not split or merge
semantic parcels. Use only available_actions, known IDs and parameter profiles.
Minimize overlap and invalid geometry, coordinate shared boundaries and reduce
vertices subject to the supplied drift/topology constraints.
The default vertex goal is at most 10 per parcel, fewer for simple shapes. Counts
include all exterior/interior rings, without the repeated closing coordinate.
Never pad a rectangle to 10 points or sacrifice the hard quality limits to reach
the goal. Simplification must not increase any individual parcel's vertex count,
even if the group's total falls. Multiple components or holes may require more than 10. A deterministic
final compaction pass handles remaining parcels except reviewed targets or keep.
Real roads, ditches, holes and narrow fields may be valid: uncertain geometry flags are not labels.
Local deterministic tools execute all edits and independently accept or roll back.
Use numeric before/candidate feedback and rejection reasons to choose subsequent
actions. Shared boundaries need joint treatment. Repeated ineffective operations
are wasteful. Review defers the specified targets while other targets can continue;
keep stops the image. A tool error is not evidence of a correct repair.
The images show raw/overlay overview followed by a crop of the first available
target group. No ground truth is available. Never change limits or invent actions.
"""
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "decision": {"type": "string", "enum": ["repair", "keep", "review"]},
    "issue": {"type": "string", "enum": list(ACTIONS) + ["none"]},
    "target_ids": {"type": "array", "items": {"type": "string"}},
    "action": {"type": ["string", "null"], "enum": list(ACTIONS) + [None]},
    "parameter_profile": {"type": "string", "enum": list(PROFILES)},
    "reason": {"type": "string"}},
    "required": ["decision", "issue", "target_ids", "action", "parameter_profile", "reason"]}


def validate_geometry_goal(goal, context):
    if not isinstance(goal, dict) or set(goal) != set(SCHEMA["required"]):
        raise ValueError("GeometryGoal fields do not match schema")
    for key in ("decision", "issue", "action", "parameter_profile"):
        if goal[key] not in SCHEMA["properties"][key]["enum"]:
            raise ValueError(f"Invalid GeometryGoal {key}")
    ids = goal["target_ids"]
    if (not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Targets must be unique string IDs")
    if not set(ids) <= {p["parcel_id"] for p in context["instances"]}:
        raise ValueError("Unknown geometry target")
    if not isinstance(goal["reason"], str) or not 1 <= len(goal["reason"]) <= 2000:
        raise ValueError("Expected a short observable reason")
    if goal["decision"] == "repair":
        if goal["issue"] != goal["action"] or not any(
                a["action"] == goal["action"] and set(a["target_ids"]) == set(ids)
                for a in context["available_actions"]):
            raise ValueError("Unavailable geometry action or mismatched issue")
        # Canonical ordering is important for seam provenance and cache consistency.
        goal = dict(goal, target_ids=sorted(ids))
    elif goal["action"] is not None:
        raise ValueError("keep/review cannot execute a tool")
    elif goal["decision"] == "review" and not ids:
        raise ValueError("Local review requires at least one target")
    return goal


class GeometryPlanner(APIPlanner):
    def __init__(self, config, cache_dir=None):
        self.crop_ids = set()
        super().__init__(config, cache_dir, schema=SCHEMA, system_prompt=PROMPT,
                         validator=validate_geometry_goal, prompt_version=VERSION,
                         visualizer=self.images)

    def images(self, image, parcels):
        urls = visual_inputs(image, parcels)
        selected = [p for p in parcels if p.parcel_id in self.crop_ids]
        if selected:
            import numpy as np
            rows, cols = np.nonzero(np.logical_or.reduce([p.mask for p in selected]))
            if len(rows):
                y0, y1 = max(0, int(rows.min()) - 12), min(image.shape[0], int(rows.max()) + 13)
                x0, x1 = max(0, int(cols.min()) - 12), min(image.shape[1], int(cols.max()) + 13)
                local = [replace(p, mask=p.mask[y0:y1, x0:x1]) for p in selected]
                urls.extend(visual_inputs(image[y0:y1, x0:x1], local))
        return urls

    def plan(self, context, image, parcels, remaining_seconds):
        first = context["available_actions"][0]
        self.crop_ids = set(first["target_ids"])
        context = dict(context, crop_target_ids=sorted(self.crop_ids))
        return super().plan(context, image, parcels, remaining_seconds)
