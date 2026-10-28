"""Configurable repair recipes for the field-parcel agent.

Recipes are the task-aware solution space: they restrict which tools are
eligible for an observed issue. They contain no labels or test metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


TOOL_ACTIONS = {
    "split_instances",
    "fragment_merge",
    "contour_simplify",
    "local_rerun",
    "boundary_cleanup",
}

DEFAULT_RECIPES = {
    "merge_warning": ("split_instances", "local_rerun"),
    "fragmentation": ("fragment_merge", "local_rerun"),
    "contour_complexity": ("contour_simplify",),
    "boundary_shift": ("boundary_cleanup", "local_rerun"),
    "low_quality": ("local_rerun",),
    "small_instance": ("local_rerun",),
}


class RepairRecipeLibrary:
    """Validated issue-to-tool recipes used by rule and LLM routing."""

    def __init__(self, recipes: Mapping[str, Sequence[str]] | None = None):
        values = recipes or DEFAULT_RECIPES
        cleaned = {}
        for issue, actions in values.items():
            issue = str(issue)
            if not issue or not isinstance(actions, (list, tuple)):
                raise ValueError(f"Invalid repair recipe for {issue!r}")
            actions = tuple(str(action) for action in actions)
            unknown = set(actions) - TOOL_ACTIONS
            if unknown:
                raise ValueError(f"Unknown repair actions for {issue}: {sorted(unknown)}")
            if len(set(actions)) != len(actions):
                raise ValueError(f"Duplicate repair action for {issue}: {actions}")
            cleaned[issue] = actions
        self._recipes = cleaned

    @classmethod
    def from_json(cls, path: str | Path) -> "RepairRecipeLibrary":
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict):
            raise ValueError("Repair recipe config must be an object")
        values = document.get("recipes", document)
        if not isinstance(values, dict):
            raise ValueError("Repair recipe config must contain an object named recipes")
        return cls({issue: value.get("actions", value) if isinstance(value, dict) else value
                    for issue, value in values.items()})

    def actions_for(self, issue_types: Sequence[str]) -> tuple[str, ...]:
        actions = []
        for issue in issue_types:
            for action in self._recipes.get(str(issue), ()):
                if action not in actions:
                    actions.append(action)
        return tuple(actions)

    def to_dict(self) -> dict[str, list[str]]:
        return {issue: list(actions) for issue, actions in self._recipes.items()}
