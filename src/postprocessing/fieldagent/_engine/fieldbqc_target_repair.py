"""Achievement-oriented local Goal profile; shared v9 tools and safety checks.

Targets are development aspirations, not a guarantee or a literature standard.
No reference labels, baseline output, automatic rule completion, or new transport.
"""
from dataclasses import asdict, dataclass, fields
import math
import time

import fieldbqc_goal_repair as base
from fieldbqc_geometry import scene_metrics, to_parcels, vertices

VERSION = "goal-two-task-target-v10.0"


@dataclass(frozen=True)
class TargetConfig(base.GoalConfig):
    target_vertex_reduction: float = .40
    target_overlap_reduction: float = .99

    def __post_init__(self):
        # Validate the inherited configuration without passing target-only keys.
        base.GoalConfig(**{f.name: getattr(self, f.name) for f in fields(base.GoalConfig)})
        for key in ("target_vertex_reduction", "target_overlap_reduction"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Invalid development target {key}")


def goal_contract(config):
    return dict(
        goal_id=VERSION, tasks=["simplify_contour", "resolve_overlap"],
        objective="Meet BOTH measurable targets under the shared safety constraints; then improve further within budget",
        development_targets=dict(surviving_instance_vertex_reduction_at_least=config.target_vertex_reduction,
                                 overlap_excess_area_reduction_at_least=config.target_overlap_reduction),
        target_status="Aspirational development settings; infeasibility must be reported, never hidden",
        measurement="Vertex denominator is original vertices of current survivors; suppressed vertices are not simplification gains. Overlap denominator is frozen original overlap excess area.",
        quality_constraints={f.name: getattr(config, f.name) for f in fields(base.GoalConfig)},
        fixed_vertex_target=None, runtime_ground_truth=False,
        scope="A1 same registered local tools; no online discovery or automatic rule finish",
        uncertainty="Record review notes without permanently excluding targets from later safe trials",
        stop="If either target is unmet, an early stop is rejected while model rounds remain; budget exhaustion or final stop must report target_unmet, not successful optimization")


class TargetEngine(base.GoalEngine):
    def __init__(self, image, original, config=None):
        super().__init__(image, original, config or TargetConfig())

    def objective_status(self):
        initial = {p.identifier: p for p in self.initial}
        denominator = sum(vertices(initial[p.identifier].geometry) for p in self.current)
        current = scene_metrics(self.current)
        before = scene_metrics(self.raw)
        reduction = 1 - current["vertices"] / max(1, denominator)
        original_overlap = before["overlap_excess_area"]
        residual = max(0., current["overlap_excess_area"])
        overlap_reduction = max(0., 1 - residual / original_overlap) if original_overlap > 1e-7 else 1. if residual <= 1e-7 else 0.
        vertex_met = denominator > 0 and reduction + 1e-10 >= self.config.target_vertex_reduction
        overlap_met = residual <= max(1e-7, original_overlap * (1-self.config.target_overlap_reduction))
        return dict(vertex_reduction=reduction, surviving_original_vertices=denominator,
                    current_vertices=current["vertices"], original_vertices=before["vertices"],
                    suppressed_original_vertices=before["vertices"]-denominator,
                    original_overlap_excess_area=original_overlap, current_overlap_excess_area=residual,
                    overlap_reduction=overlap_reduction,
                    target_vertex_reduction=self.config.target_vertex_reduction,
                    target_overlap_reduction=self.config.target_overlap_reduction,
                    vertex_target_met=vertex_met, overlap_target_met=overlap_met,
                    both_targets_met=vertex_met and overlap_met,
                    vertex_reduction_gap=max(0., self.config.target_vertex_reduction-reduction),
                    overlap_area_gap=max(0., residual-original_overlap*(1-self.config.target_overlap_reduction)),
                    next_focus="resolve_overlap" if not overlap_met else "simplify_contour" if not vertex_met else "improve_feasible_solution")

    def trial(self, *args, **kwargs):
        # Review is a visible uncertainty note, not an irreversible search lock.
        notes = self.reviewed
        self.reviewed = set()
        try:
            return super().trial(*args, **kwargs)
        finally:
            self.reviewed = notes


TARGET_PROMPT = """You optimize agricultural parcel vectors toward TWO EXPLICIT GOALS.
Read objective_status EVERY round: reduce redundant vertices by the target
fraction AND reduce original overlap excess area by the target fraction.
These are development targets, NOT guarantees; never weaken or bypass validators,
invent tools, remove real parcels just to meet a number, or claim an unmet target.
Within those constraints, make substantial measurable progress, not just a small
safe edit followed by stop. Keep identity correctness and image-supported edges.
You have the SAME tools and tool-check/time budget as the rule baseline.
Use the budget efficiently: batches may cover all appropriate IDs; all checks are
counted. Compare directions/allocation/suppression where supported, prioritize
large conflicts, then simplify the surviving coverage. High containment alone is
not evidence of a duplicate. If a direction fragments a parcel, test a feasible
alternative before abandoning that conflict. Review notes do not block later trials.
For contours do not rely exclusively on DP at tolerance=1: compare allowed
tolerances and line/VW alternatives. After resolving overlaps, consider JOINT VW
on a valid coverage or suitable neighboring subsets if independent simplification
is rejected for shared-boundary changes. Do not repeat unchanged failed trials.
No mandatory rule finish or hidden compaction will rescue your result.
If targets are unmet, stop is rejected while another model round remains. Use
feedback to search alternatives; at the final budget explain residual obstacles.
Targets may be infeasible under the shared tools and checks; report that plainly.
""" + base.PROMPT


class TargetPlanner(base.GoalPlanner):
    def __init__(self, config, engine, cache):
        super().__init__(config, engine, cache)
        self.system_prompt = TARGET_PROMPT
        self.prompt_version = VERSION


def run_goal(engine, planner, progress=None):
    events, stop, failed = [], "model_call_budget", False
    for iteration in range(engine.config.model_calls):
        if engine.remaining() <= 0 or engine.generated >= engine.config.tool_budget:
            stop = "time_budget" if engine.remaining() <= 0 else "tool_budget"
            break
        context = dict(engine.inspect(), goal_contract=goal_contract(engine.config),
                       objective_status=engine.objective_status(), available_actions=[],
                       local_tool_index=[{k:c[k] for k in ("tool_id","task","description","parameters")} for c in base.TOOL_CARDS],
                       max_commands=engine.config.max_commands,
                       remaining_budget=dict(iterations=engine.config.model_calls-iteration,
                                             tools=engine.config.tool_budget-engine.generated,
                                             seconds=round(engine.remaining(),2)),
                       history=[{k:v for k,v in e.items() if k not in {"input_context","raw_output","usage","billed_usage","cache_key"}} for e in events[-18:]],
                       active_trial_ids=[t for t,(r,_) in engine.trials.items() if r["base_state"]==base.state_key(engine.current)],
                       review_ids=sorted(engine.reviewed), review_is_search_lock=False)
        try:
            receipt = planner.plan(context,engine.tools.image,to_parcels(engine.current,engine.tools.shape),engine.remaining())
        except Exception as error:
            record = dict(getattr(error,"record",{}))
            record.update(action="plan",iteration=iteration,api_request=True,api_attempts=1,
                          error=f"{type(error).__name__}: {error}",input_context=context)
            events.append(record)
            failed, stop = True, "planner_error"
            break
        receipt.update(action="plan",iteration=iteration,api_request=not receipt.get("cache_hit",False),api_attempts=int(not receipt.get("cache_hit",False)))
        events.append(receipt)
        packet = receipt["goal"]
        for c in packet["commands"]:
            event = dict(action=c["kind"],iteration=iteration,command=c,diagnosis=packet["diagnosis"],plan=packet["plan"])
            try:
                if c["kind"] == "stop":
                    status = engine.objective_status()
                    event["objective_status"] = status
                    if not status["both_targets_met"] and iteration+1 < engine.config.model_calls and engine.remaining()>0 and engine.generated<engine.config.tool_budget:
                        event.update(accepted=False,error="Targets unmet: early stop denied; inspect objective gaps and try safe alternatives. No automatic rule fallback.")
                        events.append(event)
                        break  # Return the denial as feedback in the next round.
                    event["accepted"] = True
                    events.append(event)
                    return events, "targets_met" if status["both_targets_met"] else "target_unmet_final_stop", False
                if c["kind"] == "inspect":
                    planner.focus_ids = c["target_ids"]
                    event["result"] = engine.inspect(c["target_ids"])
                elif c["kind"] == "find_tools":
                    event["result"] = engine.find_tools(c["query"] or "parcel contour overlap")
                elif c["kind"] in {"trial","apply"}:
                    if c["kind"]=="apply" and not (c["tool_id"] or "").startswith("contour_"):
                        raise ValueError("Overlap requires preview trial then explicit later commit")
                    record = engine.trial(c["tool_id"],c["target_ids"],c["tolerance"],c["keep_id"])
                    planner.preview_trials.append(record["trial_id"])
                    event["result"] = record
                    if c["kind"]=="apply" and record["accept"]:
                        event["commit"] = engine.commit(record["trial_id"],c["reason"])
                        event["accepted"] = True
                elif c["kind"] == "commit":
                    if c["trial_id"] not in context["active_trial_ids"]:
                        raise ValueError("Commit must reference an observed non-stale trial")
                    event["result"] = engine.commit(c["trial_id"],c["reason"])
                    event["accepted"] = True
                elif c["kind"] == "restore":
                    event["result"] = engine.restore()
                    event["accepted"] = True
                elif c["kind"] == "review":
                    if not c["target_ids"]:
                        raise ValueError("Review requires explicit IDs")
                    engine.reviewed.update(c["target_ids"])
                    event["result"] = dict(review_ids=sorted(engine.reviewed),review_is_search_lock=False)
            except Exception as error:
                event.update(accepted=False,error=f"{type(error).__name__}: {error}")
            event["objective_status"] = engine.objective_status()
            events.append(event)
            if progress:
                progress(iteration,event)
        if progress:
            progress(iteration,dict(action="round_complete",scene=scene_metrics(engine.current),objective_status=engine.objective_status()))
    return events, stop, failed


def run(engine, method, planner=None, progress=None):
    if method != "A1-GOAL-LOCAL":
        output, trace = base.run(engine,method,planner,progress)
    else:
        if planner is None:
            raise ValueError("Target branch requires a real planner")
        events, stop, failed = run_goal(engine,planner,progress)
        current = {p.identifier:p for p in engine.current}
        initial = {p.identifier:p for p in engine.initial}
        removed = {r["original_id"]:r for r in engine.lineage}
        denominator = sum(vertices(initial[i].geometry) for i in current)
        final = scene_metrics(engine.current)
        trace = dict(method=method,initial=scene_metrics(engine.raw),normalized_initial=scene_metrics(engine.initial),
                     final=final,action_trace=events,stop_reason=stop,planner_failed=failed,
                     review_ids=sorted(engine.reviewed),
                     lineage=[removed.get(p.identifier,dict(original_id=p.identifier,active=p.identifier in current,replaced_by=None)) for p in engine.initial],
                     candidate_evaluations=engine.generated,postprocess_seconds=time.monotonic()-engine.started,
                     survivor_original_vertices=denominator,survivor_compression=1-final["vertices"]/max(1,denominator),
                     suppressed_original_vertices=sum(vertices(p.geometry) for p in engine.initial if p.identifier not in current),
                     initial_conflict_components=base.conflict_components(engine.initial))
        output = engine.current
    trace.update(version=VERSION,goal_contract=goal_contract(engine.config),objective_status=engine.objective_status(),
                 scope="target-driven local two-task development; same tools/validators/budgets as rule; no mandatory completion")
    return output, trace
