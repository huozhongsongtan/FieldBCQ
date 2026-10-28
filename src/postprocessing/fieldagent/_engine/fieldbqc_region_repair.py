"""Region strategies over shared verified tools; runtime never reads labels.

A model proposes strategies/continuous parameters, local branches execute them,
and a later response commits an observed branch. Internal trials are retained
and charged to the same global check budget as the ordinary search baseline.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import json
import math
import time

import numpy as np
import shapely
from shapely.ops import unary_union
from shapely.strtree import STRtree

import fieldbqc_goal_repair as base
from fieldbqc_geometry import polygons, scene_metrics, to_parcels, vertices
from fieldbqc_planner import APIPlanner
from fieldbqc_target_repair import TargetConfig, TargetEngine, goal_contract as target_contract
from fieldbqc_vertex_optimization import contact_groups

VERSION = "goal-region-policy-v11.2"
ALGORITHMS = ("contour_dp", "contour_vw", "contour_lines")
STRATEGIES = ("confidence", "edge", "allocate", "priority", "none")


@dataclass(frozen=True)
class RegionConfig(TargetConfig):
    max_strategies: int = 4
    max_tolerances: int = 4
    max_region_checks: int = 192
    tool_preflight: bool = True
    use_trial_cache: bool = True
    preserve_all_ids: bool = False

    def __post_init__(self):
        super().__post_init__()
        for name in ("max_strategies", "max_tolerances", "max_region_checks"):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f"Invalid {name}")
        for name in ("tool_preflight","use_trial_cache","preserve_all_ids"):
            if type(getattr(self,name)) is not bool:
                raise ValueError(f"Invalid {name}")


def goal_contract(config):
    contract = target_contract(config)
    contract.update(goal_id=VERSION,
        policy="Generate region-specific tool combinations, numeric tolerances, ownership priorities and boundary anchors. Execute common local search then inspect real feedback.",
        feedback="Objective gaps, complete candidate changes, rejection counts, region evidence and preview",
        preserve_all_original_ids=config.preserve_all_ids,
        scope="Local registered tools, shared batch executor, no online discovery or rule fallback",
        stop="Targets met, budget exhausted, or explicitly explained search stall; unmet targets remain disclosed")
    return contract


def ring_evidence(geometry):
    lengths, deviations = [], []
    for p in polygons(geometry):
        for ring in (p.exterior, *p.interiors):
            xy = np.asarray(ring.coords[:-1], dtype=float)
            if len(xy)<3:
                continue
            a, b = xy-np.roll(xy,1,axis=0), np.roll(xy,-1,axis=0)-xy
            la, lb = np.linalg.norm(a,axis=1), np.linalg.norm(b,axis=1)
            angles = np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1)/np.maximum(la*lb,1e-12),-1,1)))
            lengths.extend(lb.tolist())
            deviations.extend(angles.tolist())
    n = max(1,len(lengths))
    return dict(perimeter=float(geometry.length),vertices=vertices(geometry),
                short_edge_fraction=sum(x<=1.01 for x in lengths)/n,
                near_straight_fraction=sum(x<=10 for x in deviations)/n,
                sharp_turn_fraction=sum(x>=60 for x in deviations)/n,
                mean_turn_degrees=float(np.mean(deviations)) if deviations else 0.,
                thresholds_note="Diagnostic pixel/degrees descriptors, not calibrated semantic labels")


def linked_regions(items):
    """Area overlap OR shared edge defines a region; include isolated instances."""
    if not items:
        return []
    geoms=[p.geometry for p in items]
    tree=STRtree(geoms)
    parent=list(range(len(items)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    links=[]
    for i,g in enumerate(geoms):
        for j in sorted(int(v) for v in tree.query(g)):
            if j<=i:
                continue
            overlap=float(g.intersection(geoms[j]).area)
            shared=float(g.boundary.intersection(geoms[j].boundary).length)
            if overlap>1e-8 or shared>1e-7:
                parent[root(j)]=root(i)
                links.append(dict(target_ids=[items[i].identifier,items[j].identifier],
                                  overlap_area=overlap,shared_boundary_length=shared))
    groups={}
    for i,p in enumerate(items):
        groups.setdefault(root(i),[]).append(p.identifier)
    result=[]
    lookup={p.identifier:p for p in items}
    for ids in groups.values():
        ids=sorted(ids)
        state=scene_metrics([lookup[i] for i in ids])
        result.append(dict(region_id="r-"+hashlib.sha256("\0".join(ids).encode()).hexdigest()[:10],
                           target_ids=ids,bbox=list(unary_union([lookup[i].geometry for i in ids]).bounds),
                           metrics=state,links=[p for p in links if set(p["target_ids"])<=set(ids)]))
    return sorted(result,key=lambda r:(-r["metrics"]["overlap_excess_area"],-r["metrics"]["vertices"],r["region_id"]))


ANCHOR = dict(type="object",additionalProperties=False,properties={
    "parcel_id":dict(type="string"),"x":dict(type="number"),"y":dict(type="number"),
    "max_shift":dict(type="number")},required=["parcel_id","x","y","max_shift"])
POLICY = dict(type="object",additionalProperties=False,properties={
    "conflict_strategies":dict(type="array",items=dict(type="string",enum=list(STRATEGIES))),
    "keep_order":dict(type="array",items=dict(type="string")),
    "algorithms":dict(type="array",items=dict(type="string",enum=list(ALGORITHMS))),
    "tolerances":dict(type="array",items=dict(type="number")),
    "joint":dict(type="boolean"),"allow_suppression":dict(type="boolean"),
    "preserve_ids":dict(type="array",items=dict(type="string")),
    "anchors":dict(type="array",items=ANCHOR),
    "rank":dict(type="string",enum=["balanced","compression","boundary"]),
    "max_checks":dict(type="integer")},
    required=["conflict_strategies","keep_order","algorithms","tolerances","joint","allow_suppression","preserve_ids","anchors","rank","max_checks"])
COMMAND = dict(type="object",additionalProperties=False,properties={
    "kind":dict(type="string",enum=["propose","commit","inspect","review","restore","stop"]),
    "target_ids":dict(type="array",items=dict(type="string")),
    "policy":dict(anyOf=[POLICY,dict(type="null")]),
    "trial_id":dict(type=["string","null"]),"reason":dict(type="string")},
    required=["kind","target_ids","policy","trial_id","reason"])
PACKET = dict(type="object",additionalProperties=False,properties={
    "diagnosis":dict(type="string"),"plan":dict(type="string"),
    "commands":dict(type="array",items=COMMAND)},required=["diagnosis","plan","commands"])


def validate_policy(policy, ids, config, known_ids=None):
    if not isinstance(policy,dict) or set(policy)!=set(POLICY["required"]):
        raise ValueError("Expected complete region policy")
    for name,allowed,limit in (("conflict_strategies",STRATEGIES,config.max_strategies),
                               ("algorithms",ALGORITHMS,len(ALGORITHMS))):
        values=policy[name]
        if not isinstance(values,list) or not values or len(values)>limit or any(not isinstance(x,str) or x not in allowed for x in values) or len(set(values))!=len(values):
            raise ValueError(f"Invalid {name}")
    values=policy["tolerances"]
    if not isinstance(values,list) or not 1<=len(values)<=config.max_tolerances:
        raise ValueError("Invalid tolerance search")
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<=config.max_boundary_shift for v in values) or len(set(values))!=len(values):
        raise ValueError("Tolerance outside shared limits")
    for name in ("keep_order","preserve_ids"):
        v=policy[name]
        allowed_ids=known_ids if name=="preserve_ids" and known_ids is not None else ids
        if not isinstance(v,list) or any(not isinstance(i,str) for i in v) or len(set(v))!=len(v) or not set(v)<=set(allowed_ids):
            raise ValueError(f"Invalid {name}")
    if any(type(policy[k]) is not bool for k in ("joint","allow_suppression")):
        raise ValueError("Expected boolean switches")
    if policy["rank"] not in {"balanced","compression","boundary"}:
        raise ValueError("Unknown ranking policy")
    if type(policy["max_checks"]) is not int or not 1<=policy["max_checks"]<=config.max_region_checks:
        raise ValueError("Invalid internal check cap")
    if not isinstance(policy["anchors"],list) or len(policy["anchors"])>32:
        raise ValueError("Invalid anchors")
    for a in policy["anchors"]:
        if not isinstance(a,dict) or set(a)!=set(ANCHOR["required"]) or a["parcel_id"] not in ids:
            raise ValueError("Invalid anchor identity")
        if any(isinstance(a[k],bool) or not isinstance(a[k],(int,float)) or not math.isfinite(a[k]) for k in ("x","y","max_shift")) or not 0<=a["max_shift"]<=config.max_boundary_shift:
            raise ValueError("Invalid anchor coordinates/shift")


def validate_packet(packet, context):
    if not isinstance(packet,dict) or set(packet)!={"diagnosis","plan","commands"}:
        raise ValueError("Expected diagnosis/plan/commands")
    if any(not isinstance(packet[k],str) or not 1<=len(packet[k])<=3000 for k in ("diagnosis","plan")):
        raise ValueError("Observable diagnosis and plan required")
    commands=packet["commands"]
    if not isinstance(commands,list) or not 1<=len(commands)<=context["max_commands"]:
        raise ValueError("Invalid command count")
    known={x["parcel_id"] for x in context["instances"]}
    config=RegionConfig(**context["config"])
    for c in commands:
        if not isinstance(c,dict) or set(c)!=set(COMMAND["required"]) or c["kind"] not in COMMAND["properties"]["kind"]["enum"]:
            raise ValueError("Invalid region command")
        ids=c["target_ids"]
        if not isinstance(ids,list) or any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids) or not set(ids)<=known:
            raise ValueError("Unknown/duplicate target")
        if not isinstance(c["reason"],str) or not 1<=len(c["reason"])<=2000:
            raise ValueError("Reason required")
        if c["kind"]=="propose":
            validate_policy(c["policy"],ids or sorted(known),config,known_ids=known)
        elif c["policy"] is not None:
            raise ValueError("Only propose accepts policy")
        if c["kind"]=="commit":
            if not isinstance(c["trial_id"],str):
                raise ValueError("Commit needs observed proposal ID")
        elif c["trial_id"] is not None:
            raise ValueError("Only commit accepts trial_id")
    return packet


def default_policy(config):
    return dict(conflict_strategies=["confidence","edge","allocate"],keep_order=[],
                algorithms=list(ALGORITHMS),tolerances=[.5*config.max_boundary_shift,config.max_boundary_shift],
                joint=True,allow_suppression=not config.preserve_all_ids,preserve_ids=[],anchors=[],rank="balanced",
                max_checks=min(config.tool_budget,config.max_region_checks))


class RegionEngine(TargetEngine):
    def __init__(self,image,original,config=None):
        super().__init__(image,original,config or RegionConfig())
        self.proposals={}
        self.policy_searches=[]
        # Shared only within this image/controller, including speculative forks.
        self.trial_cache={}
        self.eligibility_cache={}
        self.execution_stats=Counter()
        self.search_requests=[]

    def evidence(self):
        raw=self.inspect()
        extra={p.identifier:ring_evidence(p.geometry) for p in self.current}
        for row in raw["instances"]:
            row["boundary_features"]=extra[row["parcel_id"]]
        raw.update(regions=linked_regions(self.current),
                   probability_map_available=False,
                   evidence_provenance=dict(image="original BGR imagery",confidence="saved instance score, NOT pixel probability",
                                            geometry="current pixel-cell vectors",edge="Canny image support, not ground-truth boundary"))
        return raw

    def fork(self):
        # Same immutable image/tools and absolute deadline; no new time allowance.
        child=RegionEngine.__new__(RegionEngine)
        child.__dict__=dict(self.__dict__)
        child.current=list(self.current)
        child.anchors=list(self.anchors)
        child.lineage=deepcopy(self.lineage)
        child.reviewed=set(self.reviewed)
        child.checkpoints=[]
        child.trials={}
        child.proposals={}
        child.policy_searches=[]
        return child

    def trial(self,tool,ids,tolerance=None,keep_id=None):
        r=super().trial(tool,ids,tolerance,keep_id)
        # Additional model-requested anchors/preserved identities are stricter
        # than common gates; they never relax a base validation condition.
        candidate=self.trials[r["trial_id"]][1]
        policy=getattr(self,"active_policy",None)
        reasons=[]
        if self.config.preserve_all_ids and {p.identifier for p in candidate}!={p.identifier for p in self.current}:
            reasons.append("global_identity_protection")
        if policy:
            from shapely.geometry import Point
            new={p.identifier:p for p in candidate}
            for uid in policy["preserve_ids"]:
                if uid not in new:
                    reasons.append("policy_preserved_identity_removed")
            for a in policy["anchors"]:
                p=new.get(a["parcel_id"])
                if p is None or p.geometry.boundary.distance(Point(a["x"],a["y"]))>a["max_shift"]+1e-7:
                    reasons.append("policy_anchor_displaced")
        if reasons:
            r.update(accept=False,candidate=scene_metrics(self.current))
            r["details"].append(dict(target_ids=ids,accept=False,reasons=sorted(set(reasons)),metrics={}))
            self.trials[r["trial_id"]]=(r,list(self.current))
        return r

    def trial_key(self,tool,ids,tolerance,keep_id):
        policy=getattr(self,"active_policy",{})
        # Geometry alone is insufficient: cumulative drift gates depend on anchors
        # and model constraints can change between otherwise identical requests.
        values=dict(state=base.state_key(self.current),anchors=base.state_key(self.anchors),
                    tool=tool,ids=list(ids),tolerance=tolerance,keep_id=keep_id,
                    preserve_ids=sorted(policy.get("preserve_ids",[])),
                    boundary_anchors=policy.get("anchors",[]))
        return hashlib.sha256(json.dumps(values,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def eligibility(self,tool,ids):
        if tool=="overlap_suppress" and self.config.preserve_all_ids:
            return False,"global_identity_protection"
        if tool!="contour_vw" or len(ids)<2:
            return True,None
        key=(base.state_key(self.current),tuple(ids))
        if key not in self.eligibility_cache:
            lookup={p.identifier:p for p in self.current}
            self.execution_stats["coverage_precondition_checks"]+=1
            valid=bool(shapely.coverage_is_valid(np.asarray([lookup[i].geometry for i in ids],dtype=object)))
            self.eligibility_cache[key]=(valid,None if valid else "requires_valid_noded_coverage")
        return self.eligibility_cache[key]

    def commit(self,trial_id,reason):
        if trial_id not in self.proposals:
            return super().commit(trial_id,reason)
        r,candidate=self.trials[trial_id]
        if r["base_state"]!=base.state_key(self.current) or not r["accept"]:
            raise ValueError("Stale or non-improving region proposal")
        self.checkpoints.append((list(self.current),list(self.anchors),deepcopy(self.lineage)))
        anchors,lineage=self.proposals[trial_id]
        self.current=list(candidate)
        self.anchors=list(anchors)
        self.lineage=deepcopy(lineage)
        r["committed_to_scene"]=True
        for internal_id in r["internal_trials"]:
            internal=self.trials[internal_id][0]
            if internal.get("committed_in_branch"):
                internal["committed_to_scene"]=True
        return dict(committed=True,trial_id=trial_id,tool_id="region_policy",reason=reason,
                    scene=scene_metrics(self.current),objective_status=self.objective_status())

    def _rank(self,child,mode):
        status=child.objective_status()
        vgap=status["vertex_reduction_gap"]/child.config.target_vertex_reduction
        egap=max(0.,child.config.target_overlap_reduction-status["overlap_reduction"])/child.config.target_overlap_reduction
        edge=sum(child.tools.edge_alignment(child.tools.mask(p)) for p in child.current)/max(1,len(child.current))
        nratio=1-status["vertex_reduction"]
        eratio=status["current_overlap_excess_area"]/max(1.,status["original_overlap_excess_area"])
        if mode=="boundary":
            return (vgap+egap,-edge,eratio,nratio)
        if mode=="compression":
            return (vgap+egap,nratio,eratio,-edge)
        return (vgap+egap,eratio+nratio,-edge)

    def propose(self,ids,policy):
        ids=list(ids) or [p.identifier for p in self.current]
        current={p.identifier:p for p in self.current}
        validate_policy(policy,ids,self.config,known_ids=current)
        if not ids or not set(ids)<=set(current):
            raise ValueError("Proposal needs active target IDs")
        from shapely.geometry import Point
        for a in policy["anchors"]:
            point=Point(a["x"],a["y"])
            if not 0<=a["x"]<=self.tools.shape[1] or not 0<=a["y"]<=self.tools.shape[0] or current[a["parcel_id"]].geometry.boundary.distance(point)>self.config.max_boundary_shift+1e-7:
                raise ValueError("Anchor must refer to an observable current boundary")
        if self.generated>=self.config.tool_budget or self.remaining()<=0:
            raise RuntimeError("Search budget exhausted")
        before=scene_metrics(self.current)
        origin=base.state_key(self.current)
        start=self.generated
        cap=min(self.config.tool_budget,start+policy["max_checks"])
        search_id=f"s{len(self.policy_searches):04d}"
        stats_before=dict(self.execution_stats)
        summaries=[]
        for index,strategy in enumerate(policy["conflict_strategies"]):
            if self.generated>=cap or self.remaining()<=0:
                break
            child=self.fork()
            branch_start=self.generated
            child.active_policy=deepcopy(policy)
            local_cap=self.generated+max(1,(cap-self.generated)//(len(policy["conflict_strategies"])-index))
            operations=[]
            request_start=len(self.search_requests)
            def attempt(tool,targets,tolerance=None,keep=None):
                if self.generated>=local_cap or self.remaining()<=0:
                    return None
                request=dict(search_id=search_id,strategy=strategy,tool_id=tool,target_ids=list(targets),
                             tolerance=tolerance,keep_id=keep,base_state=base.state_key(child.current))
                self.execution_stats["requests"]+=1
                if self.config.tool_preflight:
                    eligible,reason=child.eligibility(tool,targets)
                    if not eligible:
                        self.execution_stats["preflight_skips"]+=1
                        self.search_requests.append(dict(request,status="inapplicable",reason=reason,new_tool_checks=0))
                        return None
                key=child.trial_key(tool,targets,tolerance,keep)
                child.generated=self.generated
                reused=self.config.use_trial_cache and key in self.trial_cache
                source_id=None
                if reused:
                    record,candidate,source_id=self.trial_cache[key]
                    record=deepcopy(record)
                    record["trial_id"]=f"t{len(child.trials):04d}"
                    child.trials[record["trial_id"]]=(record,list(candidate))
                    self.execution_stats["cache_hits"]+=1
                    checks=0
                else:
                    previous=self.generated
                    try:
                        record=child.trial(tool,targets,tolerance,keep)
                    finally:
                        self.generated=child.generated
                    checks=self.generated-previous
                    self.execution_stats["new_tool_checks"]+=checks
                internal_id=f"t{len(self.trials):05d}"
                if self.config.use_trial_cache and not reused:
                    self.trial_cache[key]=(deepcopy(record),list(child.trials[record["trial_id"]][1]),internal_id)
                saved=deepcopy(record)
                saved.update(trial_id=internal_id,internal_trial=True,search_id=search_id,strategy=strategy,
                             committed_in_branch=False,cache_hit=reused,cache_source_trial_id=source_id,
                             trial_cache_key=key,new_tool_checks=checks)
                self.trials[internal_id]=(saved,list(child.trials[record["trial_id"]][1]))
                self.search_requests.append(dict(request,status="cache_hit" if reused else "executed",trial_id=internal_id,
                                                 cache_source_trial_id=source_id,new_tool_checks=checks))
                operations.append(internal_id)
                return record,internal_id
            def commit_attempt(result):
                if result:
                    record,internal_id=result
                    previous_lineage_count=len(child.lineage)
                    child.commit(record["trial_id"],"Local verified step within speculative region strategy")
                    for row in child.lineage[previous_lineage_count:]:
                        row["trial_id"]=internal_id
                    self.trials[internal_id][0]["committed_in_branch"]=True
            # Leave a declared fraction for contour search in each branch.
            conflict_cap=self.generated+int(.65*(local_cap-self.generated))
            seen=set()
            if strategy!="none":
                while self.generated<conflict_cap and child.remaining()>0:
                    pairs=[p for p in base.overlap_pairs(child.current) if set(p["target_ids"])<=set(ids) and tuple(p["target_ids"]) not in seen]
                    if not pairs:
                        break
                    pair=pairs[0]; pair_ids=pair["target_ids"]
                    seen.add(tuple(pair_ids))
                    lookup={p.identifier:p for p in child.current}
                    def preference(uid):
                        if strategy=="priority" and uid in policy["keep_order"]:
                            return (2,-policy["keep_order"].index(uid),uid)
                        edge=child.tools.edge_alignment(child.tools.mask(lookup[uid]))
                        return (1,edge if strategy=="edge" else lookup[uid].score,uid)
                    ordered=sorted(pair_ids,key=preference,reverse=True)
                    specs=[]
                    if policy["allow_suppression"] and pair["containment"]>=child.config.duplicate_containment:
                        specs.extend(("overlap_suppress",i) for i in ordered if next(u for u in pair_ids if u!=i) not in policy["preserve_ids"])
                    if strategy=="allocate":
                        specs.append(("overlap_allocate",None))
                    specs.extend(("overlap_clip",i) for i in ordered)
                    if strategy!="allocate":
                        specs.append(("overlap_allocate",None))
                    # Rule and model have the same preference search primitive.
                    # Try the preferred feasible direction, then alternatives.
                    for tool,keep in specs:
                        if self.generated>=conflict_cap:
                            break
                        result=attempt(tool,pair_ids,keep=keep)
                        if result and result[0]["accept"]:
                            commit_attempt(result)
                            break
            active=[p for p in child.current if p.identifier in ids]
            if policy["joint"]:
                groups=[g for g in contact_groups(active) if len(g)>1]
                for group in groups:
                    options=[]
                    for tolerance in sorted(policy["tolerances"],reverse=True):
                        result=attempt("contour_vw",group,tolerance)
                        if result and result[0]["accept"]:
                            options.append(result)
                    if options:
                        commit_attempt(min(options,key=lambda r:(r[0]["candidate"]["vertices"],r[0]["tolerance"])))
            # Search all requested algorithms/parameters for each surviving ID.
            # No mandatory trailing simplification exists outside this policy.
            for p in sorted((p for p in child.current if p.identifier in ids),key=lambda p:(-vertices(p.geometry),p.identifier)):
                options=[]
                for algorithm in policy["algorithms"]:
                    for tolerance in sorted(policy["tolerances"],reverse=True):
                        result=attempt(algorithm,[p.identifier],tolerance)
                        if result and result[0]["accept"]:
                            options.append(result)
                if options:
                    commit_attempt(min(options,key=lambda r:(r[0]["candidate"]["vertices"],r[0]["tolerance"])))
                if self.generated>=local_cap or self.remaining()<=0:
                    break
            final=scene_metrics(child.current)
            # A valid branch can increase vertex count while fixing ownership;
            # require overlap OR survivor-compression progress, not total ID loss.
            initial_status=self.objective_status(); status=child.objective_status()
            accept=(final["overlap_excess_area"]<before["overlap_excess_area"]-1e-7 or status["vertex_reduction"]>initial_status["vertex_reduction"]+1e-10)
            tid=f"t{len(self.trials):05d}"
            rejection_counts=Counter(z for t in operations for d in self.trials[t][0]["details"] for z in d.get("reasons",[]) if not d.get("accept"))
            record=dict(trial_id=tid,tool_id="region_policy",target_ids=ids,tolerance=None,keep_id=None,
                        base_state=origin,accept=accept,before=before,candidate=final,details=[],
                        search_id=search_id,strategy=strategy,policy=deepcopy(policy),
                        objective_status=status,internal_trials=operations,internal_checks=self.generated-branch_start,
                        internal_requests=len(self.search_requests)-request_start,
                        preflight_reasons=dict(Counter(r["reason"] for r in self.search_requests[request_start:] if r["status"]=="inapplicable")),
                        rejection_counts=dict(rejection_counts),ranking=list(self._rank(child,policy["rank"])),
                        branch_actions_not_committed_to_scene=True)
            self.trials[tid]=(record,list(child.current))
            self.proposals[tid]=(list(child.anchors),deepcopy(child.lineage))
            summaries.append(record)
        result=dict(search_id=search_id,target_ids=ids,base_state=origin,policy=deepcopy(policy),
                    checks_used=self.generated-start,candidates=summaries,
                    objective_status_before=self.objective_status(),
                    execution_stats={k:v-stats_before.get(k,0) for k,v in self.execution_stats.items()},
                    no_runtime_reference=True,search_cost_included_in_global_budget=True)
        self.policy_searches.append(result)
        return result


PROMPT = """You control region-adaptive agricultural parcel VECTOR postprocessing.
Use real imagery, region/boundary diagnostics and objective_status. Meet both
development targets under the SAME hard geometry/identity gates and total tool
budget as ordinary search. There is no online discovery, hidden rule finish,
pixel-probability map or ground truth. Instance scores are not pixel probabilities.
If config.preserve_all_ids is true, every original identity is protected by a hard
global gate. Do not request suppression. Preflight rejections and cache reuse are
reported separately; only new low-level checks consume the common check budget.
Generate REGION POLICIES, not one low-level operation per model round. propose
accepts target_ids (empty=all active IDs) and a complete policy. Choose tool
combinations, continuous numeric tolerances, ownership strategies and keep_order,
shared-boundary search, preserved identities and optional observed boundary anchors.
Local search executes your actual policy and returns whole-scene candidates,
rejections and previews. Each conflict strategy starts from the SAME current state;
its internal verified changes are speculative, not committed to the scene.
confidence/edge strategies prefer feasible ownership in that order; allocate
prefers allocation; priority uses your keep_order; none leaves conflicts alone.
High containment is NOT semantic proof of a duplicate. allow_suppression is your
explicit choice and preserve_ids prevent removal, not harmless simplification.
preserve_ids may include any active scene ID, even outside target_ids. keep_order
and boundary anchors must refer only to this proposal's target_ids (all if empty).
Anchor x/y refer to pixel-cell coordinates on observable boundaries; max_shift is
an additional bound, never a relaxation. Omit invented evidence by using [].
max_checks includes ALL branch checks and must fit the remaining global budget.
One broad propose can cover many IDs; target subsets allow region-specific
parameters. Divide checks between regions when necessary, but cover the scene.
Return diagnosis/plan/commands in schema. Other than propose, policy must be null.
commit needs a region candidate trial_id observed BEFORE this response. Review
actual feedback and preview, then commit ONE candidate from a state. After any
commit/restore, all old-state candidates are stale; regenerate affected plans.
inspect chooses focus IDs for crops; review is a note, not a permanent search lock.
restore undoes one scene commit. stop is denied if targets are unmet, search budget
remains and rounds remain. At exhausted budget, report unmet gaps plainly.
Even with zero tool checks remaining you can still commit a previously observed
valid proposal; do not waste the last model round generating an uncommitted plan.
Prefer policies based on actual region shape/imagery, use rejection feedback to
change strategy or parameters, and preserve uncertain real structures.
"""


class RegionPlanner(base.GoalPlanner):
    def __init__(self,config,engine,cache):
        self.engine=engine
        self.focus_ids=[]
        self.preview_trials=[]
        APIPlanner.__init__(self,config,cache,schema=PACKET,system_prompt=PROMPT,
                            validator=validate_packet,prompt_version=VERSION,visualizer=self.images)

    def images(self,image,parcels):
        previous=self.focus_ids
        if not self.focus_ids:
            regions=linked_regions(self.engine.current)
            self.focus_ids=regions[0]["target_ids"] if regions else []
        try:
            return super().images(image,parcels)
        finally:
            self.focus_ids=previous


def context_for(engine,events,iteration):
    from dataclasses import asdict
    return dict(engine.evidence(),goal_contract=goal_contract(engine.config),config=asdict(engine.config),
                objective_status=engine.objective_status(),available_actions=[],
                max_commands=engine.config.max_commands,shared_tools=base.TOOL_CARDS,
                remaining_budget=dict(iterations=engine.config.model_calls-iteration,
                                      tools=engine.config.tool_budget-engine.generated,seconds=round(engine.remaining(),2)),
                history=[{k:v for k,v in e.items() if k not in {"input_context","raw_output","usage","billed_usage","cache_key"}} for e in events[-12:]],
                active_trial_ids=[t for t in engine.proposals if engine.trials[t][0]["accept"] and engine.trials[t][0]["base_state"]==base.state_key(engine.current)],
                review_ids=sorted(engine.reviewed))


def run_goal(engine,planner,progress=None):
    events=[]; stop="model_call_budget"; failed=False
    for iteration in range(engine.config.model_calls):
        if engine.remaining()<=0:
            stop="time_budget"; break
        # Tool budget limits execution, not reading feedback/committing a branch.
        context=context_for(engine,events,iteration)
        try:
            receipt=planner.plan(context,engine.tools.image,to_parcels(engine.current,engine.tools.shape),engine.remaining())
        except Exception as error:
            record=dict(getattr(error,"record",{}))
            record.update(action="plan",iteration=iteration,api_request=True,api_attempts=1,
                          error=f"{type(error).__name__}: {error}",input_context=context)
            events.append(record); failed=True; stop="planner_error"; break
        receipt.update(action="plan",iteration=iteration,api_request=not receipt.get("cache_hit",False),api_attempts=int(not receipt.get("cache_hit",False)))
        events.append(receipt)
        packet=receipt["goal"]
        for c in packet["commands"]:
            event=dict(action=c["kind"],iteration=iteration,command=c,diagnosis=packet["diagnosis"],plan=packet["plan"])
            try:
                if c["kind"]=="propose":
                    result=engine.propose(c["target_ids"],c["policy"])
                    event["result"]=result
                    # Preview top feasible branches; full numeric results persist.
                    candidates=sorted((r for r in result["candidates"] if r["accept"]),key=lambda r:r["ranking"])
                    planner.preview_trials=[r["trial_id"] for r in candidates[:2]]
                    planner.focus_ids=c["target_ids"]
                elif c["kind"]=="commit":
                    if c["trial_id"] not in context["active_trial_ids"]:
                        raise ValueError("Commit must reference an observed current-state region proposal")
                    event["result"]=engine.commit(c["trial_id"],c["reason"]); event["accepted"]=True
                elif c["kind"]=="inspect":
                    planner.focus_ids=c["target_ids"]
                    event["result"]=engine.inspect(c["target_ids"])
                elif c["kind"]=="review":
                    if not c["target_ids"]:
                        raise ValueError("Review requires explicit IDs")
                    engine.reviewed.update(c["target_ids"])
                    event["result"]=dict(review_ids=sorted(engine.reviewed))
                elif c["kind"]=="restore":
                    event["result"]=engine.restore(); event["accepted"]=True
                elif c["kind"]=="stop":
                    status=engine.objective_status()
                    if not status["both_targets_met"] and iteration+1<engine.config.model_calls and engine.generated<engine.config.tool_budget and engine.remaining()>0:
                        event.update(accepted=False,error="Targets unmet; remaining budget permits revised region policies")
                        event["objective_status"]=status; events.append(event); break
                    event["accepted"]=True; event["objective_status"]=status; events.append(event)
                    return events,"targets_met" if status["both_targets_met"] else "target_unmet_stop",False
            except Exception as error:
                event.update(accepted=False,error=f"{type(error).__name__}: {error}")
            event["objective_status"]=engine.objective_status()
            events.append(event)
            if progress:
                progress(iteration,event)
        if progress:
            progress(iteration,dict(action="round_complete",scene=scene_metrics(engine.current),objective_status=engine.objective_status()))
    return events,stop,failed


def run_rule(engine):
    """Same region executor, fixed policy and search ranking, no model or labels."""
    events=[]
    for region in linked_regions(engine.current):
        if engine.generated>=engine.config.tool_budget or engine.remaining()<=0:
            break
        policy=default_policy(engine.config)
        if engine.config.preserve_all_ids:
            policy["preserve_ids"]=[p.identifier for p in engine.current]
        policy["max_checks"]=min(engine.config.max_region_checks,engine.config.tool_budget-engine.generated)
        result=engine.propose(region["target_ids"],policy)
        events.append(dict(action="propose",result=result))
        options=[r for r in result["candidates"] if r["accept"]]
        if options:
            chosen=min(options,key=lambda r:r["ranking"])
            events.append(dict(action="commit",accepted=True,**engine.commit(chosen["trial_id"],"Frozen region search policy and objective ranking; no reference")))
    stop="tool_budget" if engine.generated>=engine.config.tool_budget else "time_budget" if engine.remaining()<=0 else "rule_complete"
    return events,stop,False


def run(engine,method,planner=None,progress=None):
    if method=="B0-RAW":
        engine.current=list(engine.raw); events,stop,failed=[],"raw",False
    elif method=="B0-NORM":
        events,stop,failed=[],"common_legalization_only",False
    elif method=="B1-RULE":
        events,stop,failed=run_rule(engine)
    elif method=="A1-GOAL-LOCAL" and planner is not None:
        events,stop,failed=run_goal(engine,planner,progress)
    else:
        raise ValueError("Unknown region method or missing real planner")
    return finish(engine,method,events,stop,failed)


def finish(engine,method,events,stop,failed):
    """Shared trace accounting; extensible profiles retain identical lineage rules."""
    lookup={p.identifier:p for p in engine.current}
    initial={p.identifier:p for p in engine.initial}
    removed={r["original_id"]:r for r in engine.lineage}
    denominator=sum(vertices(initial[i].geometry) for i in lookup)
    final=scene_metrics(engine.current)
    trace=dict(version=VERSION,method=method,initial=scene_metrics(engine.raw),normalized_initial=scene_metrics(engine.initial),
               final=final,goal_contract=goal_contract(engine.config),objective_status=engine.objective_status(),
               action_trace=events,stop_reason=stop,planner_failed=failed,review_ids=sorted(engine.reviewed),
               lineage=[removed.get(p.identifier,dict(original_id=p.identifier,active=p.identifier in lookup,replaced_by=None)) for p in engine.initial],
               candidate_evaluations=engine.generated,postprocess_seconds=time.monotonic()-engine.started,
               survivor_original_vertices=denominator,survivor_compression=1-final["vertices"]/max(1,denominator),
               suppressed_original_vertices=sum(vertices(p.geometry) for p in engine.initial if p.identifier not in lookup),
               initial_conflict_components=base.conflict_components(engine.initial),
               region_policies=len(engine.policy_searches),region_diagnostics=engine.evidence()["regions"],
               execution_stats=dict(engine.execution_stats),
               scope="Regional policy generation and real local batch search; no labels, online discovery or hidden finish")
    return engine.current,trace
