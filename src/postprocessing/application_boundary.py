"""GT-blind local tasks and task-specific, cumulative geometry checks."""
from dataclasses import dataclass, asdict, replace
import math
import numpy as np
from shapely.geometry import LineString, Point
from shapely.errors import GEOSException
from shapely.ops import unary_union
import dp_structure as legacy
from dp_correction import ImageEvidence, verify as common_verify
from fieldbqc_direct_edit import ring_records

TASKS = ('remove_redundancy', 'restore_corner', 'realign_edge')


@dataclass(frozen=True)
class TaskGates:
    # Development defaults in pixel units, NOT surveying accuracy standards.
    simplify_area_fraction: float = .05
    correction_area_fraction: float = .15
    correction_area_allowance_px2: float = 16.
    raw_area_fraction: float = .15
    raw_area_allowance_px2: float = 16.
    edge_search_radius_px: float = 2.
    edge_search_step_px: float = .5
    corner_angle_degrees: float = 155.
    edge_support_gain: float = .06

    def __post_init__(self):
        for k, v in asdict(self).items():
            if type(v) not in (float, int) or not math.isfinite(v) or v < 0:
                raise ValueError('Invalid task gate: ' + k)
        if self.edge_search_step_px <= 0 or self.edge_search_radius_px > 6 or self.corner_angle_degrees > 180:
            raise ValueError('Invalid search radius, step, or corner angle')
        if any(getattr(self, k) > 1 for k in ('simplify_area_fraction', 'correction_area_fraction', 'raw_area_fraction')):
            raise ValueError('Area fractions must be <=1')


def edge_search(path, evidence, gates):
    """Fit two interior controls around a supplied literal hint, endpoints fixed.

    Numeric gradient/contrast is a proposal heuristic, never a correctness oracle.
    Both visual and no-raster-input planners have the same executor.
    """
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 4 or not np.isfinite(points).all():
        raise ValueError('Search hint needs 2..4 finite coordinates')
    line = LineString(points)
    if line.length <= 1e-6:
        raise ValueError('Zero-length search hint')
    a, b = points[0], points[-1]
    direction = b-a
    if np.linalg.norm(direction) < 1e-6:
        raise ValueError('Coincident search endpoints')
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    controls = [np.array(line.interpolate(t, normalized=True).coords[0]) for t in (1/3, 2/3)]
    offsets = np.arange(-gates.edge_search_radius_px, gates.edge_search_radius_px+1e-8, gates.edge_search_step_px)
    best = None; evaluated = 0
    h, w = evidence.image.shape[:2]
    for x in offsets:
        for y in offsets:
            candidate = [a.tolist(), (controls[0]+x*normal).tolist(), (controls[1]+y*normal).tolist(), b.tolist()]
            if any(not (0 <= p[0] <= w and 0 <= p[1] <= h) for p in candidate):
                continue
            measured = evidence.measure(candidate)
            evaluated += 1
            score = measured['support'] - .015*(abs(x)+abs(y))
            rank = (score, -abs(x)-abs(y))
            if best is None or rank > best[0]:
                best = (rank, candidate, measured, [float(x), float(y)])
    if best is None:
        raise ValueError('No in-bounds edge-search path')
    return best[1], dict(tool='normal_control_search', hint_path=path,
        radius_px=gates.edge_search_radius_px, step_px=gates.edge_search_step_px,
        selected_offsets_px=best[3], evaluated_paths=evaluated, evidence=best[2], uses_image=True)


def _corner_angle(segment):
    a, b = np.asarray(segment['endpoints'], float)
    far = np.asarray(segment['raw_arc'][segment['farthest_index']], float)
    u, v = a-far, b-far
    norm = np.linalg.norm(u)*np.linalg.norm(v)
    return 180. if norm < 1e-9 else float(np.degrees(np.arccos(np.clip(np.dot(u,v)/norm, -1, 1))))


def diagnose(raw, dp, image, config, gates):
    evidence = ImageEvidence(image)
    observation = legacy.detect(raw, dp, evidence, config)
    rows = observation['segments']; search_calls=[]
    for row in rows:
        row['legacy_task'] = row['task']
        row['suspected_task'] = ('remove_redundancy' if row['task']=='remove_DP_redundancy' else
            'restore_corner' if _corner_angle(row)<gates.corner_angle_degrees else 'realign_edge')
        row['corner_angle_degrees'] = _corner_angle(row)
        row['risk_evidence'] = dict(raw_DP_disagreement_px=row['max_deviation_px'],
            image_support_difference=row.get('raw_support',0)-row.get('dp_support',0))
        row['risk_is_calibrated_probability'] = False
    # A raw/DP disagreement alone misses edges where BOTH are displaced.
    observed = {(r['instance_id'],r['ring_id'],r['dp_edge_index']) for r in rows if r['legacy_task']=='correct_edge'}
    for original, simple in zip(raw, dp):
        sources = {r['ring_id']:r['coordinates'] for r in ring_records(original.geometry)}
        for ring in ring_records(simple.geometry):
            coords = ring['coordinates']; source = sources.get(ring['ring_id'], [])
            lookup = {tuple(p):i for i,p in enumerate(source)}; n = len(source)
            if n < 3 or len(lookup)!=n or any(tuple(p) not in lookup for p in coords):
                continue
            anchors = [lookup[tuple(p)] for p in coords]
            if sum((b-a)%n for a,b in zip(anchors,anchors[1:]+anchors[:1])) != n:
                continue
            for k, a in enumerate(coords):
                b = coords[(k+1)%len(coords)]
                key = (simple.identifier, ring['ring_id'], k)
                if key in observed or LineString([a,b]).length < config.min_segment_length:
                    continue
                fitted, audit = edge_search([a,b], evidence, gates)
                before = evidence.measure([a,b])['support']; gain = audit['evidence']['support']-before
                search_calls.append(dict(instance_id=simple.identifier,ring_id=ring['ring_id'],dp_edge_index=k,
                    audit=audit,actual_path=fitted,support_gain=gain,risk_triggered=gain>=gates.edge_support_gain))
                if gain < gates.edge_support_gain:
                    continue
                si, ei = anchors[k], anchors[(k+1)%len(coords)]
                ids = [(si+j)%n for j in range((ei-si)%n+1)]
                arc = [source[j] for j in ids]; chord = LineString([a,b])
                devs = [Point(p).distance(chord) for p in arc]
                rows.append(dict(instance_id=simple.identifier,ring_id=ring['ring_id'],task='correct_edge',
                    legacy_task='correct_edge',suspected_task='realign_edge',dp_edge_index=k,dp_edge_indices=[k],
                    dp_anchor_path=[a,b],endpoints=[a,b],raw_arc=arc,raw_indices=ids,
                    farthest_index=int(np.argmax(devs)),max_deviation_px=max(devs),chord_length_px=chord.length,
                    dp_support=before,raw_support=evidence.measure(arc)['support'],removable_dp_points=0,
                    score=1.+gain,original_ring_vertices=n,dp_ring_vertices=len(coords),
                    risk_evidence=dict(edge_search_support_gain=gain,raw_DP_disagreement_px=max(devs)),
                    risk_is_calibrated_probability=False,search_proposal=fitted,search_audit=audit))
    # Balance task opportunities, rather than letting many windows of one type starve the others.
    ordered = sorted(rows,key=lambda r:(-r['score'],r['instance_id'],r['ring_id'],r['dp_edge_index']))
    queues = {task:[r for r in ordered if r['suspected_task']==task] for task in TASKS}
    for i, row in enumerate(ordered):
        row.update(segment_id=f'app_{i:04d}',selected=False,selection_reason='scene_segment_budget')
    selected=[]; used=set(); counts={}
    for i in range(max([len(q) for q in queues.values()]+[0])):
        for task in TASKS:
            if i >= len(queues[task]): continue
            r=queues[task][i]; uid=r['instance_id']
            edges={(uid,r['ring_id'],k) for k in r['dp_edge_indices']}
            if len(selected)>=config.max_segments: continue
            if counts.get(uid,0)>=config.max_segments_per_instance:
                r['selection_reason']='instance_segment_budget'; continue
            if edges & used:
                r['selection_reason']='overlapping_arc'; continue
            r.update(selected=True,selection_reason='task_balanced_risk_selection')
            selected.append(r); used.update(edges); counts[uid]=counts.get(uid,0)+1
    pools={}
    for row in selected:
        options=legacy.candidate_paths(row,evidence,config)
        if 'search_proposal' in row and all(o['path']!=row['search_proposal'] for o in options):
            options.append(dict(candidate_id=f'cand_{len(options):02d}',method='normal_control_search',
                path=row['search_proposal'],evidence=legacy.path_score(row['search_proposal'],row['dp_anchor_path'],evidence),
                tool_audit=row['search_audit']))
        pools[row['segment_id']]=options
    return dict(segments=ordered,selected=selected,detected=len(rows),selected_count=len(selected),
        selected_by_task={task:sum(r['suspected_task']==task for r in selected) for task in TASKS},
        skipped_rings=observation['skipped_rings'],runtime_GT_access=False,
        tool_audit=dict(raw_DP_arc_scan=True,DP_multi_edge_scan=True,normal_search_calls=search_calls,
            candidate_generation_segments=[s['segment_id'] for s in selected]),
        warning='These are unconfirmed risk hypotheses, NOT annotated defects'),pools


def verify(raw, dp, candidate, config, gates, ledger):
    # Keep topology, identities, displacement, point budget and new-overlap checks.
    result=common_verify(raw,dp,candidate,replace(config,max_area_change_dp=1e12,max_area_change_raw=1e12))
    if any('invalid_geometry' in r for r in result['reasons']) or len(candidate)!=len(dp):
        return result
    reasons=result['reasons']; checks=[]
    for a,b,c in zip(raw,dp,candidate):
        if c.geometry.equals_exact(b.geometry,0): continue
        edits=[e for e in ledger if e['instance_id']==c.identifier]
        if not edits:
            reasons.append(c.identifier+':unaudited_edit'); continue
        correcting=any(e['task']!='remove_redundancy' for e in edits)
        allowance=(gates.correction_area_allowance_px2+gates.correction_area_fraction*b.geometry.area
            if correcting else gates.simplify_area_fraction*b.geometry.area)
        # Symmetric difference prevents opposite signed area changes cancelling.
        difference=b.geometry.symmetric_difference(c.geometry).area
        raw_before=a.geometry.symmetric_difference(b.geometry).area
        raw_after=a.geometry.symmetric_difference(c.geometry).area
        raw_allowance=max(raw_before,gates.raw_area_fraction*a.geometry.area)+(
            gates.raw_area_allowance_px2 if correcting else 0.)
        corridor=unary_union([LineString(e['before_path']).buffer(config.max_shift_from_dp+1e-6) for e in edits])
        outside=b.geometry.symmetric_difference(c.geometry).difference(corridor).area
        checks.append(dict(instance_id=c.identifier,policy='correction' if correcting else 'simplification',
            symmetric_difference_DP_px2=difference,DP_allowance_px2=allowance,
            symmetric_difference_raw_px2=raw_after,raw_allowance_px2=raw_allowance,
            changed_area_outside_edit_corridor_px2=outside))
        if difference>allowance+1e-7: reasons.append(c.identifier+':task_area_budget_DP')
        if raw_after>raw_allowance+1e-7: reasons.append(c.identifier+':task_area_budget_raw')
        if outside>1e-5: reasons.append(c.identifier+':outside_local_edit_corridor')
    return dict(result,accept=not reasons,reasons=reasons,task_checks=checks,
        semantic_correctness_certified=False)


def preview(raw, dp, current, segment, candidate, operation, shape, config, gates, ledger):
    try:
        if operation not in TASKS: raise ValueError('Unknown operation')
        if operation=='remove_redundancy' and len(candidate['path'])>=len(segment['dp_anchor_path']):
            raise ValueError('A redundancy deletion must reduce path vertices')
        if operation=='restore_corner' and len(candidate['path'])<3:
            raise ValueError('Corner restoration needs an interior control')
        proposed=legacy.apply_path(current,segment,candidate['path'],shape,config)
        if all(a.geometry.equals_exact(b.geometry,0) for a,b in zip(current,proposed)):
            raise ValueError('Proposed edit is unchanged')
        record=dict(instance_id=segment['instance_id'],task=operation,segment_id=segment['segment_id'],
            before_path=segment['dp_anchor_path'],after_path=candidate['path'])
        verdict=verify(raw,dp,proposed,config,gates,ledger+[record])
        if any('invalid_geometry' in r for r in verdict['reasons']):
            return proposed,verdict,record
        # Also bound this edit's footprint, not only the cumulative mixed-task budget.
        ix=next(i for i,p in enumerate(current) if p.identifier==segment['instance_id'])
        footprint=current[ix].geometry.symmetric_difference(proposed[ix].geometry).area
        allowance=(gates.simplify_area_fraction*dp[ix].geometry.area if operation=='remove_redundancy' else
            gates.correction_area_allowance_px2+gates.correction_area_fraction*dp[ix].geometry.area)
        if footprint>allowance+1e-7:
            verdict['reasons'].append(segment['instance_id']+':single_edit_area_budget'); verdict['accept']=False
        verdict.update(single_edit_footprint_px2=footprint,single_edit_allowance_px2=allowance)
        return proposed,verdict,record
    except (ValueError,KeyError,TypeError,GEOSException) as error:
        return None,dict(accept=False,reasons=[str(error)],semantic_correctness_certified=False),None
