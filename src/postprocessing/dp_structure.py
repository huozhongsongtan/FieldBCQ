"""Delete residual DP vertices, or correct arcs, under the same frozen gates.

Arcs span multiple DP edges. Endpoint-only paths actually remove the observed
interior DP vertices; no second hidden simplifier is applied to model outputs.
"""
from dataclasses import dataclass
import json
import math
import numpy as np
from shapely.geometry import LineString, Point
import dp_correction as base
from dp_correction import initialize, ImageEvidence, verify, point_budget
from fieldbqc_direct_edit import ring_records,replace_ring
from fieldbqc_geometry import vertices


@dataclass(frozen=True)
class StructureConfig(base.CorrectionConfig):
    max_segments: int = 12
    max_segments_per_instance: int = 3
    max_merge_edges: int = 6
    rule_merge_support_loss: float = .06

    def __post_init__(self):
        super().__post_init__()
        if type(self.max_merge_edges) is not int or self.max_merge_edges<2:
            raise ValueError('max_merge_edges must be an integer >=2')


def detect(raw,dp,evidence,config):
    old=base.detect(raw,dp,evidence,config)
    rows=[];skipped=list(old['skipped_rings'])
    for r in old['segments']:
        r=dict(r,task='correct_edge',dp_anchor_path=r['endpoints'],
               dp_edge_indices=[r['dp_edge_index']],removable_dp_points=0)
        rows.append(r)
    for original,simple in zip(raw,dp):
        source_rings={r['ring_id']:r['coordinates'] for r in ring_records(original.geometry)}
        for ring in ring_records(simple.geometry):
            coords=ring['coordinates'];n=len(coords);source=source_rings.get(ring['ring_id'])
            if n<4 or source is None:continue
            lookup={tuple(p):i for i,p in enumerate(source)}
            if len(lookup)!=len(source) or any(tuple(p) not in lookup for p in coords):continue
            anchors=[lookup[tuple(p)] for p in coords];nr=len(source)
            if sum((b-a)%nr for a,b in zip(anchors,anchors[1:]+anchors[:1]))!=nr:continue
            for start in range(n):
                for edges in range(2,min(config.max_merge_edges,n-2)+1):
                    indices=[(start+j)%n for j in range(edges+1)]
                    path=[coords[j] for j in indices];a,b=path[0],path[-1]
                    chord=LineString([a,b])
                    if chord.length<6.:continue
                    deviation=max(Point(p).distance(chord) for p in path)
                    if deviation>config.max_shift_from_dp+1e-7:continue
                    si,ei=anchors[start],anchors[(start+edges)%n]
                    raw_ids=[(si+j)%nr for j in range((ei-si)%nr+1)]
                    arc=[source[j] for j in raw_ids]
                    raw_devs=[Point(p).distance(chord) for p in arc]
                    if max(raw_devs)>config.max_shift_from_raw+1e-7:continue
                    before=evidence.measure(path)['support'];after=evidence.measure([a,b])['support']
                    rows.append(dict(instance_id=simple.identifier,ring_id=ring['ring_id'],
                        task='remove_DP_redundancy',dp_edge_index=start,
                        dp_edge_indices=[(start+j)%n for j in range(edges)],
                        dp_anchor_path=path,dp_vertex_indices=indices,
                        indexed_dp_points=[dict(dp_index=j,original_index=anchors[j],xy=coords[j]) for j in indices],
                        endpoints=[a,b],raw_arc=arc,raw_indices=raw_ids,
                        max_deviation_px=max(raw_devs),max_DP_chord_deviation_px=deviation,
                        farthest_index=int(np.argmax(raw_devs)),chord_length_px=chord.length,
                        raw_support=evidence.measure(arc)['support'],dp_support=before,chord_support=after,
                        removable_dp_points=edges-1,score=3.*(edges-1)-deviation/max(config.max_shift_from_dp,1e-9)+max(0.,after-before),
                        original_ring_vertices=nr,dp_ring_vertices=n))
    rows.sort(key=lambda r:(-r['score'],r['instance_id'],r['ring_id'],r['dp_edge_index']))
    selected=[];counts={};used_edges=set()
    for i,row in enumerate(rows):
        row['segment_id']=f'seg_{i:04d}'
        uid=row['instance_id'];keys={(uid,row['ring_id'],edge) for edge in row['dp_edge_indices']}
        reason='selected'
        if len(selected)>=config.max_segments:reason='scene_segment_budget'
        elif counts.get(uid,0)>=config.max_segments_per_instance:reason='instance_segment_budget'
        elif keys & used_edges:reason='overlaps_selected_DP_arc'
        row['selected']=reason=='selected';row['selection_reason']=reason
        if row['selected']:
            selected.append(row);counts[uid]=counts.get(uid,0)+1;used_edges.update(keys)
    return dict(segments=rows,selected=selected,skipped_rings=skipped,detected=len(rows),selected_count=len(selected),
        detected_removal_arcs=sum(r['task']=='remove_DP_redundancy' for r in rows),
        selected_removal_arcs=sum(r['task']=='remove_DP_redundancy' for r in selected),
        selected_removable_DP_points=sum(r['removable_dp_points'] for r in selected),runtime_GT_access=False)


def path_score(path,reference,evidence):
    result=evidence.measure(path)
    displacement=LineString(path).hausdorff_distance(LineString(reference))
    result.update(point_change=len(path)-len(reference),displacement_px=displacement,
                  score=result['support']-.015*(len(path)-2)-.005*displacement)
    return result


def candidate_paths(segment,evidence,config):
    reference=segment['dp_anchor_path']
    if segment['task']!='remove_DP_redundancy':
        options=base.candidate_paths(segment,evidence,config)
        for c in options:c['evidence']=path_score(c['path'],reference,evidence)
        return options
    a,b=segment['endpoints']
    proposals=[('keep_DP_arc',reference),('merge_DP_edges_into_endpoint_chord',[a,b])]
    # Preserve one observed structural control point, if needed, rather than
    # restoring every original staircase point. Also shared with the rule arm.
    for point in reference[1:-1]:proposals.append(('retain_one_DP_control_point',[a,point,b]))
    result=[];seen=set()
    for method,path in proposals:
        key=json.dumps(path)
        if key in seen:continue
        seen.add(key)
        result.append(dict(candidate_id=f'cand_{len(result):02d}',method=method,path=path,
                           evidence=path_score(path,reference,evidence)))
    return result


def apply_path(items,segment,path,shape,config):
    if not isinstance(path,list) or not 2<=len(path)<=config.max_path_points:
        raise ValueError('Replacement path must contain 2..4 points')
    points=[]
    for p in path:
        if not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p):
            raise ValueError('Finite numeric [x,y] required')
        if not 0<=p[0]<=shape[1] or not 0<=p[1]<=shape[0]:raise ValueError('Path outside image')
        points.append([float(v) for v in p])
    if any(a==b for a,b in zip(points,points[1:])):raise ValueError('Consecutive duplicate coordinates')
    expected=segment['dp_anchor_path']
    if points[0]!=expected[0] or points[-1]!=expected[-1]:raise ValueError('Observed arc endpoints must remain fixed')
    matches=[i for i,p in enumerate(items) if p.identifier==segment['instance_id']]
    if len(matches)!=1:raise ValueError('Unknown or duplicate instance')
    index=matches[0];parcel=items[index]
    rings={r['ring_id']:r['coordinates'] for r in ring_records(parcel.geometry)}
    coords=rings[segment['ring_id']];n=len(coords)
    starts=[i for i,p in enumerate(coords) if p==expected[0]]
    if len(starts)!=1 or len(expected)>n-1:raise ValueError('Ambiguous arc or too few remaining ring points')
    start=starts[0]
    actual=[coords[(start+j)%n] for j in range(len(expected))]
    if actual!=expected:raise ValueError('Stale or previously edited observed DP arc')
    if points==expected:return list(items)
    rest=[];j=(start+len(expected))%n
    while j!=start:rest.append(coords[j]);j=(j+1)%n
    replacement=points+rest
    if len(replacement)<3 or len({tuple(p) for p in replacement})<3:raise ValueError('Ring must retain at least three unique vertices')
    result=list(items);result[index]=parcel.with_geometry(replace_ring(parcel.geometry,segment['ring_id'],replacement))
    return result


def operation_summary(segment,path):
    before=segment['dp_anchor_path'];after={tuple(p) for p in path}
    removed=[p for p in before[1:-1] if tuple(p) not in after]
    old={tuple(p) for p in before}
    added=[p for p in path[1:-1] if tuple(p) not in old]
    return dict(task=segment['task'],path_points_before=len(before),path_points_after=len(path),
                deleted_DP_coordinates=removed,added_control_coordinates=added,
                net_vertex_change=len(path)-len(before))


def preview(raw,dp,current,segment,candidate,shape,config):
    try:
        proposed=apply_path(current,segment,candidate['path'],shape,config)
        return proposed,verify(raw,dp,proposed,config)
    except (ValueError,KeyError) as error:
        return None,dict(accept=False,reasons=[str(error)],metrics=[])


def run_rule(raw,dp,diagnosis,pools,shape,config):
    current=list(dp);events=[]
    for segment in diagnosis['selected']:
        options=pools[segment['segment_id']];baseline=options[0]['evidence']
        removing=segment['task']=='remove_DP_redundancy'
        if removing:
            # A strong shared-tool baseline: prefer the fewest feasible points,
            # allowing a small numeric image-support drop; model may disagree
            # using actual visual evidence, under identical geometry constraints.
            ranked=sorted((o for o in options[1:] if len(o['path'])<len(segment['dp_anchor_path']) and
                o['evidence']['support']>=baseline['support']-config.rule_merge_support_loss),
                key=lambda o:(len(o['path']),-o['evidence']['score'],o['candidate_id']))
        else:
            ranked=sorted((o for o in options[1:] if o['evidence']['score']-baseline['score']>=config.rule_min_gain),
                key=lambda o:(-o['evidence']['score'],len(o['path']),o['candidate_id']))
        committed=False
        for option in ranked:
            proposed,verdict=preview(raw,dp,current,segment,option,shape,config)
            event=dict(segment_id=segment['segment_id'],candidate=option,verdict=verdict,
                operation=operation_summary(segment,option['path']),committed=bool(verdict['accept']))
            events.append(event)
            if verdict['accept']:current=proposed;committed=True;break
        if not committed:events.append(dict(segment_id=segment['segment_id'],action='keep_DP',reason='No feasible rule-selected improvement'))
    return current,events
