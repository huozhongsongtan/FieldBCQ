"""GT-free DP initialization, risk observations and bounded local correction.

This module receives arrays/GeometryInstance objects only, never input manifests
or reference-label paths. Model-authored paths are applied literally.
"""
from dataclasses import dataclass, asdict
import math
import json
import cv2
import numpy as np
from shapely.geometry import LineString, Point
from fieldbqc_geometry import vertices, polygons, holes
from fieldbqc_direct_edit import ring_records, replace_ring


@dataclass(frozen=True)
class CorrectionConfig:
    dp_tolerance: float = 3.
    min_segment_length: float = 12.
    min_arc_deviation: float = 1.5
    max_segments: int = 6
    max_segments_per_instance: int = 2
    max_path_points: int = 4
    max_shift_from_dp: float = 3.
    max_shift_from_raw: float = 6.
    max_area_change_dp: float = .05
    max_area_change_raw: float = .10
    max_new_pair_overlap: float = 4.
    rule_min_gain: float = .025

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid parameter: '+name)
        for name in ['max_segments', 'max_segments_per_instance', 'max_path_points']:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Positive integer required: '+name)
        if self.dp_tolerance <= 0 or self.max_path_points != 4:
            raise ValueError('Positive DP tolerance and four-point local path contract required')


def initialize(raw, config):
    return [p.with_geometry(p.geometry.simplify(config.dp_tolerance, preserve_topology=True)) for p in raw]


def point_budget(n):
    return max(n, min(n+6, math.ceil(1.25*n)+2))


def _sample_path(path, step=.75):
    rows=[]
    for a,b in zip(np.asarray(path, float)[:-1], np.asarray(path, float)[1:]):
        d=b-a; length=float(np.linalg.norm(d))
        if length < 1e-9: continue
        t=np.linspace(0., 1., max(2, math.ceil(length/step)+1), endpoint=True)
        xy=a[None,:]+t[:,None]*d[None,:]
        normal=np.array([-d[1],d[0]])/length
        rows.extend((p,normal) for p in xy)
    if not rows: return np.empty((0,2)), np.empty((0,2))
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


class ImageEvidence:
    """Texture-sensitive heuristic. Its score is not a semantic certificate."""
    def __init__(self, image):
        self.image=image.astype(np.float32)/255.
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        maps=[]
        for sigma in [.8, 1.6]:
            smooth=cv2.GaussianBlur(gray, (0,0), sigma)
            gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)
            gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
            mag=np.hypot(gx,gy)
            denom=max(float(np.percentile(mag,95)),.05)
            maps.append(np.clip(mag/denom,0,1))
        self.gradient=np.mean(maps,axis=0)

    def _values(self, array, xy):
        # Polygon coordinates denote pixel cells, whereas OpenCV indexes centers.
        return cv2.remap(array, (xy[:,0]-.5).astype(np.float32).reshape(-1,1),
                         (xy[:,1]-.5).astype(np.float32).reshape(-1,1),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(len(xy),-1)

    def measure(self, path):
        xy,normals=_sample_path(path)
        if not len(xy):return dict(gradient=0.,contrast=0.,support=0.)
        gradient=float(self._values(self.gradient,xy).mean())
        left=self._values(self.image,xy+normals*2.)
        right=self._values(self.image,xy-normals*2.)
        contrast=float(np.linalg.norm(left-right,axis=1).mean()/math.sqrt(3))
        return dict(gradient=gradient, contrast=contrast, support=.65*gradient+.35*contrast)

    def score(self, path, endpoint_path):
        evidence=self.measure(path)
        displacement=LineString(path).hausdorff_distance(LineString(endpoint_path))
        evidence.update(added_points=len(path)-2, displacement_px=displacement,
                        score=evidence['support']-.015*(len(path)-2)-.005*displacement)
        return evidence


def detect(raw, dp, evidence, config):
    """Observe original arcs removed by DP, without changing either geometry."""
    all_rows=[]; skipped=[]
    for original,simple in zip(raw,dp):
        original_rings={r['ring_id']:r['coordinates'] for r in ring_records(original.geometry)}
        for ring in ring_records(simple.geometry):
            coords=ring['coordinates']; source=original_rings.get(ring['ring_id'])
            if source is None:
                skipped.append(dict(instance_id=simple.identifier,ring_id=ring['ring_id'],reason='ring_correspondence_missing'));continue
            # Duplicate coordinates are ambiguous observations, never guessed.
            lookup={tuple(p):i for i,p in enumerate(source)}
            if len(lookup)!=len(source):
                skipped.append(dict(instance_id=simple.identifier,ring_id=ring['ring_id'],reason='ambiguous_original_indices'));continue
            if any(tuple(p) not in lookup for p in coords):
                skipped.append(dict(instance_id=simple.identifier,ring_id=ring['ring_id'],reason='DP_anchor_not_in_original'));continue
            # Shapely's simplification preserves ring direction; verify the
            # original-index forward arcs partition exactly one original ring.
            anchor_ids=[lookup[tuple(p)] for p in coords]; n=len(source)
            if sum((b-a)%n for a,b in zip(anchor_ids,anchor_ids[1:]+anchor_ids[:1]))!=n:
                skipped.append(dict(instance_id=simple.identifier,ring_id=ring['ring_id'],reason='ring_direction_or_mapping_changed'));continue
            for k,(a,b) in enumerate(zip(coords,coords[1:]+coords[:1])):
                si,ei=anchor_ids[k],anchor_ids[(k+1)%len(coords)]
                ids=[(si+j)%n for j in range((ei-si)%n+1)]
                arc=[source[j] for j in ids]; chord=LineString([a,b])
                devs=[Point(p).distance(chord) for p in arc]
                deviation=max(devs); length=chord.length
                if length<config.min_segment_length or deviation<config.min_arc_deviation:continue
                raw_support=evidence.measure(arc)['support']; dp_support=evidence.measure([a,b])['support']
                all_rows.append(dict(instance_id=simple.identifier,ring_id=ring['ring_id'],
                    dp_edge_index=k, endpoints=[a,b], raw_arc=arc, raw_indices=ids,
                    max_deviation_px=deviation, farthest_index=int(np.argmax(devs)),
                    chord_length_px=length, raw_support=raw_support, dp_support=dp_support,
                    score=deviation/max(config.dp_tolerance,1e-9)+max(0.,raw_support-dp_support),
                    original_ring_vertices=n, dp_ring_vertices=len(coords)))
    all_rows.sort(key=lambda r:(-r['score'],r['instance_id'],r['ring_id'],r['dp_edge_index']))
    counts={};selected=[]
    for i,row in enumerate(all_rows):
        row['segment_id']=f'seg_{i:03d}'
        uid=row['instance_id']
        yes=len(selected)<config.max_segments and counts.get(uid,0)<config.max_segments_per_instance
        row['selected']=yes
        if yes:selected.append(row);counts[uid]=counts.get(uid,0)+1
    return dict(segments=all_rows,selected=selected,skipped_rings=skipped,
                detected=len(all_rows),selected_count=len(selected),runtime_GT_access=False)


def candidate_paths(segment, evidence, config):
    a,b=segment['endpoints']; arc=segment['raw_arc']; proposals=[('keep_DP',[a,b])]
    far=arc[segment['farthest_index']]
    proposals.append(('restore_original_farthest_corner',[a,far,b]))
    local=list(map(list,LineString(arc).simplify(1.,preserve_topology=False).coords))
    if len(local)<=config.max_path_points: proposals.append(('original_arc_DP_1px',local))
    delta=np.array(b)-np.array(a); norm=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
    mid=(np.array(a)+np.array(b))/2.
    for offset in [-3.,-2.,-1.,1.,2.,3.]:
        proposals.append((f'midpoint_normal_offset_{offset:g}px',[a,(mid+offset*norm).tolist(),b]))
    result=[];seen=set()
    for method,path in proposals:
        key=json.dumps(path)
        if key in seen:continue
        seen.add(key)
        result.append(dict(candidate_id=f'cand_{len(result):02d}',method=method,path=path,
                           evidence=evidence.score(path,segment['endpoints'])))
    return result


def apply_path(items, segment, path, shape, config):
    if not isinstance(path,list) or not 2<=len(path)<=config.max_path_points:
        raise ValueError('Local path must contain 2..4 points')
    points=[]
    for point in path:
        if not isinstance(point,list) or len(point)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in point):
            raise ValueError('Expected finite numeric [x,y]')
        if not 0<=point[0]<=shape[1] or not 0<=point[1]<=shape[0]:raise ValueError('Path outside image')
        points.append([float(v) for v in point])
    if any(a==b for a,b in zip(points,points[1:])):raise ValueError('Repeated consecutive point')
    if points[0]!=segment['endpoints'][0] or points[-1]!=segment['endpoints'][1]:
        raise ValueError('Observed endpoints must remain fixed')
    if len(points)==2:return list(items)
    result=list(items)
    matches=[i for i,p in enumerate(items) if p.identifier==segment['instance_id']]
    if len(matches)!=1:raise ValueError('Unknown or duplicate instance')
    i=matches[0]; parcel=items[i]
    rings={r['ring_id']:r['coordinates'] for r in ring_records(parcel.geometry)}
    coords=rings[segment['ring_id']]
    start=[j for j,p in enumerate(coords) if p==points[0]]
    if len(start)!=1:raise ValueError('Ambiguous current endpoint')
    start=start[0]; end=(start+1)%len(coords)
    if coords[end]!=points[-1]:raise ValueError('Stale or already edited DP edge')
    rest=[];j=(end+1)%len(coords)
    while j!=start:rest.append(coords[j]);j=(j+1)%len(coords)
    replacement=points+rest
    geometry=replace_ring(parcel.geometry,segment['ring_id'],replacement)
    result[i]=parcel.with_geometry(geometry)
    return result


def verify(raw, dp, candidate, config):
    reasons=[];metrics=[]
    if [p.identifier for p in raw]!=[p.identifier for p in dp] or [p.identifier for p in dp]!=[p.identifier for p in candidate]:
        return dict(accept=False,reasons=['instance_identity_changed'],metrics=[])
    changed=[]
    for i,(original,base,current) in enumerate(zip(raw,dp,candidate)):
        g=current.geometry; b=base.geometry; a=original.geometry
        if g.equals_exact(b,0):continue
        changed.append(i)
        if g.is_empty or g.geom_type not in {'Polygon','MultiPolygon'} or not g.is_valid:
            reasons.append(current.identifier+':invalid_geometry');continue
        if len(polygons(g))!=len(polygons(b)) or holes(g)!=holes(b):reasons.append(current.identifier+':structure_changed')
        shift_dp=b.boundary.hausdorff_distance(g.boundary);shift_raw=a.boundary.hausdorff_distance(g.boundary)
        area_dp=abs(g.area-b.area)/max(1.,b.area);area_raw=abs(g.area-a.area)/max(1.,a.area)
        n=vertices(g); budget=point_budget(vertices(b))
        row=dict(instance_id=current.identifier,vertices=n,point_budget=budget,
                 shift_dp_px=shift_dp,shift_raw_px=shift_raw,area_change_dp=area_dp,area_change_raw=area_raw)
        metrics.append(row)
        for condition,reason in [(n>budget,'point_budget'),(shift_dp>config.max_shift_from_dp+1e-7,'shift_from_DP'),
            (shift_raw>config.max_shift_from_raw+1e-7,'shift_from_raw'),(area_dp>config.max_area_change_dp,'area_change_DP'),
            (area_raw>config.max_area_change_raw,'area_change_raw')]:
            if condition:reasons.append(current.identifier+':'+reason)
    # Compare cumulative edited scene with the fixed DP anchor, not the last edit.
    if not any('invalid_geometry' in r for r in reasons):
        checked=set()
        for i in changed:
            for j in range(len(candidate)):
                if i==j or tuple(sorted((i,j))) in checked:continue
                checked.add(tuple(sorted((i,j))))
                if not candidate[i].geometry.intersects(candidate[j].geometry):continue
                baseline=dp[i].geometry.intersection(dp[j].geometry).area
                overlap=candidate[i].geometry.intersection(candidate[j].geometry).area
                if overlap-baseline>config.max_new_pair_overlap+1e-7:
                    reasons.append(f'{candidate[i].identifier}/{candidate[j].identifier}:new_pair_overlap')
    return dict(accept=not reasons,reasons=reasons,metrics=metrics,changed_instances=len(changed))


def preview(raw, dp, current, segment, candidate, shape, config):
    try:
        proposed=apply_path(current,segment,candidate['path'],shape,config)
        verdict=verify(raw,dp,proposed,config)
        return proposed,verdict
    except (ValueError,KeyError) as exc:
        return None,dict(accept=False,reasons=[str(exc)],metrics=[])


def run_rule(raw, dp, diagnosis, pools, shape, config):
    current=list(dp);events=[]
    for segment in diagnosis['selected']:
        options=pools[segment['segment_id']]; base=options[0]['evidence']['score']
        ranked=sorted(options[1:],key=lambda r:(-r['evidence']['score'],len(r['path']),r['candidate_id']))
        committed=False
        for option in ranked:
            gain=option['evidence']['score']-base
            if gain<config.rule_min_gain:continue
            proposed,verdict=preview(raw,dp,current,segment,option,shape,config)
            events.append(dict(segment_id=segment['segment_id'],candidate=option,
                               score_gain=gain,verdict=verdict,committed=bool(verdict['accept'])))
            if verdict['accept']:current=proposed;committed=True;break
        if not committed:events.append(dict(segment_id=segment['segment_id'],action='keep_DP',reason='No sufficiently improved feasible heuristic candidate'))
    return current,events
