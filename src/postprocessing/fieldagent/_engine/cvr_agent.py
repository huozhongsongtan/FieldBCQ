"""CVR development prototype. Runtime inputs contain no ground-truth paths.

Learned action-value estimates are trained off-scene, not on reference labels.
Verification remains authoritative. Primitive geometry algorithms are not novel.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HOST = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST))
import cv2
import numpy as np
import shapely
from scipy.linalg import solve
from shapely.geometry import LineString, mapping
from shapely.ops import split, unary_union
from fieldbqc_direct_edit import DirectConfig, DirectEngine
from fieldbqc_geometry import GeometryInstance, geometry_mask, polygons, scene_metrics, vertices
from fieldbqc_region_repair import ring_evidence
from fieldbqc_selective_repair import state_key
from fieldbqc_vertex_optimization import anchor_reasons

VERSION = 'cvr-development-v1.0'
TOOLS = {
 'contour_dp': ['shapely.simplify(preserve_topology=True)'],
 'contour_lines': ['regularize_geometry', 'regularize_ring'],
 'contour_vw': ['shapely.snap', 'shapely.coverage_is_valid', 'shapely.coverage_simplify'],
 'overlap_allocate': ['exclusive_anchor_distance_transform', 'mask_geometry', 'difference'],
 'overlap_clip': ['shapely.difference'],
 'overlap_bisect': ['exclusive_region_centroids', 'shapely.split', 'disputed_face_allocation', 'shapely.union_all'],
 'repair_geometry': ['shapely.make_valid', 'polygonal_component_extraction'],
}
REASONS = ['boundary_drift','excessive_area_change','raster_drift','edge_regression',
           'new_neighbor_overlap','shared_boundary_changed_independently','new_target_fragmentation',
           'survivor_collapsed','tool_error','unchanged']
FEATURES = (['tool_'+k for k in TOOLS] +
 ['tolerance','offset','log_vertices','vertex_share','log_area','holes','components','short_edges','straight',
  'target_count','overlap_share','max_containment','shared_share','degree','external_degree',
  'neighbor_vertex_share','exclusive_fraction','edge_support','image_gradient','image_texture',
  'family_attempts','family_rejections','last_failed_tolerance'] + ['reason_'+r for r in REASONS] +
 ['budget_fraction'])
GRAPH_FEATURES = ['shared_share','degree','external_degree','neighbor_vertex_share']
IMAGE_FEATURES = ['edge_support','image_gradient','image_texture']
FEEDBACK_FEATURES = ['family_attempts','family_rejections','last_failed_tolerance']+['reason_'+r for r in REASONS]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def jsonable(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    if isinstance(value, Path):return str(value)
    raise TypeError(type(value).__name__)

def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=jsonable,allow_nan=False),encoding='utf-8')
    return dict(path=str(path),sha256=sha(path))

def fc(items):
    return dict(type='FeatureCollection',coordinate_space=dict(units='pixels',origin=[0,0]),features=[
       dict(type='Feature',properties=dict(instance_id=p.identifier,class_id=p.cls,score=p.score,
            source_ids=list(p.source_ids),vertex_count=vertices(p.geometry)),geometry=mapping(p.geometry)) for p in items])

def save_items(path,items):return dump(path,fc(items))

def read_items(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    return [GeometryInstance(str(f['properties']['instance_id']),shapely.geometry.shape(f['geometry']),
          f['properties'].get('score',1.),f['properties'].get('class_id',0),
          tuple(f['properties'].get('source_ids',[]))) for f in d['features']]

class Audit:
    """Started and terminal records share an ID; incomplete starts remain visible."""
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.counter=0;self.stack=[];self.methods=Counter()
    def write(self,row):
        row=dict(row,timestamp=datetime.now(timezone.utc).isoformat())
        with (self.root/'calls.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False,default=jsonable,allow_nan=False)+'\n')
    @contextmanager
    def call(self,operation,parameters=None,methods=None):
        self.counter+=1;uid=f'c{self.counter:07d}';start=time.monotonic()
        row=dict(call_id=uid,parent_call_id=self.stack[-1] if self.stack else None,
                 operation=operation,parameters=parameters or {},methods=methods or [],version=VERSION)
        self.write(dict(row,status='started'));self.stack.append(uid)
        result={}
        try:
            yield result
        except Exception as e:
            self.write(dict(row,status='failed',error=f'{type(e).__name__}: {e}',seconds=time.monotonic()-start))
            raise
        else:
            self.write(dict(row,status='completed',result=result,seconds=time.monotonic()-start))
            for m in methods or []:self.methods[m]+=1
        finally:self.stack.pop()

@dataclass(frozen=True)
class Action:
    tool: str
    ids: tuple[str,...]
    tolerance: float|None=None
    owner: str|None=None
    offset: float|None=None
    def key(self):return json.dumps(asdict(self),sort_keys=True)

@dataclass(frozen=True)
class Config:
    budget: int=64
    seconds: float=480.
    seed: int=20261021
    tolerances: tuple=(2.,1.,.5)
    containment_review: float=.95
    overlap_weight: float=2.
    edge_penalty: float=.2
    ridge: float=2.
    exploration: float=.02
    minimum_score: float=0.
    def __post_init__(self):
        if type(self.budget)!=int or self.budget<1 or self.seconds<=0:raise ValueError('Invalid budget')
        if any(not 0<t<=2 for t in self.tolerances):raise ValueError('Tolerance outside shared gates')

def graph_state(engine):
    nodes={}
    gray=cv2.cvtColor(engine.tools.image,cv2.COLOR_BGR2GRAY)
    grad=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    for p in engine.current:
        g=p.geometry;mask=engine.tools.mask(p)
        vals=gray[mask]
        nodes[p.identifier]=dict(area=g.area,holes=sum(len(q.interiors) for q in polygons(g)),
             components=len(polygons(g)),score=p.score,**ring_evidence(g),
             edge_support=engine.tools.edge_alignment(mask),
             image_gradient=float(grad[mask].mean()/1020) if mask.any() else 0.,
             image_texture=float(vals.std()/255) if len(vals) else 0.,valid=bool(g.is_valid))
    edges=[]
    for i,a in enumerate(engine.current):
        for b in engine.current[i+1:]:
            if not a.geometry.envelope.intersects(b.geometry.envelope):continue
            inter=a.geometry.intersection(b.geometry)
            shared=a.geometry.boundary.intersection(b.geometry.boundary).length
            if inter.area>1e-7 or shared>1e-7:
                edges.append(dict(ids=[a.identifier,b.identifier],overlap=inter.area,shared=shared,
                    containment=inter.area/max(1e-9,min(a.geometry.area,b.geometry.area)),
                    bbox=list(inter.bounds) if not inter.is_empty else None,
                    edge_support=.5*(nodes[a.identifier]['edge_support']+nodes[b.identifier]['edge_support'])))
    # Separate positive-overlap connectivity from shared-boundary adjacency.
    groups=[]
    for e in edges:
        if e['overlap']<=1e-7:continue
        ids=set(e['ids']);touch=[g for g in groups if ids&g]
        for g in touch:ids|=g;groups.remove(g)
        groups.append(ids)
    return dict(state_id=state_key(engine.current),nodes=nodes,edges=edges,
                conflicts=sorted([sorted(g) for g in groups]),metrics=scene_metrics(engine.current))

def actions_for(engine,graph,config,joint=True,include_near=True):
    result=[]
    for uid,node in graph['nodes'].items():
        if not node['valid']:
            result.append(Action('repair_geometry',(uid,)));continue
        if node['vertices']<=3:continue
        for tool in ('contour_dp','contour_lines','contour_vw'):
            for t in config.tolerances:result.append(Action(tool,(uid,),t))
    for e in graph['edges']:
        ids=tuple(sorted(e['ids']))
        if e['overlap']>1e-7:
            if not include_near and e['containment']>=config.containment_review:continue
            result.append(Action('overlap_allocate',ids))
            for owner in ids:result.append(Action('overlap_clip',ids,owner=owner))
            for f in (.35,.5,.65):result.append(Action('overlap_bisect',ids,offset=f))
        elif joint and e['shared']>1e-7:
            for t in config.tolerances:result.append(Action('contour_vw',ids,t))
    # Whole connected overlap groups expose third-party effects, not only pairs.
    for ids in graph['conflicts']:
        if len(ids)>2:result.append(Action('overlap_allocate',tuple(ids)))
    return sorted(set(result),key=lambda a:a.key())

def feature_dict(action,graph,history,baseline,remaining,config):
    ids=set(action.ids);nodes=[graph['nodes'][i] for i in action.ids]
    internal=[e for e in graph['edges'] if set(e['ids'])<=ids]
    adjacent=[e for e in graph['edges'] if set(e['ids'])&ids]
    neighbors={i for e in adjacent for i in e['ids']}-ids
    selected=[r for r in history if set(r['action']['ids'])&ids and r['action']['tool']==action.tool]
    last=selected[-1] if selected else None
    reasons=last['reasons'] if last and not last['accepted'] else []
    n=sum(p['vertices'] for p in nodes);area=sum(p['area'] for p in nodes)
    overlap=sum(e['overlap'] for e in internal)
    d={k:0. for k in FEATURES}
    d.update({'tool_'+action.tool:1.,'tolerance':(action.tolerance or 0)/2,'offset':action.offset or 0,
       'log_vertices':math.log1p(n),'vertex_share':n/max(1,baseline['vertices']),
       'log_area':math.log1p(area),'holes':sum(p['holes'] for p in nodes),
       'components':sum(p['components'] for p in nodes),'short_edges':float(np.mean([p['short_edge_fraction'] for p in nodes])),
       'straight':float(np.mean([p['near_straight_fraction'] for p in nodes])),
       'target_count':len(ids),'overlap_share':overlap/max(1,baseline['overlap_excess_area']),
       'max_containment':max([e['containment'] for e in internal] or [0]),
       'shared_share':sum(e['shared'] for e in adjacent)/max(1,sum(p['perimeter'] for p in nodes)),
       'degree':len(adjacent),'external_degree':len(neighbors),
       'neighbor_vertex_share':sum(graph['nodes'][i]['vertices'] for i in neighbors)/max(1,baseline['vertices']),
       'exclusive_fraction':max(0,1-overlap/max(1,area)),
       'edge_support':float(np.mean([p['edge_support'] for p in nodes])),
       'image_gradient':float(np.mean([p['image_gradient'] for p in nodes])),
       'image_texture':float(np.mean([p['image_texture'] for p in nodes])),
       'family_attempts':min(10,len(selected))/10,
       'family_rejections':min(10,sum(not r['accepted'] for r in selected))/10,
       'last_failed_tolerance':((last['action']['tolerance'] or 0)/2) if last and not last['accepted'] else 0,
       'budget_fraction':remaining/config.budget})
    d.update({'reason_'+k:float(k in reasons) for k in REASONS})
    return d

class ValueModel:
    """Ridge action outcomes: acceptance, qualified V/O gains, edge drop, log cost."""
    def __init__(self,disabled=()):self.disabled=set(disabled)
    def vector(self,features):return np.array([0. if k in self.disabled else features[k] for k in FEATURES])
    def fit(self,rows,training_scenes,ridge=2.):
        if not rows:raise ValueError('No training trajectories')
        self.training_scenes=sorted(training_scenes)
        x=np.stack([self.vector(r['features']) for r in rows]);y=np.array([r['outcomes'] for r in rows])
        self.mean=x.mean(0);self.std=np.maximum(x.std(0),.1)
        z=np.c_[np.ones(len(x)),(x-self.mean)/self.std]
        penalty=np.eye(z.shape[1])*ridge;penalty[0,0]=0
        # einsum avoids this Windows environment's broken NumPy BLAS DLL path;
        # the ridge objective is unchanged and SciPy provides the small solve.
        gram=np.einsum('ni,nj->ij',z,z,optimize=False)
        cross=np.einsum('ni,nk->ik',z,y,optimize=False)
        self.weights=solve(gram+penalty,cross,assume_a='pos')
        self.training_count=len(rows)
        return self
    def predict(self,features):
        z=np.r_[1.,(self.vector(features)-self.mean)/self.std]
        p=np.einsum('i,ik->k',z,self.weights,optimize=False)
        return dict(accept=float(np.clip(p[0],0,1)),vgain=float(np.clip(p[1],-.5,1)),
             ogain=float(np.clip(p[2],0,1)),edge_drop=float(np.clip(p[3],0,1)),
             cost=float(np.clip(np.expm1(np.clip(p[4],0,5)),.001,120)))
    def document(self):return dict(version=VERSION,features=FEATURES,disabled=sorted(self.disabled),
        mean=self.mean,std=self.std,weights=self.weights,training_scenes=self.training_scenes,
        training_candidates=self.training_count,targets=['accepted','qualified_vertex_gain','qualified_overlap_gain','edge_drop','log1p_actual_compute_seconds'],
        uses_gt=False)

def score_action(action,features,model,config,no_cost=False):
    p=model.predict(features)
    gain=p['vgain']+config.overlap_weight*p['ogain']-config.edge_penalty*p['edge_drop']
    # Gains already include rejection=0 during training; do not multiply twice.
    cost=1. if no_cost else .05+p['cost']
    exploration=config.exploration*features['vertex_share']/math.sqrt(1+10*features['family_attempts'])
    return float(gain/cost+exploration),p

class Executor:
    def __init__(self,image,items,config,root):
        self.config=config;self.audit=Audit(root);self.root=Path(root)
        self.engine=DirectEngine(image,items,DirectConfig(tool_budget=config.budget,seconds=config.seconds))
        self.original=list(items);self.baseline=scene_metrics(items);self.history=[];self.tried=set();self.reviews=[]
        self.graph_cache=None;self.capture=[]
        self.instrument()
        with self.audit.call('save_baseline',methods=['GeoJSON serialization']) as out:
            out.update(save_items(self.root/'baseline.pixel.geojson',items))
    def instrument(self):
        # Real dispatched method boundaries, not fabricated primitive call traces.
        bindings=[(self.engine.tools,'verify'),(self.engine,'verify_conflict_candidate'),
                  (self.engine,'_simplify'),(self.engine,'_conflict'),(self.engine.tools,'_allocate'),
                  (self.engine.tools,'mask'),(self.engine.tools,'metrics'),(self.engine.tools,'edge_alignment')]
        for obj,name in bindings:
            original=getattr(obj,name)
            def wrapped(*args,_fn=original,_name=name,**kwargs):
                params={}
                if _name=='mask':params['instance_id']=args[0].identifier
                if _name in ('verify','verify_conflict_candidate'):
                    self.capture.append(list(args[1]));params['target_ids']=list(args[3] if _name=='verify' else args[2])
                if _name in ('_simplify','_conflict'):params.update(ids=args[1],tool=args[2],parameter=args[3])
                with self.audit.call(_name,params,[_fn.__qualname__]) as out:
                    value=_fn(*args,**kwargs)
                    if isinstance(value,dict):out.update(value)
                    elif _name=='mask':out['pixels']=int(value.sum())
                    elif _name=='edge_alignment':out['score']=float(value)
                    return value
            setattr(obj,name,wrapped)
    def graph(self):
        key=state_key(self.engine.current)
        if self.graph_cache is None or self.graph_cache['state_id']!=key:
            with self.audit.call('conflict_graph',methods=['pair_intersection','shared_boundary_intersection','overlap_connected_components','Canny/Sobel_features']) as out:
                self.graph_cache=graph_state(self.engine)
                path=self.root/'states'/(key+'.graph.json')
                out.update(dump(path,self.graph_cache))
        return self.graph_cache
    def signature(self,action):
        # Unchanged local geometry and anchors imply no new information. Include
        # neighbors within the maximum shift so outside edits can invalidate it.
        targets=[p for p in self.engine.current if p.identifier in action.ids]
        envelope=unary_union([p.geometry for p in targets]).envelope.buffer(2.00001)
        relevant=[p for p in self.engine.current if p.geometry.intersects(envelope)]
        anchors=[p for p in self.engine.anchors if p.identifier in {q.identifier for q in relevant}]
        return (state_key(relevant),state_key(anchors),action.key())
    def custom_trial(self,action):
        e=self.engine;before=list(e.current);lookup={p.identifier:p for p in before}
        e._reserve()
        try:
            if action.tool=='repair_geometry':
                uid=action.ids[0];g=shapely.make_valid(lookup[uid].geometry)
                parts=polygons(g);g=unary_union(parts)
                candidate=[p.with_geometry(g) if p.identifier==uid else p for p in before]
                verdict=e.tools.verify(before,candidate,'repair_geometry',list(action.ids))
                rs=anchor_reasons(e.tools,e.anchors,candidate,list(action.ids))
                verdict['reasons']=sorted(set(verdict['reasons']+rs));verdict['accept']=not verdict['reasons']
            else:
                a,b=[lookup[i] for i in action.ids];disputed=a.geometry.intersection(b.geometry)
                anchors=[p.geometry.difference(disputed) for p in (a,b)]
                if any(g.is_empty for g in anchors):raise ValueError('No exclusive anchor for explicit partition')
                centers=np.array([[g.centroid.x,g.centroid.y] for g in anchors]);direction=centers[1]-centers[0]
                if np.linalg.norm(direction)<1e-7:raise ValueError('Coincident anchors')
                point=centers[0]+action.offset*direction
                perpendicular=np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
                extent=4*max(e.tools.shape)
                path=[(point-perpendicular*extent).tolist(),(point+perpendicular*extent).tolist()]
                faces=polygons(split(disputed,LineString(path)))
                if len(faces)<2:raise ValueError('Partition line does not split disputed area')
                assigned=[[],[]]
                for face in faces:
                    side=float((np.array([face.representative_point().x,face.representative_point().y])-point)@direction)
                    assigned[int(side>=0)].append(face)
                replacements={p.identifier:unary_union([anchors[k],*assigned[k]]) for k,p in enumerate((a,b))}
                candidate=[p.with_geometry(replacements[p.identifier]) if p.identifier in replacements else p for p in before]
                dump(self.root/'partition_requests'/f'{len(e.trials):04d}.json',dict(path=path,target_ids=action.ids,offset=action.offset,
                     method='explicit perpendicular separator between exclusive-region centroids; no image-derived ownership proof'))
                verdict=e.verify_conflict_candidate(before,candidate,list(action.ids))
        except Exception as error:
            candidate=before;verdict=dict(accept=False,reasons=['tool_error'],error=str(error),metrics={})
        tid=f't{len(e.trials):04d}'
        r=dict(trial_id=tid,tool_id=action.tool,target_ids=list(action.ids),base_state=state_key(before),
             accept=verdict['accept'],before=scene_metrics(before),candidate=scene_metrics(candidate if verdict['accept'] else before),
             details=[dict(target_ids=list(action.ids),**verdict)])
        e.trials[tid]=(r,list(candidate if verdict['accept'] else before));return r
    def execute(self,action,features,prediction=None):
        e=self.engine;signature=self.signature(action)
        if signature in self.tried:raise ValueError('Repeated action in identical state')
        self.tried.add(signature);before=list(e.current);before_metrics=scene_metrics(before)
        self.capture=[];began=time.monotonic()
        input_state=state_key(e.current)
        with self.audit.call('candidate',dict(action=asdict(action),state_id=input_state,local_signature=signature,prediction=prediction),TOOLS[action.tool]) as event:
            start=time.monotonic()
            if action.tool in ('overlap_bisect','repair_geometry'):record=self.custom_trial(action)
            else:record=e.trial(action.tool,list(action.ids),tolerance=action.tolerance,keep_id=action.owner)
            compute=time.monotonic()-start
            tid=record['trial_id'];folder=self.root/'candidates'/tid
            attempted=[save_items(folder/f'attempted-{k:02d}.pixel.geojson',v) for k,v in enumerate(self.capture)]
            accepted_candidate=e.trials[tid][1]
            candidate_file=save_items(folder/'accepted_subset.pixel.geojson',accepted_candidate)
            after=scene_metrics(accepted_candidate)
            # Independent scene-level commit gate; per-action gates remain unchanged.
            vg=(before_metrics['vertices']-after['vertices'])/max(1,self.baseline['vertices'])
            og=(before_metrics['overlap_excess_area']-after['overlap_excess_area'])/max(1,self.baseline['overlap_excess_area'])
            delta=vg+self.config.overlap_weight*og
            finite=all(np.isfinite(shapely.get_coordinates(p.geometry)).all() for p in accepted_candidate)
            identity=[p.identifier for p in accepted_candidate]==[p.identifier for p in before]
            scene_ok=(identity and finite and not after['invalid_geometries'] and
                      after['overlap_excess_area']<=before_metrics['overlap_excess_area']+1e-7 and delta>1e-9)
            reasons=sorted({x for d in record['details'] for x in d['reasons']})
            if record['accept'] and not scene_ok:reasons.append('scene_commit_gate')
            edge_drop=0.
            for detail in record['details']:
                for target in detail.get('metrics',{}).get('targets',{}).values():
                    edge_drop=max(edge_drop,target.get('edge_before',0)-target.get('edge_after',0))
            commit=bool(record['accept'] and scene_ok)
            if commit:
                with self.audit.call('commit',dict(trial_id=tid,state_id=input_state),['verified_scene_commit']) as out:
                    if action.tool=='overlap_bisect':
                        # The common commit resets shape anchors for declared ownership edits.
                        saved=record['tool_id'];record['tool_id']='overlap_clip'
                        try:out.update(e.commit(tid,'CVR scene gain and all shared gates passed'))
                        finally:record['tool_id']=saved
                    else:out.update(e.commit(tid,'CVR scene gain and all shared gates passed'))
                    out['checkpoint']=save_items(self.root/'checkpoints'/(tid+'.pixel.geojson'),e.current)
            row=dict(action=asdict(action),state_id=input_state,trial_id=tid,features=features,
               accepted=commit,validator_accepted=bool(record['accept']),reasons=reasons,
               before=before_metrics,after=scene_metrics(e.current),candidate_metrics=after,
               outcomes=[float(commit),vg if commit else 0.,og if commit else 0.,edge_drop,math.log1p(compute)],
               compute_seconds=compute,attempted_files=attempted,candidate_file=candidate_file,
               candidate_record=record,prediction=prediction)
            dump(folder/'result.json',row);event.update(trial_id=tid,committed=commit,reasons=reasons,candidate=candidate_file)
        row['wall_seconds']=time.monotonic()-began;self.history.append(row)
        return row
    def finish(self,stop,policy):
        e=self.engine;final=scene_metrics(e.current)
        with self.audit.call('export_final',methods=['GeoJSON serialization','rasterio.rasterize','np.savez_compressed']) as out:
            out['vector']=save_items(self.root/'final/prediction.repaired.pixel.geojson',e.current)
            destination=self.root/'final/prediction.repaired_masks.npz'
            masks=np.stack([geometry_mask(p.geometry,e.tools.shape) for p in e.current]).astype(np.uint8)
            np.savez_compressed(destination,masks=masks,scores=np.array([p.score for p in e.current]),
                 classes=np.array([p.cls for p in e.current]),parcel_ids=np.array([p.identifier for p in e.current]),
                 quality_valid=np.zeros(len(e.current),bool),quality_sources=np.array(['postprocessed_unverified']*len(e.current)))
            out['masks']=dict(path=str(destination),sha256=sha(destination))
        trace=dict(version=VERSION,policy=policy,baseline=self.baseline,final=final,stop_reason=stop,
             candidates=e.generated,commits=sum(r['accepted'] for r in self.history),reviews=self.reviews,
             final_graph=self.graph(),history=self.history,model_calls=0,
             independent_semantic_correctness=None,methods=dict(self.audit.methods),
             audit_scope='Actual dispatched tool/validator/helper boundaries and specified method chains. Primitive internal GEOS/NumPy operations are not individually intercepted.',
             postprocess_seconds=time.monotonic()-e.started if hasattr(e,'started') else None)
        dump(self.root/'trace.json',trace);dump(self.root/'methods.json',dict(actual_method_calls=self.audit.methods,
          actions=[dict(trial_id=r['trial_id'],action=r['action'],methods=TOOLS[r['action']['tool']],
                        committed=r['accepted'],reasons=r['reasons']) for r in self.history]))
        return trace

def fixed_key(a):
    stages={'overlap_allocate':0,'overlap_clip':1,'overlap_bisect':2,'repair_geometry':3,
            'contour_dp':4,'contour_lines':5,'contour_vw':6}
    return (stages[a.tool],a.ids,-(a.tolerance or 0),a.owner or '',a.offset or 0)

def run_controller(image,items,root,config,policy,model=None):
    executor=Executor(image,items,config,root);e=executor.engine
    rng=np.random.default_rng(config.seed)
    no_review=policy=='no_review';joint=policy!='no_joint'
    reviewed_pairs=set();stop='candidate_budget'
    while e.generated<config.budget and e.remaining()>0:
        graph=executor.graph()
        if not no_review:
            for edge in graph['edges']:
                key=tuple(edge['ids'])
                if edge['overlap']>1e-7 and edge['containment']>=config.containment_review and key not in reviewed_pairs:
                    with executor.audit.call('review',dict(ids=list(key)),['near_containment_identity_review']) as out:
                        row=dict(ids=list(key),state_id=graph['state_id'],reason='Near containment; distinct identity/ownership requires independent review',
                                 geometric_containment=edge['containment'],semantic_correctness_known=False)
                        executor.reviews.append(row);out.update(row);reviewed_pairs.add(key)
        with executor.audit.call('action_preconditions',dict(joint=joint),['registry_preconditions']) as out:
            actions=actions_for(e,graph,config,joint=joint)
            if not no_review:
                near_pairs=[set(edge['ids']) for edge in graph['edges'] if edge['containment']>=config.containment_review]
                actions=[a for a in actions if not (a.tool.startswith('overlap_') and any(pair<=set(a.ids) for pair in near_pairs))]
            if policy.startswith('single_'):
                wanted={'single_dp':'contour_dp','single_lines':'contour_lines','single_allocate':'overlap_allocate'}[policy]
                actions=[a for a in actions if a.tool==wanted]
            eligible=[]
            for a in actions:
                if executor.signature(a) in executor.tried:continue
                ok,reason=e.eligibility(a.tool,list(a.ids))
                if ok:eligible.append(a)
            out.update(available=len(actions),eligible=len(eligible))
        if not eligible:stop='no_eligible_untried_action';break
        history=[] if policy=='no_feedback' else executor.history
        features=[feature_dict(a,graph,history,executor.baseline,config.budget-e.generated,config) for a in eligible]
        predictions=None
        with executor.audit.call('select_action',dict(policy=policy,candidate_count=len(eligible)),['off_scene_ridge_action_value' if model else policy]) as out:
            if policy=='explore':index=int(rng.integers(len(eligible)))
            elif policy in ('fixed','single_dp','single_lines','single_allocate'):
                index=min(range(len(eligible)),key=lambda k:fixed_key(eligible[k]))
            elif policy=='greedy':
                values=[f['vertex_share']*(.5+.5*f['tolerance'])+config.overlap_weight*f['overlap_share']-
                        .05*f['max_containment'] if eligible[k].tool.startswith('overlap_') else
                        f['vertex_share']*(.5+.5*f['tolerance']) for k,f in enumerate(features)]
                index=int(np.argmax(values))
            else:
                scored=[score_action(a,f,model,config,no_cost=policy=='no_cost') for a,f in zip(eligible,features)]
                index=max(range(len(eligible)),key=lambda k:(scored[k][0],eligible[k].key()))
                predictions=scored[index][1]
                if scored[index][0]<config.minimum_score:stop='predicted_gain_below_cost';break
                out.update(score=scored[index][0],prediction=predictions)
            action=eligible[index];out['action']=asdict(action)
        row=executor.execute(action,features[index],predictions)
        with (Path(root)/'training_rows.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({k:row[k] for k in ['features','outcomes','action','accepted','reasons']},default=jsonable)+'\n')
    if e.remaining()<=0:stop='time_budget'
    trace=executor.finish(stop,policy)
    return trace
