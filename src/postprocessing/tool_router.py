"""GT-blind per-parcel visual choice between three frozen local outputs."""
from pathlib import Path
import base64
import math
import cv2
import numpy as np
from fieldagent.io import save
from fieldbqc_geometry import vertices,polygons,holes
from fieldbqc_direct_edit import ring_records
from dp_structure_agent import obj,XY,validate_schema
from routed_editor import request

TOOLS=('nth_point','radial_distance','dp')
ROUTING_PROMPT=Path(__file__).with_name('ROUTING_PROMPT.md')
ROUTE_SCHEMA=obj(dict(decisions=dict(type='array',items=obj(dict(
    instance_id=dict(type='string'),tool=dict(type='string',enum=list(TOOLS)),
    evidence_point=XY,reason=dict(type='string'))))))


def validate_choices(packet,expected_ids,shape):
    validate_schema(packet,ROUTE_SCHEMA)
    actual=[d['instance_id'] for d in packet['decisions']]
    if len(actual)!=len(set(actual)) or set(actual)!=set(expected_ids):
        raise ValueError('Exactly one tool choice per observed instance is required')
    for d in packet['decisions']:
        x,y=d['evidence_point']
        if not 0<=x<=shape[1] or not 0<=y<=shape[0]:raise ValueError('Evidence point outside original image')
    return packet['decisions']


def apply_choices(raw,options,decisions,shape):
    packet=dict(decisions=decisions)
    validate_choices(packet,[p.identifier for p in raw],shape)
    chosen={d['instance_id']:d for d in decisions};result=[];audit=[]
    for p in raw:
        d=chosen[p.identifier];candidate=options[d['tool']][p.identifier]
        if candidate.identifier!=p.identifier or not candidate.geometry.is_valid or candidate.geometry.is_empty:
            raise ValueError('Unknown or invalid observed tool output')
        if len(polygons(candidate.geometry))!=len(polygons(p.geometry)) or holes(candidate.geometry)!=holes(p.geometry):
            raise ValueError('Candidate component/hole contract changed')
        result.append(p.with_geometry(candidate.geometry))
        audit.append(dict(d,tool_parameter_is_fixed=True,raw_vertices=vertices(p.geometry),
            selected_vertices=vertices(candidate.geometry),selected_equals_raw=candidate.geometry.equals_exact(p.geometry,0)))
    return result,audit


def geometry_observation(item):
    rings=ring_records(item.geometry);shown=[]
    for r in rings[:6]:
        coords=r['coordinates'];n=len(coords)
        ids=sorted(set(int(i) for i in np.linspace(0,n-1,min(n,32))))
        shown.append(dict(ring_id=r['ring_id'],total_vertices=n,
            sampled=len(ids)!=n,observed_points=[dict(index=i,xy=coords[i]) for i in ids]))
    return dict(vertices=vertices(item.geometry),bounds=list(item.geometry.bounds),
        components=len(polygons(item.geometry)),holes=holes(item.geometry),
        geometry_valid=item.geometry.is_valid,total_rings=len(rings),rings_omitted=max(0,len(rings)-6),rings=shown)


def _image_url(frame,path):
    ok,data=cv2.imencode('.png',frame)
    if not ok:raise ValueError('Image encoding failed')
    Path(path).write_bytes(data.tobytes())
    return 'data:image/png;base64,'+base64.b64encode(data).decode('ascii')


def route_context(image,raw_page,options,parameters,tool_metadata,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows=[];maps=[];observed=[];side=128;header=22
    for raw in raw_page:
        x0,y0,x1,y1=raw.geometry.bounds
        x0=max(0,int(math.floor(x0))-8);y0=max(0,int(math.floor(y0))-8)
        x1=min(image.shape[1],int(math.ceil(x1))+9);y1=min(image.shape[0],int(math.ceil(y1))+9)
        crop=image[y0:y1,x0:x1].copy();panels=[]
        for method in ['plain','raw',*TOOLS]:
            frame=crop.copy()
            if method!='plain':
                item=raw if method=='raw' else options[method][raw.identifier]
                for ring in ring_records(item.geometry):
                    xy=np.round(np.asarray(ring['coordinates'])-[x0,y0]).astype(np.int32)
                    cv2.polylines(frame,[xy],True,(0,255,255),1,cv2.LINE_AA)
                    for point in xy:cv2.circle(frame,tuple(point),1,(0,255,255),-1)
            scale=min(side/max(1,crop.shape[1]),side/max(1,crop.shape[0]))
            panel=np.full((side+header,side,3),245,np.uint8)
            resized=cv2.resize(frame,(max(1,round(frame.shape[1]*scale)),max(1,round(frame.shape[0]*scale))))
            panel[header:header+resized.shape[0],:resized.shape[1]]=resized
            cv2.putText(panel,raw.identifier+' '+method,(2,15),cv2.FONT_HERSHEY_SIMPLEX,.31,(0,0,0),1)
            panels.append(panel)
        maps.append(dict(instance_id=raw.identifier,board_index=len(rows)//8,row_on_board=len(rows)%8,
            original_bounds=[x0,y0,x1,y1],panel_width=side,panel_header=header,scale=scale,
            column_order=['plain','raw',*TOOLS]))
        rows.append(np.hstack(panels));candidates=[]
        for method in TOOLS:
            item=options[method][raw.identifier];g=item.geometry
            candidates.append(dict(tool=method,parameter=parameters[method],**geometry_observation(item),
                raw_boundary_distance_px=raw.geometry.boundary.hausdorff_distance(g.boundary),
                raw_symmetric_difference_fraction=raw.geometry.symmetric_difference(g).area/max(raw.geometry.area,1),
                sampling_fallback_audit=tool_metadata.get(method,{}).get(raw.identifier,{})))
        observed.append(dict(instance_id=raw.identifier,raw=geometry_observation(raw),candidates=candidates))
    urls=[_image_url(image,folder/'original.png')]
    for start in range(0,len(rows),8):
        urls.append(_image_url(np.vstack(rows[start:start+8]),folder/f'board_{start//8:02d}.png'))
    context=dict(phase='choose_initial_tool_per_instance',runtime_GT_access=False,
        image_shape=list(image.shape[:2]),tools=list(TOOLS),parameters=parameters,
        goal='Choose the fewest visually justified structural points without erasing physical bends; later local refinement is a separate stage.',
        candidates_are_actual_tool_outputs=True,forced_tool_diversity=False,
        instances=observed,image_mapping=maps,
        geometry_feature_warning='Distance to raw is not distance to truth; annotations and prior quality scores are absent.')
    save(folder/'image_mapping.json',maps)
    return context,urls


def route_scene(image,raw,options,parameters,tool_metadata,settings,folder,page_size):
    folder=Path(folder);decisions=[];pages=0
    for start in range(0,len(raw),page_size):
        page=raw[start:start+page_size];phase=folder/f'page_{pages+1:03d}'
        context,urls=route_context(image,page,options,parameters,tool_metadata,phase)
        packet=request(context,urls,settings,ROUTE_SCHEMA,phase,prompt_path=ROUTING_PROMPT)
        decisions.extend(validate_choices(packet,[p.identifier for p in page],image.shape[:2]));pages+=1
    result,audit=apply_choices(raw,options,decisions,image.shape[:2])
    trace=dict(model_calls=pages,commits=0,direct_committed=0,unfinished_target_ids=[],
        events=audit,tool_choices={method:sum(d['tool']==method for d in decisions) for method in TOOLS},
        runtime_GT_access=False,planner_has_raster_input=True,
        execution='Select exact precomputed local tool output per parcel; no coordinate edits in routing stage',
        parameters=parameters,all_instance_ids_routed=True)
    save(folder/'route_trace.json',trace)
    return result,trace
