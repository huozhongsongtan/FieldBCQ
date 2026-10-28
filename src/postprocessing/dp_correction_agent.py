"""Two-call visual correction with literal edits and explicit preview commits."""
import base64
import json
import os
import math
from pathlib import Path
import time
import urllib.request
import urllib.error
import cv2
import numpy as np
from fieldagent.io import load, save, digest
from dp_correction import preview, verify


PROMPT=Path(__file__).with_name('DP_CORRECTION_PROMPT.md')


def obj(properties):
    return dict(type='object',additionalProperties=False,properties=properties,required=list(properties))


XY=dict(type='array',items=dict(type='number'),minItems=2,maxItems=2)
PROPOSE=obj(dict(decisions=dict(type='array',items=obj(dict(
    segment_id=dict(type='string'),action=dict(type='string',enum=['keep','candidate','direct']),
    candidate_id=dict(type='string'),path=dict(type='array',items=XY),
    evidence_point=XY,reason=dict(type='string'))))))
REVIEW=obj(dict(decisions=dict(type='array',items=obj(dict(
    segment_id=dict(type='string'),commit=dict(type='boolean'),reason=dict(type='string'))))))


def validate_schema(value, schema, path='response'):
    """Small strict validator for this protocol's object/array/scalar subset."""
    kind=schema['type']
    valid={'object':type(value) is dict,'array':type(value) is list,
           'string':type(value) is str,'boolean':type(value) is bool,
           'number':type(value) in (int,float) and math.isfinite(value) if type(value) in (int,float) else False}[kind]
    if not valid:raise ValueError(path+': wrong JSON type '+kind)
    if 'enum' in schema and value not in schema['enum']:raise ValueError(path+': unknown enum')
    if kind=='object':
        if set(value)!=set(schema['properties']):raise ValueError(path+': missing or unexpected keys')
        for key,item in value.items():validate_schema(item,schema['properties'][key],path+'.'+key)
    if kind=='array':
        if len(value)<schema.get('minItems',0) or len(value)>schema.get('maxItems',float('inf')):raise ValueError(path+': invalid array length')
        for i,item in enumerate(value):validate_schema(item,schema['items'],path+f'[{i}]')


def atlas(image, segments, folder, proposals=None):
    """Save local raw/DP/proposal evidence with explicit original-coordinate maps."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    side=224;rows=[];maps=[]
    for segment in segments:
        coords=np.asarray(segment['raw_arc']+segment['endpoints'],float)
        x0=max(0,int(np.floor(coords[:,0].min()))-12);y0=max(0,int(np.floor(coords[:,1].min()))-12)
        x1=min(image.shape[1],int(np.ceil(coords[:,0].max()))+13);y1=min(image.shape[0],int(np.ceil(coords[:,1].max()))+13)
        crop=image[y0:y1,x0:x1].copy(); frames=[crop.copy(),crop.copy()]
        scale=min(side/max(1,x1-x0),side/max(1,y1-y0))
        def line(frame,path,color):
            pts=np.round(np.asarray(path)-np.array([x0,y0])).astype(np.int32)
            cv2.polylines(frame,[pts],False,color,1,cv2.LINE_AA)
            for p in pts:cv2.circle(frame,tuple(p),1,color,-1)
        line(frames[1],segment['raw_arc'],(255,255,0))
        line(frames[1],segment['endpoints'],(0,255,255))
        names=['plain','raw cyan / DP yellow']
        if proposals is not None:
            frame=crop.copy();line(frame,segment['endpoints'],(0,255,255))
            path=proposals.get(segment['segment_id'])
            if path:line(frame,path,(0,255,0))
            frames.append(frame);names.append('DP yellow / proposal green')
        panels=[]
        for frame,name in zip(frames,names):
            panel=np.full((side+30,side,3),245,np.uint8)
            resized=cv2.resize(frame,(max(1,round(frame.shape[1]*scale)),max(1,round(frame.shape[0]*scale))))
            panel[30:30+resized.shape[0],:resized.shape[1]]=resized
            cv2.putText(panel,segment['segment_id']+' '+name,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.34,(0,0,0),1)
            panels.append(panel)
        rows.append(np.hstack(panels))
        maps.append(dict(segment_id=segment['segment_id'],original_bounds=[x0,y0,x1,y1],
                         row=len(rows)-1,panel_width=side,panel_header=30,scale=scale))
    urls=[]
    if rows:
        board=np.vstack(rows);path=folder/'local_preview.png'
        ok,data=cv2.imencode('.png',board)
        if not ok:raise ValueError('Image encoding failed')
        path.write_bytes(data.tobytes());urls.append('data:image/png;base64,'+base64.b64encode(data).decode('ascii'))
    # Original scene context has no annotations/reference layers.
    context=image
    if max(image.shape[:2])>768:
        scale=768/max(image.shape[:2]);context=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
    ok,data=cv2.imencode('.png',context)
    if not ok:raise ValueError('Image encoding failed')
    (folder/'original.png').write_bytes(data.tobytes())
    urls.insert(0,'data:image/png;base64,'+base64.b64encode(data).decode('ascii'))
    save(folder/'image_mapping.json',maps)
    return urls,maps


def request(context, urls, settings, schema, folder):
    """Completed paid responses are reused. Interrupted requests are not replayed."""
    from fieldbqc_planner import APIPlanner
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    prompt=PROMPT.read_text(encoding='utf-8')
    payload=APIPlanner(settings,schema=schema,system_prompt=prompt)._payload(context,urls)
    if (folder/'request.json').exists():
        if load(folder/'request.json')!=payload:raise ValueError('Frozen request changed')
    else:
        save(folder/'context.json',context);save(folder/'request.json',payload)
        (folder/'system_prompt.md').write_text(prompt,encoding='utf-8')
    data_path=folder/'response.json'
    if not data_path.exists():
        if (folder/'started.json').exists():
            raise RuntimeError('Unreconciled request start: inspect '+str(folder)+'; no automatic paid replay')
        key=os.environ.get(settings.get('api_key_env','OPENAI_API_KEY'))
        if not key:raise RuntimeError('Configured API key unavailable')
        endpoint=settings['base_url'].rstrip('/')+('/responses' if settings['api']=='responses' else '/chat/completions')
        meta=dict(endpoint=endpoint,requested_model=settings['model'],request_sha256=digest(folder/'request.json'),
                  retries=0)
        save(folder/'started.json',meta);began=time.monotonic()
        try:
            req=urllib.request.Request(endpoint,data=json.dumps(payload,allow_nan=False).encode('utf-8'),
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=settings.get('timeout_seconds',90)) as response:data=json.load(response)
            save(data_path,data)
            meta.update(seconds=time.monotonic()-began,usage=data.get('usage',{}),response_model=data.get('model'),response_id=data.get('id'))
            save(folder/'metadata.json',meta)
        except Exception as error:
            message='HTTP '+str(error.code) if isinstance(error,urllib.error.HTTPError) else type(error).__name__+': '+str(error)
            save(folder/'error.json',dict(meta,error=message,seconds=time.monotonic()-began))
            raise RuntimeError(message) from None
    data=load(data_path)
    if settings['api']=='responses':
        if data.get('status')!='completed':raise ValueError('Incomplete response retained')
        text=''.join(c.get('text','') for m in data.get('output',[]) if m.get('type')=='message'
                     for c in m.get('content',[]) if c.get('type')=='output_text')
    else:
        choice=data['choices'][0]
        if choice.get('finish_reason')!='stop':raise ValueError('Incomplete response retained')
        text=choice['message']['content']
    packet=json.loads(text)
    validate_schema(packet,schema)
    save(folder/'decision.json',packet)
    return packet


def check_ids(packet, expected):
    actual=[r['segment_id'] for r in packet['decisions']]
    if len(actual)!=len(set(actual)) or set(actual)!=set(expected):raise ValueError('Exactly one decision per observed segment required')


def run(image, raw, dp, diagnosis, pools, config, settings, folder):
    folder=Path(folder);segments=diagnosis['selected'];began=time.monotonic()
    if not segments:
        return list(dp),dict(model_calls=0,commits=0,direct_committed=0,seconds=time.monotonic()-began,
            stop_reason='No selected suspicious DP segments; identity result without API',events=[],unfinished_target_ids=[])
    urls,mapping=atlas(image,segments,folder/'01_propose')
    # Compress observation coordinates, not actual candidate/verification geometry.
    observed=[]
    for s in segments:
        shown=dict(s);n=len(s['raw_arc'])
        indices=sorted(set([0,n-1,s['farthest_index'],*[int(i) for i in np.linspace(0,n-1,min(n,24))]]))
        shown.pop('raw_arc');shown.pop('raw_indices')
        shown['observed_raw_points']=[dict(original_index=s['raw_indices'][i],xy=s['raw_arc'][i]) for i in indices]
        shown['raw_observation_sampled']=len(indices)<n
        shown['candidates']=pools[s['segment_id']];observed.append(shown)
    ctx=dict(phase='propose',goal='Preserve DP low complexity, correct only image-supported true corners or displaced arcs',
             runtime_GT_access=False,image_shape=list(image.shape[:2]),config=config.__dict__,segments=observed,image_mapping=mapping)
    packet=request(ctx,urls,settings,PROPOSE,folder/'01_propose');check_ids(packet,[s['segment_id'] for s in segments])
    decisions={d['segment_id']:d for d in packet['decisions']}
    provisional=list(dp);events=[];proposals={};preview_rows=[]
    for s in segments:
        d=decisions[s['segment_id']]
        if d['action']=='keep':
            if d['candidate_id'] or d['path']:raise ValueError('Keep must not contain hidden edits')
            events.append(dict(segment_id=s['segment_id'],decision=d,status='kept_DP'));continue
        if d['action']=='candidate':
            if d['path']:raise ValueError('Candidate decision must not contain an extra path')
            matches=[c for c in pools[s['segment_id']] if c['candidate_id']==d['candidate_id']]
            if len(matches)!=1:raise ValueError('Unobserved candidate ID')
            candidate=matches[0]
        else:
            if d['candidate_id']:raise ValueError('Direct path must not contain a candidate ID')
            candidate=dict(candidate_id='model_direct',method='literal_model_authored_path',path=d['path'])
        proposed,verdict=preview(raw,dp,provisional,s,candidate,image.shape[:2],config)
        event=dict(segment_id=s['segment_id'],decision=d,candidate=candidate,preview_verdict=verdict,status='previewed')
        events.append(event);proposals[s['segment_id']]=candidate['path']
        preview_rows.append(dict(segment_id=s['segment_id'],decision=d,candidate=candidate,verdict=verdict))
        if verdict['accept']:provisional=proposed
    save(folder/'proposals.json',dict(events=events,previews=preview_rows))
    current=list(dp);review_packet=None
    if preview_rows:
        reviewed=[s for s in segments if s['segment_id'] in proposals]
        urls,mapping=atlas(image,reviewed,folder/'02_review',proposals)
        review_ctx=dict(phase='review',runtime_GT_access=False,image_mapping=mapping,
            previews=preview_rows,proposed_scene_verdict=verify(raw,dp,provisional,config),
            note='Green paths are literal proposals. Commit only valid and visually supported previews. A subset is reverified against fixed DP.')
        review_packet=request(review_ctx,urls,settings,REVIEW,folder/'02_review')
        check_ids(review_packet,[s['segment_id'] for s in reviewed])
        confirms={d['segment_id']:d for d in review_packet['decisions']}
        for s in reviewed:
            event=next(e for e in events if e['segment_id']==s['segment_id'])
            confirm=confirms[s['segment_id']];event['review']=confirm
            if not confirm['commit']:event['status']='abandoned';continue
            if not event['preview_verdict']['accept']:event['status']='rejected_invalid_preview';continue
            proposed,verdict=preview(raw,dp,current,s,event['candidate'],image.shape[:2],config)
            event['commit_verdict']=verdict
            if verdict['accept']:current=proposed;event['status']='committed'
            else:event['status']='rejected_at_commit'
    commits=[e for e in events if e['status']=='committed']
    trace=dict(model_calls=1+int(review_packet is not None),commits=len(commits),
        direct_committed=sum(e['candidate']['method']=='literal_model_authored_path' for e in commits),
        seconds=time.monotonic()-began,events=events,unfinished_target_ids=[],
        runtime_GT_access=False,stop_reason='Propose/review complete; no automatic rule fallback')
    save(folder/'trace.json',trace)
    return current,trace
