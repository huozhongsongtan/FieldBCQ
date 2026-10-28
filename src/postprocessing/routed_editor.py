"""Visual local editing after per-instance tool routing; base need not be DP."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import time
import urllib.request
import urllib.error
import numpy as np
from fieldagent.io import load, save, digest
from fieldbqc_geometry import vertices
from fieldbqc_planner import APIPlanner
from dp_structure_agent import obj, XY, REVIEW, validate_schema, check_ids, atlas
from dp_structure import operation_summary
from dp_correction import ImageEvidence
from application_boundary import TASKS, preview, verify, edge_search

PROMPT=Path(__file__).with_name('ROUTED_EDIT_PROMPT.md')
PROPOSE=obj(dict(decisions=dict(type='array',items=obj(dict(
    segment_id=dict(type='string'),
    action=dict(type='string',enum=['keep','refer','candidate','direct','edge_search']),
    operation=dict(type='string',enum=list(TASKS)),candidate_id=dict(type='string'),
    path=dict(type='array',items=XY),evidence_point=XY,reason=dict(type='string'))))))


def request(context, urls, settings, schema, folder, prompt_path=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if not settings['vision'] and urls: raise ValueError('No-raster ablation must not send image URLs')
    prompt=Path(prompt_path or PROMPT).read_text(encoding='utf-8')
    transport_settings=dict(settings)
    if transport_settings['api']=='chat_completions': transport_settings['api']='chat'
    payload=APIPlanner(transport_settings,schema=schema,system_prompt=prompt)._payload(context,urls)
    if (folder/'request.json').exists():
        if load(folder/'request.json')!=payload: raise ValueError('Frozen request changed')
    else:
        save(folder/'context.json',context);save(folder/'request.json',payload)
        (folder/'system_prompt.md').write_text(prompt,encoding='utf-8')
    path=folder/'response.json'
    if not path.exists():
        if (folder/'started.json').exists():
            raise RuntimeError('Unreconciled request; inspect saved call before any paid replay: '+str(folder))
        key=os.environ.get(settings.get('api_key_env','OPENAI_API_KEY'))
        if not key: raise RuntimeError('Configured API key unavailable')
        endpoint=settings['base_url'].rstrip('/')+('/responses' if settings['api']=='responses' else '/chat/completions')
        meta=dict(endpoint=endpoint,requested_model=settings['model'],request_sha256=digest(folder/'request.json'),
            images_sent=len(urls),vision=settings['vision'],retries=0)
        save(folder/'started.json',meta);began=time.perf_counter()
        try:
            req=urllib.request.Request(endpoint,data=json.dumps(payload,allow_nan=False).encode('utf-8'),
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=settings.get('timeout_seconds',90)) as response:
                data=json.load(response)
            save(path,data)
            save(folder/'metadata.json',dict(meta,seconds=time.perf_counter()-began,usage=data.get('usage',{}),
                response_model=data.get('model'),response_id=data.get('id')))
        except Exception as error:
            message='HTTP '+str(error.code) if isinstance(error,urllib.error.HTTPError) else type(error).__name__+': '+str(error)
            save(folder/'error.json',dict(meta,error=message,seconds=time.perf_counter()-began))
            raise RuntimeError(message) from None
    data=load(path)
    if settings['api']=='responses':
        if data.get('status')!='completed': raise ValueError('Incomplete response retained')
        text=''.join(c.get('text','') for m in data.get('output',[]) if m.get('type')=='message'
            for c in m.get('content',[]) if c.get('type')=='output_text')
    else:
        choice=data['choices'][0]
        if choice.get('finish_reason')!='stop': raise ValueError('Incomplete response retained')
        text=choice['message']['content']
    packet=json.loads(text);validate_schema(packet,schema);save(folder/'decision.json',packet)
    return packet


def observed_segments(segments,pools):
    """Same geometry-only JSON in both arms; only attached rasters differ.

    Shared routing and search tools still use imagery in BOTH arms. Thus this is
    an ablation of planner image input, not an image-free entire pipeline.
    """
    result=[]
    for s in segments:
        n=len(s['raw_arc'])
        indices=sorted(set([0,n-1,s['farthest_index'],*[int(i) for i in np.linspace(0,n-1,min(n,24))]]))
        result.append(dict(segment_id=s['segment_id'],instance_id=s['instance_id'],ring_id=s['ring_id'],
            suspected_task=s['suspected_task'],suspected_task_is_unconfirmed=True,
            dp_anchor_path=s['dp_anchor_path'],endpoints=s['endpoints'],
            observed_raw_points=[dict(original_index=s['raw_indices'][i],xy=s['raw_arc'][i]) for i in indices],
            raw_observation_sampled=len(indices)<n,
            candidates=[{k:c[k] for k in ('candidate_id','method','path')} for c in pools[s['segment_id']]]))
    return result


def run(image,raw,dp,diagnosis,pools,config,gates,settings,folder,vision=True,initialization=None):
    folder=Path(folder);began=time.perf_counter();segments=diagnosis['selected']
    settings=dict(settings,vision=vision)
    events=[];ledger=[];current=list(dp);calls=0
    if segments:
        urls,mapping=atlas(image,segments,folder/'01_propose') if vision else ([],[])
        context=dict(phase='propose',planner_has_raster_input=vision,runtime_GT_access=False,
            goal='Further simplify and visually correct the actual per-instance routed BASE; preserve physical bends and author literal local paths when needed',
            initialization=initialization or dict(method='routed_initial_tools',role='selected base, not necessarily DP'),
            image_shape=list(image.shape[:2]),geometry_config=asdict(config),task_gates=asdict(gates),
            segments=observed_segments(segments,pools),image_mapping=mapping)
        packet=request(context,urls,settings,PROPOSE,folder/'01_propose');calls+=1
        check_ids(packet,[s['segment_id'] for s in segments])
        decisions={r['segment_id']:r for r in packet['decisions']};provisional=list(dp);provisional_ledger=[]
        previews=[];paths={};evidence=ImageEvidence(image)
        for s in segments:
            d=decisions[s['segment_id']];event=dict(segment_id=s['segment_id'],instance_id=s['instance_id'],
                suspected_task=s['suspected_task'],decision=d,task=d['operation'])
            events.append(event)
            if d['action'] in ('keep','refer'):
                if d['candidate_id'] or d['path']: raise ValueError('Keep/refer cannot contain hidden edits')
                event['status']='needs_human_review' if d['action']=='refer' else 'kept_DP';continue
            if d['action']=='candidate':
                if d['path']: raise ValueError('Candidate choice cannot contain an extra path')
                matches=[c for c in pools[s['segment_id']] if c['candidate_id']==d['candidate_id']]
                if len(matches)!=1: raise ValueError('Unknown candidate')
                candidate=matches[0]
                if candidate['path']==s['dp_anchor_path']:
                    event['status']='kept_DP';continue
            else:
                if d['candidate_id']: raise ValueError('Direct/search cannot contain candidate ID')
                candidate=dict(candidate_id='model_direct',method='literal_model_authored_path',path=d['path'])
                if d['action']=='edge_search':
                    # Validate hint contract BEFORE any image search can alter it.
                    from dp_structure import apply_path
                    apply_path(provisional,s,d['path'],image.shape[:2],config)
                    path,audit=edge_search(d['path'],evidence,gates)
                    candidate=dict(candidate_id='model_requested_search',method='normal_control_search',path=path,tool_audit=audit)
            proposed,verdict,record=preview(raw,dp,provisional,s,candidate,d['operation'],image.shape[:2],config,gates,provisional_ledger)
            event.update(candidate=candidate,preview_verdict=verdict,status='previewed',
                operation=operation_summary(s,candidate['path']),tool_calls=[dict(tool=candidate['method'],path=candidate['path'])])
            previews.append(dict(segment_id=s['segment_id'],task=d['operation'],
                candidate={k:candidate[k] for k in ('candidate_id','method','path')},
                before_path=s['dp_anchor_path'],verdict=verdict,operation=event['operation']))
            paths[s['segment_id']]=candidate['path']
            if verdict['accept']: provisional=proposed;provisional_ledger.append(record)
        save(folder/'proposals.json',dict(events=events,previews=previews))
        if previews:
            reviewed=[s for s in segments if s['segment_id'] in paths]
            urls,mapping=atlas(image,reviewed,folder/'02_review',paths) if vision else ([],[])
            context=dict(phase='review',planner_has_raster_input=vision,runtime_GT_access=False,
                previews=previews,image_mapping=mapping,
                goal='Confirm only visually supported further simplification or correction; geometric acceptance alone does not establish boundary accuracy',
                proposed_scene_verdict=verify(raw,dp,provisional,config,gates,provisional_ledger))
            packet=request(context,urls,settings,REVIEW,folder/'02_review');calls+=1
            check_ids(packet,[s['segment_id'] for s in reviewed]);confirms={r['segment_id']:r for r in packet['decisions']}
            for s in reviewed:
                e=next(e for e in events if e['segment_id']==s['segment_id']);e['review']=confirms[s['segment_id']]
                if not e['review']['commit']: e['status']='abandoned';continue
                if not e['preview_verdict']['accept']: e['status']='rejected_invalid_preview';continue
                proposed,verdict,record=preview(raw,dp,current,s,e['candidate'],e['task'],image.shape[:2],config,gates,ledger)
                e['commit_verdict']=verdict
                if verdict['accept']: current=proposed;ledger.append(record);e['status']='committed'
                else: e['status']='rejected_at_commit'
    final_check=verify(raw,dp,current,config,gates,ledger)
    if not final_check['accept']: raise ValueError('Final cumulative check failed')
    commits=[e for e in events if e['status']=='committed']
    trace=dict(model_calls=calls,commits=len(commits),events=events,ledger=ledger,
        direct_committed=sum(e['candidate']['method']=='literal_model_authored_path' for e in commits),
        commits_by_task={t:sum(e['task']==t for e in commits) for t in TASKS},
        seconds=time.perf_counter()-began,unfinished_target_ids=[],runtime_GT_access=False,
        planner_has_raster_input=vision,shared_executor_uses_image=True,
        net_vertex_change=sum(vertices(p.geometry) for p in current)-sum(vertices(p.geometry) for p in dp),
        deleted_existing_DP_points=sum(len(e['operation']['deleted_DP_coordinates']) for e in commits),
        added_control_points=sum(len(e['operation']['added_control_coordinates']) for e in commits),
        final_verdict=final_check,stop_reason='Propose/review complete' if segments else 'No selected risks; DP unchanged')
    save(folder/'trace.json',trace)
    return current,trace
