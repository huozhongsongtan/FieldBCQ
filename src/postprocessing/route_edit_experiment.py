"""Per-parcel visual tool routing, then literal visual refinement; GT offline."""
from pathlib import Path
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json,math,shutil,time,traceback
from fieldagent import activate_engine
from fieldagent.io import load,save,digest,now

NEW_METHODS=['tool_routed','tool_routed_edit']
BASE_METHODS=['raw','nth_point','radial_distance','dp','dp_visual_agent']
NAMES={'raw':'原始预测','nth_point':'固定步长抽点','radial_distance':'径向距离抽稀',
    'dp':'DP 1px','dp_visual_agent':'DP 1px＋视觉智能体','tool_routed':'智能体仅选工具',
    'tool_routed_edit':'智能体选工具＋视觉编辑'}


def freeze(source,out,page_size,workers):
    from fieldagent.experiment import completed_method,source_hashes
    fm=load(source/'manifest.json');original=Path(fm['source_run']);om=load(original/'manifest.json')
    if fm['fixed_DP_tolerance_px']!=1.:raise ValueError('Expected fixed DP 1px controls')
    files=['tool_router.py','routed_editor.py','route_edit_experiment.py','route_edit_report.py',
        'ROUTING_PROMPT.md','ROUTED_EDIT_PROMPT.md','scripts/run_route_edit_experiment.py']
    root=Path(__file__).parent;names=om['source_sha256'];old_sources={}
    for name,h in names.items():
        if digest(root/name)!=h:raise ValueError('Original control code changed: '+name)
        old_sources[name]=h
    if source_hashes()!=om['engine_sha256']:raise ValueError('Original engine changed')
    hashes={};route_calls=0
    for e in fm['entries']:
        if digest(e['image'])!=e['image_sha256']:raise ValueError('Image changed')
        for method in BASE_METHODS:
            if not completed_method(source,e,method):raise ValueError('Control incomplete')
            for name in ['final.pixel.geojson','completed.json','trace.json']:
                p=source/'scenes'/e['scene']/method/name;hashes[str(p.relative_to(source))]=digest(p)
        n=load(source/'scenes'/e['scene']/'raw/completed.json')['prediction_instances']
        route_calls+=math.ceil(n/page_size)
    protocol=dict(version='visual-tool-route-then-edit-v1',entries=fm['entries'],development_only=True,
        input_split=fm['input_split'],source_run=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        original_application_run=str(original),source_outputs_sha256=hashes,
        source_code_sha256={n:digest(root/n) for n in files},original_sources_sha256=old_sources,
        engine_sha256=source_hashes(),planner=om['planner'],config=om['config'],task_gates=om['task_gates'],
        tool_parameters=dict(load(source/'CALIBRATION.json')['selected'],dp=1.),
        routing_page_size=page_size,workers=workers,max_successful_routing_requests=route_calls,
        max_successful_refinement_requests=2*len(fm['entries']),
        max_successful_requests=route_calls+2*len(fm['entries']),
        runtime_GT_access=False,ground_truth_export=False,forced_tool_diversity=False,
        controls_reused=True,methods=BASE_METHODS+NEW_METHODS,
        semantics='Model sees three actual previews, chooses one exact full output per instance; a second propose-preview-review stage may author literal local paths. No oracle tool/parameter choice.',
        refinement_budget='Same 12 arcs per image / 3 per instance and gates as prior DP-only editor. BASE is now mixed selected tool outputs.',
        GT_use='Offline evaluation only after both routing and refinement geometry have been saved.',
        endpoint=om['planner']['base_url'],model_alias=om['planner']['model'])
    path=out/'manifest.json'
    if path.exists():
        if load(path)!=protocol:raise ValueError('Frozen route/edit protocol changed; use new output')
    else:
        if out.exists() and any(out.iterdir()):raise ValueError('Use empty output')
        save(path,protocol)
        for name in files:
            dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
    return protocol


def check_frozen(m,source):
    from fieldagent.experiment import source_hashes
    root=Path(__file__).parent
    if source_hashes()!=m['engine_sha256']:raise ValueError('Engine fingerprint changed')
    for n,h in dict(m['source_code_sha256'],**m['original_sources_sha256']).items():
        if digest(root/n)!=h:raise ValueError('Runtime source fingerprint changed: '+n)
    for n,h in m['source_outputs_sha256'].items():
        if digest(source/n)!=h:raise ValueError('Frozen control changed: '+n)


def process_scene(e,m,source,out):
    from cvr_agent import read_items,save_items
    from run_boundary_tools import load_image
    from dp_structure import StructureConfig
    from application_boundary import TaskGates,diagnose
    from fieldagent.experiment import completed_method,persist_method
    from tool_router import TOOLS,route_scene
    from routed_editor import run as refine
    began=time.perf_counter();scene=out/'scenes'/e['scene']
    if completed_method(out,e,'tool_routed_edit'):return dict(scene=e['scene'],reused_completed=True)
    if digest(e['image'])!=e['image_sha256']:raise ValueError('Input image changed')
    image=load_image(e['image']);raw=read_items(source/'scenes'/e['scene']/'raw/final.pixel.geojson')
    options={method:{p.identifier:p for p in read_items(source/'scenes'/e['scene']/method/'final.pixel.geojson')} for method in TOOLS}
    metadata={method:{r['instance_id']:r for r in load(source/'scenes'/e['scene']/method/'trace.json').get('events',[])} for method in TOOLS}
    route_folder=scene/'tool_routed';ready=scene/'routed_base.ready.json'
    if ready.exists():
        if digest(route_folder/'final.pixel.geojson')!=load(ready)['vector_sha256']:raise ValueError('Routed base changed')
        base=read_items(route_folder/'final.pixel.geojson');route_trace=load(route_folder/'trace.json')
    else:
        started=time.perf_counter()
        base,route_trace=route_scene(image,raw,options,m['tool_parameters'],metadata,m['planner'],route_folder/'run',m['routing_page_size'])
        route_trace['seconds']=time.perf_counter()-started
        save_items(route_folder/'final.pixel.geojson',base);save(route_folder/'trace.json',route_trace)
        save(ready,dict(vector_sha256=digest(route_folder/'final.pixel.geojson'),runtime_GT_access=False))
    config=StructureConfig(**m['config']);gates=TaskGates(**m['task_gates'])
    prep=scene/'preparation.json'
    if prep.exists():
        p=load(prep)
        if p['base_sha256']!=digest(route_folder/'final.pixel.geojson'):raise ValueError('Diagnosis base changed')
        diagnosis=load(scene/'diagnosis.json');pools=load(scene/'candidate_pool.json')
    else:
        started=time.perf_counter();diagnosis,pools=diagnose(raw,base,image,config,gates)
        save(scene/'diagnosis.json',diagnosis);save(scene/'candidate_pool.json',pools)
        p=dict(seconds=time.perf_counter()-started,base_sha256=digest(route_folder/'final.pixel.geojson'),
            selected_arcs=len(diagnosis['selected']),selected_instances=len({s['instance_id'] for s in diagnosis['selected']}),runtime_GT_access=False)
        save(prep,p)
    initialization=dict(method='model_selected_per_instance_tool',parameters=m['tool_parameters'],
        selected_tools=[dict(instance_id=d['instance_id'],tool=d['tool']) for d in route_trace['events']],
        note='Current base is the actual selected output. Internal legacy DP keys refer to this mixed BASE, not uniform DP.')
    started=time.perf_counter()
    final,edit_trace=refine(image,raw,base,diagnosis,pools,config,gates,m['planner'],scene/'tool_routed_edit/run',vision=True,initialization=initialization)
    edit_seconds=time.perf_counter()-started
    edit_trace.update(refinement_model_calls=edit_trace['model_calls'],routing_model_calls=route_trace['model_calls'],
        model_calls=route_trace['model_calls']+edit_trace['model_calls'],
        seconds=route_trace['seconds']+p['seconds']+edit_seconds,stage_order='visual tool routing, then visual local edit',
        tool_choices=route_trace['tool_choices'])
    save_items(scene/'tool_routed_edit/final.pixel.geojson',final);save(scene/'tool_routed_edit/trace.json',edit_trace)
    # Only now does offline evaluation receive the label path. It never enters
    # route_scene, diagnose, the refinement call, or any external request.
    for method,items,trace in [('tool_routed',base,route_trace),('tool_routed_edit',final,edit_trace)]:
        if not completed_method(out,e,method):persist_method(out,e,method,items,raw,image,trace)
    return dict(scene=e['scene'],seconds=time.perf_counter()-began,tool_choices=route_trace['tool_choices'],
        edit_commits=edit_trace['commits'],route_calls=route_trace['model_calls'],edit_calls=edit_trace['refinement_model_calls'])


def execute(m,source,out):
    from fieldagent.experiment import completed_method
    jobs=[e for e in m['entries'] if not completed_method(out,e,'tool_routed_edit')]
    start=time.perf_counter();finished=[];errors=[]
    with ThreadPoolExecutor(max_workers=m['workers']) as pool:
        todo=iter(jobs);pending={}
        def submit():
            try:e=next(todo)
            except StopIteration:return
            pending[pool.submit(process_scene,e,m,source,out)]=e
        for _ in range(m['workers']):submit()
        while pending:
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            if not done:continue
            for future in done:
                e=pending.pop(future)
                try:
                    result=future.result();finished.append(result)
                    print('ROUTE+EDIT',json.dumps(result,ensure_ascii=False),flush=True)
                except Exception as error:
                    record=dict(scene=e['scene'],error=type(error).__name__+': '+str(error),traceback=traceback.format_exc());errors.append(record)
                    print('ROUTE+EDIT ERROR',e['scene'],record['error'],flush=True)
                if not errors:submit()
            count=sum(completed_method(out,e,'tool_routed_edit') for e in m['entries'])
            ready=sum((out/'scenes'/e['scene']/'routed_base.ready.json').exists() for e in m['entries'])
            save(out/'progress.json',dict(completed_images=count,routed_base_ready=ready,expected_images=len(m['entries']),
                active=len(pending),session_seconds=time.perf_counter()-start,errors=errors,updated_utc=now()))
            print('ROUTE+EDIT PROGRESS',count,'/',len(m['entries']),'elapsed',round(time.perf_counter()-start,1),flush=True)
    session=dict(completed_utc=now(),wall_seconds=time.perf_counter()-start,workers=m['workers'],finished=finished,errors=errors)
    save(out/f'EXECUTION_SESSION_{time.time_ns()}.json',session)
    return 1 if errors else 0


def main(argv=None):
    p=argparse.ArgumentParser(description='Visual choose-tool then visual local-edit experiment')
    project=Path(__file__).resolve().parents[1]
    p.add_argument('stage',choices=['prepare','gpt','report'])
    p.add_argument('--source',type=Path,default=project/'results/runs/five-method-val50-20261025-v1')
    p.add_argument('--output',type=Path,default=project/'results/runs/tool-route-edit-val50-20261025-v1')
    p.add_argument('--page-size',type=int,default=24);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(argv)
    if not 1<=a.workers<=4 or not 1<=a.page_size<=32:p.error('workers=1..4 and page-size=1..32 required')
    activate_engine()
    import cv2
    cv2.setNumThreads(1)
    # Load once before worker threads, including native numerical dependencies.
    import tool_router,routed_editor,application_boundary
    source=a.source.resolve();out=a.output.resolve()
    if source==out or source in out.parents or out in source.parents:p.error('Use independent source/output directories')
    from fieldagent.experiment import stage_lock
    if a.stage=='prepare':
        m=freeze(source,out,a.page_size,a.workers)
        print(json.dumps(dict(images=len(m['entries']),parameters=m['tool_parameters'],
            routing_requests_max=m['max_successful_routing_requests'],refinement_requests_max=m['max_successful_refinement_requests'],
            requests_max=m['max_successful_requests'],external_calls_during_prepare=0,
            runtime_GT_access=False,endpoint=m['endpoint'],model_alias=m['model_alias']),ensure_ascii=False,indent=2));return 0
    m=load(out/'manifest.json');source=Path(m['source_run']);check_frozen(m,source)
    if a.stage=='report':
        from route_edit_report import report
        report(m,source,out);return 0
    with stage_lock(out,'route_edit'):
        code=execute(m,source,out);check_frozen(m,source)
        if code==0:
            from route_edit_report import report
            report(m,source,out)
    return code


if __name__=='__main__':raise SystemExit(main())
