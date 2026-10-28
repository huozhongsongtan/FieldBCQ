"""Final five-arm DP1 protocol; route-only is a separate stage ablation."""
import argparse,csv,json,shutil,time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from fieldagent import activate_engine
from fieldagent.io import load,save,digest,now,read_inputs,planner_config

VERSION='fieldagent-final-dp1-three-tools-visual-edit-v1'
LOCAL=['raw','nth_point','radial_distance','dp']
MAIN=LOCAL+['tool_routed_edit']
ABLATION='tool_routed'
PARAMETERS=dict(nth_point=12,radial_distance=8.,dp=1.)
NAMES=dict(raw='原始预测',nth_point='固定步长抽点',radial_distance='径向距离抽稀',
    dp='DP 1px',tool_routed='智能体仅选工具（消融）',tool_routed_edit='智能体选工具＋视觉编辑')

def contract(m):
    if m['tool_parameters']!=PARAMETERS or m['config']['dp_tolerance']!=1.:
        raise ValueError('Final protocol requires fixed stride12 / radial8px / DP1px')
    if m['main_methods']!=MAIN or m['ablation_methods']!=[ABLATION]:raise ValueError('Final method groups changed')

def freeze(inputs,out,planner,workers=4):
    from fieldagent.experiment import source_hashes
    from dp_structure import StructureConfig
    from application_boundary import TaskGates
    names=['final_protocol.py','scripts/run_final_protocol.py','simple_sampling.py','tool_router.py',
        'routed_editor.py','route_edit_experiment.py','ROUTING_PROMPT.md','ROUTED_EDIT_PROMPT.md',
        'application_boundary.py','dp_structure.py','dp_correction.py','dp_structure_agent.py',
        'compact_inputs.py','final_experiment.py','evaluate_final.py','application_experiment.py']
    root=Path(__file__).parent
    m=dict(version=VERSION,entries=read_inputs(inputs),inputs_sha256=digest(inputs),
        input_split=load(inputs).get('split','undeclared'),development_only=load(inputs).get('development_only',True),
        main_methods=MAIN,ablation_methods=[ABLATION],methods=MAIN+[ABLATION],tool_parameters=PARAMETERS,
        config=asdict(StructureConfig(dp_tolerance=1.)),task_gates=asdict(TaskGates()),
        planner=planner_config(planner),workers=workers,routing_page_size=24,
        source_code_sha256={n:digest(root/n) for n in names},engine_sha256=source_hashes(),
        runtime_GT_access=False,GT_use='Offline evaluation only; no labels in model payloads',
        selection='Fixed development parameters; no per-test-set recalibration or GT selection')
    contract(m)
    if (out/'manifest.json').exists():
        old=load(out/'manifest.json')
        if {k:v for k,v in old.items() if k!='created_utc'}!=m:raise ValueError('Frozen configuration changed; use a new output')
        return old
    if out.exists() and any(out.iterdir()):raise ValueError('New output must be empty')
    save(out/'manifest.json',dict(m,created_utc=now()))
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/n,dest)
    return load(out/'manifest.json')

def check(m):
    from fieldagent.experiment import source_hashes,check_inputs
    if m['version']!=VERSION:raise ValueError('Not the final five-arm protocol')
    contract(m);root=Path(__file__).parent
    for n,h in m['source_code_sha256'].items():
        if digest(root/n)!=h:raise ValueError('Frozen source changed: '+n)
    if source_hashes()!=m['engine_sha256']:raise ValueError('Frozen engine changed')
    for e in m['entries']:check_inputs(e)

def freeze_controls(m,out):
    from fieldagent.experiment import completed_method
    if not all(completed_method(out,e,k) for e in m['entries'] for k in LOCAL):return
    hashes={}
    for e in m['entries']:
        for k in LOCAL:
            for name in ['completed.json','trace.json','final.pixel.geojson']:
                p=out/'scenes'/e['scene']/k/name;hashes[str(p.relative_to(out))]=digest(p)
    path=out/'LOCAL_OUTPUT_HASHES.json'
    if path.exists() and load(path)!=hashes:raise ValueError('Local baseline files changed')
    if not path.exists():save(path,hashes)

def local(m,out):
    from final_experiment import load_scene
    from dp_structure import initialize,StructureConfig
    from simple_sampling import simplify_items
    from application_experiment import persist
    from fieldagent.experiment import completed_method
    config=StructureConfig(**m['config'])
    for i,e in enumerate(m['entries'],1):
        missing=[method for method in LOCAL if not completed_method(out,e,method)]
        if not missing:continue
        image,raw=load_scene(e,out)
        for method in missing:
            began=time.perf_counter()
            if method=='raw':items,trace=raw,dict(method=method)
            elif method=='dp':items,trace=initialize(raw,config),dict(method=method,tolerance_px=1.)
            else:items,trace=simplify_items(raw,method,m['tool_parameters'][method])
            trace.update(seconds=time.perf_counter()-began,model_calls=0,runtime_GT_access=False)
            persist(out,e,method,items,raw,image,trace)
        print(f'FINAL LOCAL {i}/{len(m["entries"])} {e["scene"]}',flush=True)

def import_verified(m,out,source):
    """Copy finished geometry/metrics only; keep original API logs at source."""
    from fieldagent.experiment import completed_method
    sm=load(source/'manifest.json');contract(dict(sm,main_methods=MAIN,ablation_methods=[ABLATION]))
    source_entries={e['scene']:e for e in sm['entries']}
    controls=Path(sm['source_run']);hashes={}
    from route_edit_experiment import check_frozen
    check_frozen(sm,controls)
    for e in m['entries']:
        old=source_entries[e['scene']]
        for key in ['image_sha256','prediction_vector_sha256','label_sha256']:
            if e.get(key)!=old.get(key):raise ValueError('Source inputs differ: '+key)
        for method in MAIN+[ABLATION]:
            root=source if method.startswith('tool_') else controls
            if not completed_method(root,e,method):raise ValueError('Incomplete source: '+method)
            for name in ['completed.json','trace.json','final.pixel.geojson']:
                src=root/'scenes'/e['scene']/method/name;dest=out/'scenes'/e['scene']/method/name
                if dest.exists() and digest(dest)!=digest(src):raise ValueError('Conflicting imported result')
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);hashes[str(src)]=digest(src)
        for name in ['preparation.json','diagnosis.json','candidate_pool.json']:
            src=source/'scenes'/e['scene']/name
            if src.exists():shutil.copy2(src,out/'scenes'/e['scene']/name)
    save(out/'IMPORT_PROVENANCE.json',dict(source=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        source_files_sha256=hashes,new_API_calls=0,source_API=load(source/'summary.json')['API'],
        actual_requests_and_previews='Preserved at the original source; not regenerated or called again'))

def report(m,out):
    from fieldagent.experiment import completed_method,aggregate,csv_file
    from evaluate_final import bootstrap
    methods=MAIN+[ABLATION];entries=m['entries'];records={};coverage={}
    for method in methods:
        records[method]=[load(out/'scenes'/e['scene']/method/'completed.json') for e in entries if completed_method(out,e,method)]
        coverage[method]=len(records[method])
    save(out/'progress.json',dict(expected_images=len(entries),complete_by_method=coverage,main_methods=MAIN,ablation_methods=[ABLATION]))
    # Local controls are independently reviewable before paid agent execution.
    if all(coverage[k]==len(entries) for k in LOCAL):
        csv_file(out/'LOCAL_METRICS.csv',[dict(method=k,**aggregate(records[k])) for k in LOCAL])
    if any(n!=len(entries) for n in coverage.values()):
        print(json.dumps(dict(complete_by_method=coverage,complete=False),ensure_ascii=False),flush=True);return
    agg={method:aggregate(records[method]) for method in methods};counts=[];choices=Counter();ops=Counter();prep=[];hashes={};paired=[]
    for e in entries:
        scene=out/'scenes'/e['scene'];maps={}
        for method in methods:
            r=load(scene/method/'completed.json');maps[method]={p['instance_id']:p['after'] for p in r['parcel_vertices']}
            hashes[str(scene/method/'final.pixel.geojson')]=r['final_sha256']
        if any(set(v)!=set(maps['raw']) for v in maps.values()):raise ValueError('Method instance IDs differ')
        decisions={d['instance_id']:d for d in load(scene/ABLATION/'trace.json')['events']}
        choices.update(load(scene/ABLATION/'trace.json')['tool_choices'])
        for ident in maps['raw']:counts.append(dict(scene=e['scene'],instance_id=ident,chosen_tool=decisions[ident]['tool'],
            reason=decisions[ident]['reason'],**{k:v[ident] for k,v in maps.items()}))
        t=load(scene/'tool_routed_edit/trace.json');ops.update(t.get('commits_by_task',{}))
        prep.append(load(scene/'preparation.json'))
    comparisons=[]
    for method,ref in [('tool_routed','dp'),('tool_routed_edit','tool_routed'),('tool_routed_edit','dp')]:
        ds=[]
        for a,b in zip(records[ref],records[method]):
            if a['scene']!=b['scene']:raise ValueError('Unpaired images')
            av=a.get('evaluation',{}).get('metrics',{}).get('boundary_f1')
            bv=b.get('evaluation',{}).get('metrics',{}).get('boundary_f1')
            delta=bv-av if av is not None and bv is not None else None
            if delta is not None:ds.append(delta)
            paired.append(dict(scene=a['scene'],method=method,reference=ref,delta_BF1=delta,delta_vertices=b['vertices']-a['vertices']))
        comparisons.append(dict(method=method,reference=ref,extra_removed=agg[ref]['vertices_after']-agg[method]['vertices_after'],BF1_delta=bootstrap(ds)))
    inherited=load(out/'IMPORT_PROVENANCE.json') if (out/'IMPORT_PROVENANCE.json').exists() else None
    calls=[]
    for p in (out/'scenes').glob('*/tool_*/run/*/metadata.json'):
        c=load(p);context=load(p.with_name('context.json'))
        if context['runtime_GT_access'] is not False:raise ValueError('Runtime GT access')
        calls.append(dict(call=str(p.relative_to(out)),**{k:c.get(k) for k in ['seconds','images_sent','response_model','usage']}))
    sessions=[load(p) for p in out.glob('EXECUTION_SESSION_*.json')]
    s=dict(version=VERSION,images=len(entries),instances=agg['raw']['instances'],main_methods=MAIN,ablation_methods=[ABLATION],
        main_metrics={k:agg[k] for k in MAIN},ablation_metrics={ABLATION:agg[ABLATION]},comparisons=comparisons,
        tool_choices=dict(choices),commits_by_task=dict(ops),coverage=dict(selected_instances=sum(p['selected_instances'] for p in prep),
        selected_arcs=sum(p['selected_arcs'] for p in prep)),new_API_calls=len(calls),
        new_execution_seconds=sum(p['wall_seconds'] for p in sessions),imported_results=inherited is not None,
        original_API=inherited['source_API'] if inherited else None,runtime_GT_access=False,
        development_only=m['development_only'],claim='Point count / boundary quality tradeoff; stable superiority not established')
    save(out/'summary.json',s);save(out/'FINAL_VECTOR_HASHES.json',hashes)
    csv_file(out/'MAIN_METRICS.csv',[dict(method=k,**agg[k]) for k in MAIN])
    csv_file(out/'ABLATION_METRICS.csv',[dict(method=ABLATION,**agg[ABLATION])]);csv_file(out/'PARCEL_COUNTS.csv',counts)
    csv_file(out/'PAIRED_DELTAS.csv',paired);save(out/'API_AUDIT.json',dict(new_successful_calls=calls,import_provenance=inherited))
    lines=['# 最终五组对比：DP 1px、三工具选择与视觉编辑','',f"{len(entries)}张，{s['instances']}个预测地块。主表固定五组，仅工具选择另作消融。",'',
        '|方法|总点数|平均点数|BF1@2px↑|Chamfer px↓|','|---|---:|---:|---:|---:|']
    def fmt(value,digits=6):return '—' if value is None else f'{value:.{digits}f}'
    for k in MAIN:
        v=agg[k];lines.append(f"|{NAMES[k]}|{v['vertices_after']}|{fmt(v['mean_vertices'],4)}|{fmt(v['boundary_f1_macro'])}|{fmt(v['chamfer_macro_px'])}|")
    v=agg[ABLATION];lines+=['',f"仅选工具消融：{v['vertices_after']}点，平均{fmt(v['mean_vertices'],4)}点，BF1={fmt(v['boundary_f1_macro'])}。",'']
    for c in comparisons:
        delta=c['BF1_delta']['mean'];delta_text='—' if delta is None else f'{delta:+.6f}'
        lines.append(f"- {NAMES[c['method']]}相对{NAMES[c['reference']]}：净少{c['extra_removed']:+d}点，BF1变化{delta_text}，探索性95%区间{c['BF1_delta']['ci95']}。")
    lines+=['',f"工具选择：{dict(choices)}。视觉编辑仅观察{s['coverage']['selected_instances']}个地块/{s['coverage']['selected_arcs']}段；不是每个地块均直接编辑。",
        f"本目录新增成功API调用{s['new_API_calls']}次。"+('本表逐文件校验复用已完成真实GPT实验；原始调用及耗时见IMPORT_PROVENANCE.json和来源实验。' if inherited else '调用、预览、响应和反馈在scenes目录；执行会话见EXECUTION_SESSION。'),
        '少点与精度下降应同时报告。验证集开发、单次API及未完成人工核验限制结论；区间跨零不能证明精度无损。GT只在矢量保存后离线评价。1px只是DP候选容差，不是最终智能体输出误差上界。','']
    (out/'最终五组实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
    for p,h in hashes.items():
        if digest(p)!=h:raise ValueError('Final geometry changed')
    print(json.dumps(dict(complete=True,images=s['images'],main_methods=MAIN,new_API_calls=len(calls),output=str(out)),ensure_ascii=False),flush=True)

def main(argv=None):
    p=argparse.ArgumentParser(description='Final: raw / stride12 / radial8 / DP1 / visual routing + editing')
    p.add_argument('stage',choices=['local','gpt','report','import'])
    p.add_argument('--inputs',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--planner',type=Path)
    p.add_argument('--source',type=Path,help='Finished DP1 route/edit run to import without API calls')
    p.add_argument('--workers',type=int,default=4,choices=range(1,5));a=p.parse_args(argv)
    if a.stage in ['local','import'] and (a.inputs is None or a.planner is None):
        p.error('--inputs and --planner are required for local/import')
    activate_engine()
    import cv2
    cv2.setNumThreads(1)
    import tool_router,routed_editor,application_boundary
    from fieldagent.experiment import stage_lock,completed_method
    out=a.output.resolve()
    if a.stage in ['local','import']:m=freeze(a.inputs,out,a.planner,a.workers)
    else:m=load(out/'manifest.json')
    check(m)
    with stage_lock(out,'final_protocol'):
        if (out/'LOCAL_OUTPUT_HASHES.json').exists():freeze_controls(m,out)
        if a.stage=='local':local(m,out)
        elif a.stage=='import':
            if not a.source:p.error('--source required for import')
            source=a.source.resolve()
            if source==out or source in out.parents or out in source.parents:p.error('Use independent output')
            import_verified(m,out,source)
        elif a.stage=='gpt':
            if not all(completed_method(out,e,k) for e in m['entries'] for k in LOCAL):raise ValueError('Run local first')
            from route_edit_experiment import execute
            if execute(m,out,out):return 1
        check(m);freeze_controls(m,out);report(m,out)
    return 0

if __name__=='__main__':raise SystemExit(main())
