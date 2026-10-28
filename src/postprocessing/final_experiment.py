"""Final three-arm experiment: raw predictions, DP, DP plus visual agent."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time
from fieldagent import activate_engine
from fieldagent.io import load,save,digest,now,read_inputs,planner_config

METHODS=['raw','dp','dp_visual_agent']
VERSION='fieldagent-final-three-v1'


def freeze(inputs,out,config,settings,tolerances=(1.,2.,3.,4.,6.,8.)):
    from fieldagent.experiment import source_hashes
    paths=[Path(__file__),Path(__file__).with_name('dp_structure.py'),
           Path(__file__).with_name('dp_structure_agent.py'),Path(__file__).with_name('DP_STRUCTURE_PROMPT.md'),Path(__file__).with_name('dp_correction.py'),
           Path(__file__).with_name('compact_inputs.py'), Path(__file__).with_name('evaluate_final.py'),
           Path(__file__).with_name('analyze_final.py'), Path(__file__).with_name('evaluate_dp_structure.py')]
    record=dict(version=VERSION,entries=read_inputs(inputs),inputs_sha256=digest(inputs),
        methods=METHODS,config=asdict(config),planner=settings,max_calls_per_scene=2,
        dp_sweep_tolerances_px=list(tolerances),
        runtime_GT_access=False,development_only=True,
        source_sha256={p.name:digest(p) for p in paths},engine_sha256=source_hashes(),
        fairness='Raw is the common prediction. DP and agent use the same fixed DP tolerance. Agent starts from that DP, sees image/prediction only, may choose candidates or author literal paths, and must preview/commit passing edits. No deterministic repair branch is executed. Sweep is a separate diagnostic, never selected per test image using GT. Development status must be declared before interpreting results.')
    record=json.loads(json.dumps(record))
    if (out/'manifest.json').exists():
        old=load(out/'manifest.json')
        if {k:v for k,v in old.items() if k!='created_utc'}!=record:raise ValueError('Frozen code/settings/inputs changed: use new output')
        return old
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use an empty output directory')
    (out/'source_snapshot').mkdir(exist_ok=True)
    for p in paths:shutil.copyfile(p,out/'source_snapshot'/p.name)
    save(out/'manifest.json',dict(record,created_utc=now()))
    return load(out/'manifest.json')


def load_scene(entry,out):
    # Only image/prediction enter the actual generation functions.
    from compact_inputs import prepare
    from fieldagent.experiment import read_scene
    derived,_=prepare(entry,out)
    return read_scene(derived)


def run_local(manifest,out,config):
    from fieldagent.experiment import persist_method,completed_method
    from dp_structure import initialize,ImageEvidence,detect,candidate_paths
    for number,entry in enumerate(manifest['entries'],1):
        scene=out/'scenes'/entry['scene']
        if all(completed_method(out,entry,m) for m in METHODS[:2]) and all((scene/n).exists() for n in ['diagnosis.json','candidate_pool.json','preparation.json']):continue
        image,raw=load_scene(entry,out);begin=time.monotonic()
        if not completed_method(out,entry,'raw'):persist_method(out,entry,'raw',raw,raw,image)
        began=time.monotonic();dp=initialize(raw,config);dp_seconds=time.monotonic()-began
        if not completed_method(out,entry,'dp'):
            trace=dict(seconds=dp_seconds,model_calls=0,commits=0,direct_committed=0,unfinished_target_ids=[],
                controller='Shapely_topology_preserving_DP',tolerance_px=config.dp_tolerance)
            save(out/'scenes'/entry['scene']/'dp/trace.json',trace)
            persist_method(out,entry,'dp',dp,raw,image,trace)
        began=time.monotonic();evidence=ImageEvidence(image);diagnosis=detect(raw,dp,evidence,config)
        pools={s['segment_id']:candidate_paths(s,evidence,config) for s in diagnosis['selected']}
        preparation=time.monotonic()-began
        save(out/'scenes'/entry['scene']/'diagnosis.json',diagnosis)
        save(out/'scenes'/entry['scene']/'candidate_pool.json',pools)
        save(out/'scenes'/entry['scene']/'preparation.json',dict(dp_seconds=dp_seconds,shared_diagnosis_candidate_seconds=preparation,runtime_GT_access=False))
        print(f'LOCAL {number}/{len(manifest["entries"])} {entry["scene"]} suspects={diagnosis["detected"]} selected={diagnosis["selected_count"]}',flush=True)



def run_gpt(manifest,out,config):
    from fieldagent.experiment import persist_method,completed_method
    from cvr_agent import read_items
    from dp_structure_agent import run
    for number,entry in enumerate(manifest['entries'],1):
        if completed_method(out,entry,'dp_visual_agent'):continue
        if not all(completed_method(out,entry,m) for m in METHODS[:2]):raise ValueError('Run local stage first')
        image,raw=load_scene(entry,out)
        scene=out/'scenes'/entry['scene'];dp=read_items(scene/'dp/final.pixel.geojson')
        diagnosis=load(scene/'diagnosis.json');pools=load(scene/'candidate_pool.json')
        final,trace=run(image,raw,dp,diagnosis,pools,config,manifest['planner'],scene/'dp_visual_agent/run')
        prep=load(scene/'preparation.json');trace['correction_stage_active_seconds']=trace['seconds']
        trace['seconds']+=prep['dp_seconds']+prep['shared_diagnosis_candidate_seconds']
        trace['timing_note']='Sum of measured DP/preparation and current active correction stage. Resumed API duration must be read from call metadata; includes image/JSON work, excludes offline evaluation.'
        save(scene/'dp_visual_agent/trace.json',trace)
        persist_method(out,entry,'dp_visual_agent',final,raw,image,trace)
        print(f'VISUAL {number}/{len(manifest["entries"])} {entry["scene"]} calls={trace["model_calls"]} commits={trace["commits"]}',flush=True)


def report(manifest,out):
    from fieldagent.experiment import aggregate,csv_file,completed_method
    records={}
    for entry in manifest['entries']:
        for m in METHODS:
            if completed_method(out,entry,m):records[entry['scene'],m]=load(out/'scenes'/entry['scene']/m/'completed.json')
    result=dict(version=VERSION,development_only=True,expected_images=len(manifest['entries']),
        complete_by_method={m:sum(key[1]==m for key in records) for m in METHODS},groups={})
    table=[];parcels=[];scene_rows=[]
    for group,methods in [('local',METHODS[:2]),('three_arm',METHODS)]:
        chosen=[e for e in manifest['entries'] if all((e['scene'],m) in records for m in methods)]
        aggregates={m:aggregate([records[e['scene'],m] for e in chosen]) for m in methods} if chosen else {}
        result['groups'][group]=dict(images=len(chosen),scenes=[e['scene'] for e in chosen],aggregate=aggregates)
        for m,stats in aggregates.items():table.append(dict(comparison=group,method=m,**stats))
        result['groups'][group]['by_source']={d:{m:aggregate([records[e['scene'],m] for e in chosen if e['domain']==d]) for m in methods} for d in sorted({e['domain'] for e in chosen})}
    for (scene,m),record in records.items():
        parcels.extend(dict(scene=scene,method=m,**p) for p in record['parcel_vertices'])
        scene_rows.append(dict(scene=scene,method=m,vertices=record['vertices'],
            boundary_f1=record.get('evaluation',{}).get('metrics',{}).get('boundary_f1'),
            chamfer_px=record.get('evaluation',{}).get('metrics',{}).get('pixel_boundary_chamfer'),
            commits=record['commits'],calls=record['model_calls'],seconds=record['seconds']))
    calls=[]
    for path in sorted((out/'scenes').glob('*/dp_visual_agent/run/*/started.json')):
        folder=path.parent;terminal=folder/'metadata.json'
        meta=load(terminal if terminal.exists() else folder/'error.json') if terminal.exists() or (folder/'error.json').exists() else load(path)
        usage=meta.get('usage',{})
        calls.append(dict(scene=path.parents[3].name,phase=folder.name,call_folder=str(folder),
            completed_response=(folder/'response.json').exists(),validated_decision=(folder/'decision.json').exists(),
            input_tokens=usage.get('input_tokens'),output_tokens=usage.get('output_tokens'),
            seconds=meta.get('seconds'),error=meta.get('error'),response_model=meta.get('response_model')))
    reused={e['scene'] for e in manifest['entries'] if (out/'scenes'/e['scene']/'reuse_provenance.json').exists()}
    result['reused_scenes']=sorted(reused)
    result['API']=dict(new_attempts=sum(r['scene'] not in reused for r in calls),historical_attempts=sum(r['scene'] in reused for r in calls),attempts=len(calls),completed_responses=sum(r['completed_response'] for r in calls),
        input_tokens_reported=sum(r['input_tokens'] or 0 for r in calls),output_tokens_reported=sum(r['output_tokens'] or 0 for r in calls),cost=None)
    delta=[]
    for entry in manifest['entries']:
        scene=entry['scene']
        if not all((scene,m) in records for m in METHODS):continue
        for ref in ['dp']:
            a=records[scene,ref];b=records[scene,'dp_visual_agent']
            am=a.get('evaluation',{}).get('metrics',{});bm=b.get('evaluation',{}).get('metrics',{})
            bf=None if am.get('boundary_f1') is None or bm.get('boundary_f1') is None else bm['boundary_f1']-am['boundary_f1']
            delta.append(dict(scene=scene,reference=ref,delta_vertices=b['vertices']-a['vertices'],delta_BF1=bf,
                BF1_outcome=None if bf is None else ('improved' if bf>1e-8 else 'degraded' if bf< -1e-8 else 'tied')))
    csv_file(out/'METRICS.csv',table);csv_file(out/'PARCEL_COUNTS.csv',parcels)
    csv_file(out/'SCENE_COUNTS.csv',scene_rows);csv_file(out/'API_CALLS.csv',calls);csv_file(out/'PAIRED_DELTAS.csv',delta)
    save(out/'summary.json',result)
    lines=['# 原始预测、DP、DP＋视觉智能体：三组实验','', '主表仅包含三组共同完成的图像；local表为原始与DP的阶段进度。BF1为GT 2px；GT只在最终矢量保存后读取。此入口默认标记为开发实验，不能直接宣称独立盲测。','']
    for group,data in result['groups'].items():
        lines += [f'## {group}：{data["images"]}张','', '|方法|总点数|平均点数|中位数|减点率|BF1|Chamfer/px|提交|调用|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        def fmt(x):return '—' if x is None else f'{x:.4f}'
        for m,s in data['aggregate'].items():
            lines.append(f'|{m}|{s["vertices_after"]}|{fmt(s["mean_vertices"])}|{fmt(s["median_vertices"])}|{fmt(s["reduction"])}|{fmt(s["boundary_f1_macro"])}|{fmt(s["chamfer_macro_px"])}|{s["commits"]}|{s["attempted_model_calls"]}|')
        lines.append('')
    lines += ['API用量：'+json.dumps(result['API'],ensure_ascii=False),'',
        'DP贡献与模型增量分开解释。允许少量必要增点；不能把几何验收通过当作真实纠错成功。',
        '选择性定位仅覆盖一部分疑似边段，无触发图像不调用GPT，保持DP；导出图数不等于实际调用图数。',
        '运行evaluate_final.py可补充BF1@1/2/3px、前景并集IoU、配对区间和三栏图；运行sweep阶段可输出DP容差曲线。不得把曲线上的GT最优值当作每张图的自动选参结果。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def sweep_report(manifest,out):
    """DP sweep is a diagnostic curve, not an extra main experiment arm."""
    from fieldagent.experiment import aggregate,csv_file,completed_method
    rows=[]
    for tolerance in manifest['dp_sweep_tolerances_px']:
        folder=out/'dp_sweep'/('eps_'+str(tolerance).replace('.','p'))
        entries=[e for e in manifest['entries'] if completed_method(folder,e,'dp')]
        if entries:
            rows.append(dict(tolerance_px=tolerance,**aggregate([load(folder/'scenes'/e['scene']/'dp/completed.json') for e in entries])))
    csv_file(out/'DP_SWEEP.csv',rows)
    save(out/'DP_SWEEP.json',dict(expected_images=len(manifest['entries']),rows=rows,
        selection_uses_GT=False,note='Compare complete equal-image curves only; no per-image GT-based tolerance choice'))
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for r in rows:
            if r['images']!=len(manifest['entries']) or r.get('boundary_f1_macro') is None:continue
            ax.scatter(r['vertices_after'],r['boundary_f1_macro'],color='tab:blue')
            ax.annotate(f"DP eps={r['tolerance_px']:g}",(r['vertices_after'],r['boundary_f1_macro']),fontsize=8)
        summary=load(out/'summary.json')
        group=summary['groups']['three_arm']
        if group['images']==len(manifest['entries']):
            r=group['aggregate']['dp_visual_agent']
            if r.get('boundary_f1_macro') is not None:
                ax.scatter(r['vertices_after'],r['boundary_f1_macro'],marker='*',s=150,color='tab:red',label='DP + visual agent')
                ax.legend()
        ax.set_xlabel('Total vertices (same images)');ax.set_ylabel('Boundary F1 @ 2 px')
        ax.set_title('DP tolerance / vertex-quality diagnostic')
        (out/'figures').mkdir(exist_ok=True)
        fig.savefig(out/'figures/DP_VERTEX_QUALITY.png',dpi=160);plt.close(fig)
    return rows


def run_sweep(manifest,out,config):
    from dataclasses import replace
    from dp_structure import initialize
    from fieldagent.experiment import completed_method,persist_method
    for number,entry in enumerate(manifest['entries'],1):
        pending=[]
        for tolerance in manifest['dp_sweep_tolerances_px']:
            folder=out/'dp_sweep'/('eps_'+str(tolerance).replace('.','p'))
            if not completed_method(folder,entry,'dp'):pending.append((tolerance,folder))
        if not pending:continue
        image,raw=load_scene(entry,out)
        for tolerance,folder in pending:
            begin=time.monotonic();dp=initialize(raw,replace(config,dp_tolerance=tolerance))
            trace=dict(seconds=time.monotonic()-begin,model_calls=0,commits=0,direct_committed=0,
                unfinished_target_ids=[],controller='Shapely_topology_preserving_DP',tolerance_px=tolerance)
            save(folder/'scenes'/entry['scene']/'dp/trace.json',trace)
            persist_method(folder,entry,'dp',dp,raw,image,trace)
        print(f'SWEEP {number}/{len(manifest["entries"])} {entry["scene"]} tolerances={len(pending)}',flush=True)


def import_results(manifest,out,source):
    """Explicit verified reuse; never label imported API calls as new experiments."""
    from fieldagent.experiment import completed_method
    from cvr_agent import read_items,state_key
    prior=load(source/'manifest.json')
    if prior.get('version')!='dp-structure-removal-and-correction-v1':
        raise ValueError('Import supports only the validated historical DP-structure protocol')
    for key in ['entries','config','planner','engine_sha256']:
        if prior.get(key)!=manifest.get(key):raise ValueError('Import mismatch: '+key)
    for name in ['dp_structure.py','dp_structure_agent.py','dp_correction.py','DP_STRUCTURE_PROMPT.md']:
        if prior['source_sha256'].get(name)!=manifest['source_sha256'].get(name):raise ValueError('Import implementation mismatch: '+name)
    for entry in manifest['entries']:
        if not all(completed_method(source,entry,m) for m in METHODS):raise ValueError('Incomplete import scene: '+entry['scene'])
        scene=source/'scenes'/entry['scene']
        image,raw=load_scene(entry,out)
        if state_key(raw)!=state_key(read_items(scene/'raw/final.pixel.geojson')):raise ValueError('Imported raw differs from frozen prediction')
        from dp_structure import initialize,StructureConfig
        expected=initialize(raw,StructureConfig(**manifest['config']))
        if state_key(expected)!=state_key(read_items(scene/'dp/final.pixel.geojson')):raise ValueError('Imported DP differs from current initialization')
        destination=out/'scenes'/entry['scene'];destination.mkdir(parents=True,exist_ok=True)
        provenance=dict(source=str(source),source_manifest_sha256=digest(source/'manifest.json'),
            imported_methods=METHODS,new_API_requests=0,created_utc=now(),
            note='Historical responses and vectors explicitly reused after matching inputs, implementation, config and planner; this is not a new GPT replication')
        # Mark reuse before copying paid call records, so a partial import cannot be mistaken for fresh calls.
        save(destination/'reuse_provenance.json',provenance)
        for m in METHODS:
            if completed_method(out,entry,m):
                if digest(destination/m/'final.pixel.geojson')!=digest(scene/m/'final.pixel.geojson'):raise ValueError('Destination conflict: '+m)
            else:
                if (destination/m).exists():raise ValueError('Partial destination method; inspect before import')
                shutil.copytree(scene/m,destination/m)
        for name in ['diagnosis.json','candidate_pool.json','preparation.json']:
            shutil.copyfile(scene/name,destination/name)
        print('IMPORTED',entry['scene'],flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Final raw / DP / DP+visual-agent comparison')
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--planner',type=Path,default=Path(__file__).resolve().parent.parent/'data/planner.example.json')
    parser.add_argument('--stage',choices=['local','gpt','sweep','report','import'],default='local')
    parser.add_argument('--dp-tolerance',type=float,default=3.)
    parser.add_argument('--dp-sweep',type=float,nargs='+',default=[1.,2.,3.,4.,6.,8.])
    parser.add_argument('--reuse-from',type=Path,help='Historical verified DP-structure run; requires stage=import')
    args=parser.parse_args(argv);activate_engine()
    import cv2
    cv2.setNumThreads(1)
    from dp_structure import StructureConfig
    from fieldagent.experiment import stage_lock
    config=StructureConfig(dp_tolerance=args.dp_tolerance);out=args.output.resolve()
    if args.stage=='report':
        manifest=load(out/'manifest.json')
        if manifest.get('version')!=VERSION or manifest.get('methods')!=METHODS:raise ValueError('Not a final three-arm output')
    else:
        if args.inputs is None:parser.error('--inputs is required to generate or import results')
        if any(not __import__('math').isfinite(t) or t<=0 for t in args.dp_sweep):parser.error('Sweep tolerances must be finite and positive')
        tolerances=sorted(set(args.dp_sweep))
        manifest=freeze(args.inputs,out,config,planner_config(args.planner),tolerances)
        try:
            with stage_lock(out,'execution'):
                if args.stage=='local':run_local(manifest,out,config)
                elif args.stage=='gpt':run_gpt(manifest,out,config)
                elif args.stage=='sweep':run_sweep(manifest,out,config)
                else:
                    if args.reuse_from is None:parser.error('Import requires --reuse-from')
                    if args.reuse_from.resolve()==out:parser.error('Source and destination must differ')
                    import_results(manifest,out,args.reuse_from.resolve())
        except Exception:
            # Retain completed work and a progress report without concealing the original error.
            import traceback
            save(out/'last_failure.json',dict(stage=args.stage,traceback=traceback.format_exc(),created_utc=now()))
            try:report(manifest,out)
            except Exception:pass
            raise
    result=report(manifest,out)
    if (out/'DP_SWEEP.json').exists() or args.stage=='sweep':sweep_report(manifest,out)
    print(json.dumps(dict(complete_by_method=result['complete_by_method'],output=str(out)),ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
