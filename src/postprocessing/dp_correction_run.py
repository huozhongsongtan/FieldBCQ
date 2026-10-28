"""Independent four-arm DP / selective-correction development experiment."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time
from fieldagent import activate_engine
from fieldagent.io import load,save,digest,now,read_inputs,planner_config

METHODS=['raw','dp','dp_rule','dp_visual_agent']
VERSION='dp-selective-visual-correction-v1'


def freeze(inputs,out,config,settings):
    from fieldagent.experiment import source_hashes
    paths=[Path(__file__),Path(__file__).with_name('dp_correction.py'),
           Path(__file__).with_name('dp_correction_agent.py'),Path(__file__).with_name('DP_CORRECTION_PROMPT.md')]
    record=dict(version=VERSION,entries=read_inputs(inputs),inputs_sha256=digest(inputs),
        methods=METHODS,config=asdict(config),planner=settings,max_calls_per_scene=2,
        runtime_GT_access=False,development_only=True,
        source_sha256={p.name:digest(p) for p in paths},engine_sha256=source_hashes(),
        fairness='Both branches start at identical DP. Same suspect arcs, shared candidate pool, cumulative verifier and point budget. GPT additionally may author literal paths. Rule search and GPT search costs differ. Existing twenty images are development cases; no blind-test claim.')
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
    from dp_correction import initialize,ImageEvidence,detect,candidate_paths,run_rule
    for number,entry in enumerate(manifest['entries'],1):
        if all(completed_method(out,entry,m) for m in METHODS[:3]):continue
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
        began=time.monotonic();final,events=run_rule(raw,dp,diagnosis,pools,image.shape[:2],config)
        trace=dict(seconds=dp_seconds+preparation+time.monotonic()-began,model_calls=0,
            commits=sum(e.get('committed',False) for e in events),direct_committed=0,unfinished_target_ids=[],
            controller='image_geometry_heuristic_selection',events=events,runtime_GT_access=False)
        save(out/'scenes'/entry['scene']/'dp_rule/trace.json',trace)
        persist_method(out,entry,'dp_rule',final,raw,image,trace)
        print(f'LOCAL {number}/{len(manifest["entries"])} {entry["scene"]} suspects={diagnosis["detected"]} selected={diagnosis["selected_count"]} commits={trace["commits"]}',flush=True)


def run_gpt(manifest,out,config):
    from fieldagent.experiment import persist_method,completed_method
    from cvr_agent import read_items
    from dp_correction_agent import run
    for number,entry in enumerate(manifest['entries'],1):
        if completed_method(out,entry,'dp_visual_agent'):continue
        if not completed_method(out,entry,'dp_rule'):raise ValueError('Run local stage first')
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
    for group,methods in [('local',METHODS[:3]),('four_arm',METHODS)]:
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
    result['API']=dict(attempts=len(calls),completed_responses=sum(r['completed_response'] for r in calls),
        input_tokens_reported=sum(r['input_tokens'] or 0 for r in calls),output_tokens_reported=sum(r['output_tokens'] or 0 for r in calls),cost=None)
    delta=[]
    for entry in manifest['entries']:
        scene=entry['scene']
        if not all((scene,m) in records for m in METHODS):continue
        for ref in ['dp','dp_rule']:
            a=records[scene,ref];b=records[scene,'dp_visual_agent']
            am=a.get('evaluation',{}).get('metrics',{});bm=b.get('evaluation',{}).get('metrics',{})
            bf=None if am.get('boundary_f1') is None or bm.get('boundary_f1') is None else bm['boundary_f1']-am['boundary_f1']
            delta.append(dict(scene=scene,reference=ref,delta_vertices=b['vertices']-a['vertices'],delta_BF1=bf,
                BF1_outcome=None if bf is None else ('improved' if bf>1e-8 else 'degraded' if bf< -1e-8 else 'tied')))
    csv_file(out/'METRICS.csv',table);csv_file(out/'PARCEL_COUNTS.csv',parcels)
    csv_file(out/'SCENE_COUNTS.csv',scene_rows);csv_file(out/'API_CALLS.csv',calls);csv_file(out/'PAIRED_DELTAS.csv',delta)
    save(out/'summary.json',result)
    lines=['# DP与选择性视觉纠错：开发实验','', '本批用于方法开发；四组共同完成的图像才进入配对表。BF1为GT 2px；GT只在最终矢量保存后读取。','']
    for group,data in result['groups'].items():
        lines += [f'## {group}：{data["images"]}张','', '|方法|总点数|平均点数|中位数|减点率|BF1|Chamfer/px|提交|调用|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        def fmt(x):return '—' if x is None else f'{x:.4f}'
        for m,s in data['aggregate'].items():
            lines.append(f'|{m}|{s["vertices_after"]}|{fmt(s["mean_vertices"])}|{fmt(s["median_vertices"])}|{fmt(s["reduction"])}|{fmt(s["boundary_f1_macro"])}|{fmt(s["chamfer_macro_px"])}|{s["commits"]}|{s["attempted_model_calls"]}|')
        lines.append('')
    lines += ['API用量：'+json.dumps(result['API'],ensure_ascii=False),'',
        'DP贡献与模型增量分开解释。允许少量必要增点；不能把几何验收通过当作真实纠错成功。',
        '选择性定位仅覆盖一部分疑似边段，无触发图像不调用GPT，保持DP；导出图数不等于实际调用图数。',
        '首轮只实现BF1@2px、全图Chamfer/P95、实例指标、点数和配对差值。局部GT纠错成功率、同点数曲线及额外容差扫描仍需独立补充，不能据本表宣称已证明。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--planner',type=Path,default=Path(__file__).resolve().parent.parent/'data/planner.example.json')
    parser.add_argument('--stage',choices=['local','gpt','report'],default='local')
    parser.add_argument('--dp-tolerance',type=float,default=3.)
    args=parser.parse_args(argv);activate_engine()
    import cv2
    cv2.setNumThreads(1)
    from dp_correction import CorrectionConfig
    from fieldagent.experiment import stage_lock
    config=CorrectionConfig(dp_tolerance=args.dp_tolerance);out=args.output.resolve()
    manifest=freeze(args.inputs,out,config,planner_config(args.planner))
    if args.stage!='report':
        with stage_lock(out,args.stage):
            if args.stage=='local':run_local(manifest,out,config)
            else:run_gpt(manifest,out,config)
    result=report(manifest,out)
    print(json.dumps(dict(complete_by_method=result['complete_by_method'],output=str(out)),ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
