"""Raw / DP / selective visual editing, with a separately reported ablation."""
import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import time
from fieldagent import activate_engine
from fieldagent.io import load,save,digest,now,read_inputs,planner_config

VERSION='parcel-application-light-dp-v2'
LEGACY_VERSION='parcel-application-three-tasks-v1'
MAIN_METHODS=['raw','dp','dp_visual_agent']
ABLATION='dp_agent_no_raster_input'
STRONG_DP='dp_strong'
DEFAULT_DP_TOLERANCE=1.
DEFAULT_STRONG_DP_TOLERANCE=3.


def freeze(inputs,out,config,gates,settings,strong_dp_tolerance=DEFAULT_STRONG_DP_TOLERANCE):
    import math
    if not math.isfinite(strong_dp_tolerance) or strong_dp_tolerance<=config.dp_tolerance:
        raise ValueError('Strong DP tolerance must be greater than the light DP tolerance')
    from fieldagent.experiment import source_hashes
    names=['application_boundary.py','application_agent.py','application_experiment.py','application_review.py',
        'application_data.py','APPLICATION_PROMPT.md','dp_structure.py','dp_correction.py','dp_structure_agent.py','compact_inputs.py']
    paths=[Path(__file__).with_name(n) for n in names]
    record=dict(version=VERSION,entries=read_inputs(inputs),inputs_sha256=digest(inputs),
        input_split=load(inputs).get('split','undeclared'),development_only=True,methods=MAIN_METHODS,
        supplementary_methods=[STRONG_DP,ABLATION],config=asdict(config),task_gates=asdict(gates),planner=settings,
        strong_dp_tolerance_px=strong_dp_tolerance,
        experiment_goal='Light DP removes pixel-scale redundancy; the visual agent further simplifies and corrects local boundaries',
        comparison_contract='Strong DP is an offline comparator, never an agent input, candidate, or automatic fallback',
        max_calls_per_scene_per_agent_arm=2,runtime_GT_access=False,
        source_sha256={p.name:digest(p) for p in paths},engine_sha256=source_hashes(),
        ablation_contract='Same fixed routing/candidates/tools/gates. Only planner raster input is removed; local executor still uses images.',
        gate_status='Uncalibrated development defaults, not accuracy standards')
    out=Path(out)
    if (out/'manifest.json').exists():
        old=load(out/'manifest.json')
        if {k:v for k,v in old.items() if k!='created_utc'}!=record: raise ValueError('Frozen inputs/code/settings changed; use new output')
        return old
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Use an empty experiment output')
    for p in paths:
        dest=out/'source_snapshot'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
    save(out/'manifest.json',dict(record,created_utc=now()))
    return load(out/'manifest.json')


def persist(out,entry,method,items,raw,image,trace):
    from fieldagent.experiment import persist_method
    from cvr_agent import save_items,read_items,state_key
    folder=out/'scenes'/entry['scene']/method;path=folder/'final.pixel.geojson'
    # Persist first, and reject partial conflicting geometry before offline GT read.
    if path.exists() and state_key(read_items(path))!=state_key(items): raise ValueError('Conflicting partial output: '+str(path))
    save_items(path,items);save(folder/'trace.json',trace)
    return persist_method(out,entry,method,items,raw,image,trace)


def local(manifest,out,config,gates):
    from dataclasses import replace
    from final_experiment import load_scene
    from fieldagent.experiment import completed_method
    from dp_structure import initialize
    from application_boundary import diagnose
    local_methods=MAIN_METHODS[:2]+([STRONG_DP] if 'strong_dp_tolerance_px' in manifest else [])
    for i,e in enumerate(manifest['entries'],1):
        scene=out/'scenes'/e['scene']
        if all(completed_method(out,e,m) for m in local_methods) and (scene/'diagnosis.json').exists() and (scene/'candidate_pool.json').exists(): continue
        image,raw=load_scene(e,out);began=time.perf_counter();dp=initialize(raw,config);dp_seconds=time.perf_counter()-began
        began=time.perf_counter();diagnosis,pools=diagnose(raw,dp,image,config,gates)
        preparation_seconds=time.perf_counter()-began
        save(scene/'diagnosis.json',diagnosis);save(scene/'candidate_pool.json',pools)
        save(scene/'preparation.json',dict(dp_seconds=dp_seconds,preparation_seconds=preparation_seconds,
            input_instances=len(raw),selected_instances=len({s['instance_id'] for s in diagnosis['selected']}),
            selected_segments=len(diagnosis['selected']),scene_segment_budget=config.max_segments,
            instance_segment_budget=config.max_segments_per_instance,
            coverage_note='Only selected risk arcs are sent to the agent; unselected arcs remain the light DP result'))
        controls=[('raw',raw,0.),('dp',dp,dp_seconds)]
        if STRONG_DP in local_methods:
            began=time.perf_counter()
            strong=initialize(raw,replace(config,dp_tolerance=manifest['strong_dp_tolerance_px']))
            controls.append((STRONG_DP,strong,time.perf_counter()-began))
        # Detection and candidates are frozen before any reference evaluation.
        for m,items,seconds in controls:
            if not completed_method(out,e,m): persist(out,e,m,items,raw,image,dict(seconds=seconds,model_calls=0,commits=0))
        print(f'APPLICATION LOCAL {i}/{len(manifest["entries"])} {e["scene"]} selected={diagnosis["selected_by_task"]}',flush=True)


def agent(manifest,out,config,gates,vision):
    from final_experiment import load_scene
    from cvr_agent import read_items
    from fieldagent.experiment import completed_method
    from application_agent import run
    method='dp_visual_agent' if vision else ABLATION
    for i,e in enumerate(manifest['entries'],1):
        if completed_method(out,e,method): continue
        if not all(completed_method(out,e,m) for m in MAIN_METHODS[:2]): raise ValueError('Run local first')
        scene=out/'scenes'/e['scene'];image,raw=load_scene(e,out);dp=read_items(scene/'dp/final.pixel.geojson')
        final,trace=run(image,raw,dp,load(scene/'diagnosis.json'),load(scene/'candidate_pool.json'),
            config,gates,manifest['planner'],scene/method/'run',vision=vision)
        persist(out,e,method,final,raw,image,trace)
        print(f'APPLICATION {method} {i}/{len(manifest["entries"])} {e["scene"]} calls={trace["model_calls"]} commits={trace["commits_by_task"]}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description='Application-oriented parcel boundary experiment')
    p.add_argument('stage',choices=['prepare','local','gpt','ablation','sweep','review','report'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path)
    p.add_argument('--planner',type=Path,default=Path(__file__).resolve().parents[1]/'data/planner.example.json')
    p.add_argument('--dp-tolerance',type=float,default=DEFAULT_DP_TOLERANCE,
        help='Light DP initialization tolerance in image pixels (default: 1)')
    p.add_argument('--strong-dp-tolerance',type=float,default=DEFAULT_STRONG_DP_TOLERANCE,
        help='Supplementary standalone DP comparator; never exposed to the agent (default: 3)')
    p.add_argument('--max-segments',type=int,default=12)
    p.add_argument('--gates',type=Path,help='Optional calibrated TaskGates JSON; frozen in manifest')
    p.add_argument('--dataset',type=Path,default=Path('D:/field/datasets'))
    p.add_argument('--weights',type=Path,default=Path('F:/project/FieldBCQ/model/abc/best.pt'))
    p.add_argument('--model-source',type=Path,default=Path('F:/project/FieldBCQ/src'))
    p.add_argument('--exclude-manifest',type=Path,default=Path(__file__).resolve().parents[1]/'data/inputs.json')
    p.add_argument('--count',type=int,default=50);p.add_argument('--seed',type=int,default=20261025)
    p.add_argument('--select-only',action='store_true');p.add_argument('--annotations',type=Path)
    p.add_argument('--dp-sweep',type=float,nargs='+',default=[1.,2.,3.,3.5,4.,6.])
    a=p.parse_args(argv);activate_engine()
    import cv2
    cv2.setNumThreads(1)
    out=a.output.resolve()
    if a.stage=='prepare':
        from application_data import select,predict
        record=select(a.dataset,out,a.count,a.seed,a.weights,a.model_source,a.exclude_manifest)
        if not a.select_only: predict(record,out)
        print('Validation selection:',record['selected_by_source'],str(out),flush=True);return 0
    from dp_structure import StructureConfig
    from application_boundary import TaskGates
    from fieldagent.experiment import stage_lock
    if a.stage=='local':
        if not a.inputs: p.error('--inputs required for local')
        config=StructureConfig(dp_tolerance=a.dp_tolerance,max_segments=a.max_segments)
        gates=TaskGates(**load(a.gates)) if a.gates else TaskGates()
        manifest=freeze(a.inputs,out,config,gates,planner_config(a.planner),a.strong_dp_tolerance)
    else:
        manifest=load(out/'manifest.json')
        if manifest['version'] not in (VERSION,LEGACY_VERSION): raise ValueError('Not an application experiment')
        config=StructureConfig(**manifest['config']);gates=TaskGates(**manifest['task_gates'])
        if a.stage in ('gpt','ablation'):
            from fieldagent.experiment import source_hashes
            for name,sha in manifest['source_sha256'].items():
                if digest(Path(__file__).with_name(name))!=sha: raise ValueError('Frozen source changed: '+name)
            if source_hashes()!=manifest['engine_sha256']: raise ValueError('Frozen engine changed')
            for e in manifest['entries']:
                from fieldagent.experiment import check_inputs
                check_inputs(e)
    with stage_lock(out,'application'):
        if a.stage=='local': local(manifest,out,config,gates)
        elif a.stage in ('gpt','ablation'): agent(manifest,out,config,gates,a.stage=='gpt')
        elif a.stage=='sweep':
            from final_experiment import run_sweep
            if any(not __import__('math').isfinite(t) or t<=0 for t in a.dp_sweep): p.error('Positive finite sweep tolerances required')
            tolerances=sorted(set(a.dp_sweep));path=out/'DP_SWEEP_PROTOCOL.json'
            if path.exists() and load(path)!=dict(tolerances=tolerances): raise ValueError('Sweep protocol changed')
            save(path,dict(tolerances=tolerances))
            run_sweep(dict(manifest,dp_sweep_tolerances_px=tolerances),out,config)
        from application_review import report,review_kit
        result=report(manifest,out,a.annotations)
        if a.stage=='review': review_kit(manifest,out)
    print(__import__('json').dumps(dict(complete_by_method=result['complete_by_method'],output=str(out)),ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
