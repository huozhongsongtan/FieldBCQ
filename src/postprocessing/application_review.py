"""Offline reports and an explicitly unannotated human-review kit."""
from pathlib import Path
from collections import Counter
import csv
import json
import html
import cv2
import numpy as np
from fieldagent.io import load,save,digest


def read_annotations(path,case_keys,manifest_sha,expected_vectors=None):
    if path is None: return []
    doc=load(path)
    if doc.get('manifest_sha256')!=manifest_sha: raise ValueError('Annotations belong to another experiment')
    result=doc.get('annotations',[]);seen=set()
    for row in result:
        key=(row.get('scene'),row.get('segment_id'),row.get('method'))
        if key in seen or key not in case_keys: raise ValueError('Unknown or duplicate annotation case: '+str(key))
        seen.add(key)
        if expected_vectors is not None and row.get('vector_sha256')!=expected_vectors.get((key[0],key[2])):
            raise ValueError('Annotation vector fingerprint mismatch')
        if row.get('problem') not in ('unknown','none','remove_redundancy','restore_corner','realign_edge','uncertain'):
            raise ValueError('Unknown manual problem label')
        if row.get('judgement') not in ('unjudged','improved','harmful','neutral','uncertain'):
            raise ValueError('Unknown manual edit judgement')
        if row.get('real_corner_deleted') not in ('unknown','yes','no'): raise ValueError('Invalid corner annotation')
        if row.get('judgement') not in ('unjudged','uncertain') and not str(row.get('reviewer','')).strip():
            raise ValueError('A judged result needs a reviewer')
        seconds=row.get('review_seconds')
        if seconds is not None and (type(seconds) not in (int,float) or not np.isfinite(seconds) or seconds<0):
            raise ValueError('Invalid manually recorded review seconds')
    return result


def report(manifest,out,annotations=None):
    from application_experiment import MAIN_METHODS,ABLATION,STRONG_DP
    from fieldagent.experiment import aggregate,csv_file,completed_method
    from application_boundary import TASKS
    has_strong='strong_dp_tolerance_px' in manifest
    methods=MAIN_METHODS+[ABLATION]+([STRONG_DP] if has_strong else [])
    records={};operations=[];cases=set();call_rows=[];coverage=[]
    for e in manifest['entries']:
        scene=out/'scenes'/e['scene']
        diag=load(scene/'diagnosis.json') if (scene/'diagnosis.json').exists() else {'selected':[]}
        raw_record=load(scene/'raw/completed.json') if completed_method(out,e,'raw') else {}
        coverage.append(dict(scene=e['scene'],domain=e['domain'],input_instances=raw_record.get('prediction_instances',0),
            detected_segments=diag.get('detected',0),selected_segments=len(diag['selected']),
            selected_instances=len({s['instance_id'] for s in diag['selected']}),
            selected_removable_DP_points=sum(s.get('removable_dp_points',0) for s in diag['selected']),
            scene_segment_budget=manifest['config']['max_segments'],
            instance_segment_budget=manifest['config']['max_segments_per_instance']))
        for s in diag['selected']:
            for m in methods[1:]: cases.add((e['scene'],s['segment_id'],m))
        for m in methods:
            if not completed_method(out,e,m): continue
            records[e['scene'],m]=load(scene/m/'completed.json')
            if (scene/m/'trace.json').exists():
                trace=load(scene/m/'trace.json')
                for event in trace.get('events',[]):
                    operations.append(dict(scene=e['scene'],method=m,segment_id=event['segment_id'],
                        instance_id=event['instance_id'],task=event['task'],status=event['status'],
                        tool=event.get('candidate',{}).get('method'),
                        net_vertices=event.get('operation',{}).get('net_vertex_change',0),
                        deleted=len(event.get('operation',{}).get('deleted_DP_coordinates',[])),
                        added=len(event.get('operation',{}).get('added_control_coordinates',[]))))
        for path in scene.glob('*/run/*/started.json'):
            folder=path.parent;meta=load(folder/'metadata.json') if (folder/'metadata.json').exists() else load(path)
            call_rows.append(dict(scene=e['scene'],method=folder.parents[1].name,phase=folder.name,
                completed=(folder/'response.json').exists(),seconds=meta.get('seconds'),
                input_tokens=meta.get('usage',{}).get('input_tokens'),output_tokens=meta.get('usage',{}).get('output_tokens'),
                images_sent=meta.get('images_sent'),response_model=meta.get('response_model')))
    groups={}
    group_specs=[('local',MAIN_METHODS[:2]),('main',MAIN_METHODS),('ablation',MAIN_METHODS[1:]+[ABLATION])]
    if has_strong:
        group_specs.extend([('local_with_strong',['raw','dp',STRONG_DP]),
            ('strong_reference',['dp',STRONG_DP,'dp_visual_agent'])])
    for name,arms in group_specs:
        chosen=[e for e in manifest['entries'] if all((e['scene'],m) in records for m in arms)]
        groups[name]=dict(images=len(chosen),scenes=[e['scene'] for e in chosen],
            aggregate={m:aggregate([records[e['scene'],m] for e in chosen]) for m in arms} if chosen else {})
    manual=read_annotations(annotations,cases,digest(out/'manifest.json'),
        {key:value['final_sha256'] for key,value in records.items()})
    manual_by_key={(r['scene'],r['segment_id'],r['method']):r for r in manual}
    task_rows=[]
    for method in MAIN_METHODS[2:]+[ABLATION]:
        for task in TASKS:
            rows=[r for r in operations if r['method']==method and r['task']==task]
            committed=[r for r in rows if r['status']=='committed']
            judged=[manual_by_key.get((r['scene'],r['segment_id'],method)) for r in committed]
            judged=[r for r in judged if r and r['judgement'] in ('improved','harmful','neutral')]
            outcomes=Counter(r['judgement'] for r in judged)
            task_rows.append(dict(method=method,task=task,observed_decisions=len(rows),committed=len(committed),
                human_judged_commits=len(judged),human_unjudged_commits=len(committed)-len(judged),
                human_improved=outcomes['improved'],human_harmful=outcomes['harmful'],human_neutral=outcomes['neutral'],
                human_improvement_rate=outcomes['improved']/len(judged) if judged else None,
                human_confirmed_corner_deletions=sum(r['real_corner_deleted']=='yes' for r in judged)))
    rows=[];paired=[]
    for group,data in groups.items():
        rows.extend(dict(comparison=group,method=m,**stats) for m,stats in data['aggregate'].items())
    for e in manifest['entries']:
        for m in MAIN_METHODS[2:]+[ABLATION]:
            for reference in ['dp']+([STRONG_DP] if has_strong else []):
                if (e['scene'],reference) not in records or (e['scene'],m) not in records: continue
                a=records[e['scene'],reference];b=records[e['scene'],m]
                am=a.get('evaluation',{}).get('metrics',{});bm=b.get('evaluation',{}).get('metrics',{})
                paired.append(dict(scene=e['scene'],domain=e['domain'],method=m,reference=reference,
                    delta_vertices=b['vertices']-a['vertices'],
                    delta_BF1=None if am.get('boundary_f1') is None else bm['boundary_f1']-am['boundary_f1']))
    from evaluate_final import bootstrap
    intervals={m:bootstrap([r['delta_BF1'] for r in paired if r['method']==m and r['reference']=='dp' and r['delta_BF1'] is not None])
        for m in MAIN_METHODS[2:]+[ABLATION]}
    strong_intervals={m:bootstrap([r['delta_BF1'] for r in paired if r['method']==m and r['reference']==STRONG_DP and r['delta_BF1'] is not None])
        for m in MAIN_METHODS[2:]+[ABLATION]} if has_strong else {}
    sweep=[]
    if (out/'DP_SWEEP_PROTOCOL.json').exists():
        for t in load(out/'DP_SWEEP_PROTOCOL.json')['tolerances']:
            root=out/'dp_sweep'/('eps_'+str(t).replace('.','p'))
            chosen=[e for e in manifest['entries'] if completed_method(root,e,'dp')]
            if chosen: sweep.append(dict(tolerance_px=t,**aggregate([load(root/'scenes'/e['scene']/'dp/completed.json') for e in chosen])))
    result=dict(version=manifest['version'],development_only=True,input_split=manifest['input_split'],
        expected_images=len(manifest['entries']),complete_by_method={m:sum(k[1]==m for k in records) for m in methods},
        groups=groups,task_results=task_rows,BF1_image_bootstrap=intervals,
        BF1_vs_strong_DP_image_bootstrap=strong_intervals,
        dp_tolerance_px=manifest['config']['dp_tolerance'],strong_dp_tolerance_px=manifest.get('strong_dp_tolerance_px'),
        routing_coverage=dict(images=len(coverage),input_instances=sum(r['input_instances'] for r in coverage),
            selected_instances=sum(r['selected_instances'] for r in coverage),
            detected_segments=sum(r['detected_segments'] for r in coverage),
            selected_segments=sum(r['selected_segments'] for r in coverage),
            note='Selection is a processing budget, not defect recall; unselected arcs remain light DP'),
        manual_annotation_rows=len(manual),
        API=dict(attempts=len(call_rows),completed=sum(r['completed'] for r in call_rows),
            input_tokens_reported=sum(r['input_tokens'] or 0 for r in call_rows),
            output_tokens_reported=sum(r['output_tokens'] or 0 for r in call_rows),
            request_seconds=sum(r['seconds'] or 0 for r in call_rows)),
        unresolved_events=sum(r['status'] in ('needs_human_review','rejected_invalid_preview','rejected_at_commit','abandoned') for r in operations),
        ablation_contract=manifest['ablation_contract'],annotation_source_sha256=digest(annotations) if annotations else None,
        annotation_note='Human success unknown until adjudicated. Geometric acceptance is not correctness. Review time is not editing time.')
    save(out/'summary.json',result);csv_file(out/'METRICS.csv',rows);csv_file(out/'OPERATIONS.csv',operations)
    csv_file(out/'TASK_RESULTS.csv',task_rows);csv_file(out/'PAIRED_DELTAS.csv',paired);csv_file(out/'API_CALLS.csv',call_rows)
    csv_file(out/'COVERAGE.csv',coverage)
    if sweep: csv_file(out/'DP_SWEEP.csv',sweep)
    lines=['# 应用实验进度与结果','',f'数据划分：{manifest["input_split"]}；开发实验；预计 {len(manifest["entries"])} 张。','',
        f'主实验：原始预测、DP {manifest["config"]["dp_tolerance"]:g}px、DP＋视觉智能体。无图片输入组仅作消融；共享本地图像工具，不等于全流程不使用影像。',
        ('补充强对照：独立DP '+str(manifest['strong_dp_tolerance_px'])+'px；不进入智能体上下文或自动回填。' if has_strong else '本目录为历史配置。'),'',
        '|比较|方法|共同完成图数|总点数|平均点数|BF1@2px|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        bf=r.get('boundary_f1_macro');bf='—' if bf is None else f'{bf:.5f}'
        lines.append(f'|{r["comparison"]}|{r["method"]}|{r["images"]}|{r["vertices_after"]}|{r["mean_vertices"]:.2f}|{bf}|')
    lines+=['','人工核验记录：'+str(len(manual))+' 条。没有人工判定的修复成功率保留为空。',
        '路由覆盖：'+str(result['routing_coverage']['selected_instances'])+'/'+str(result['routing_coverage']['input_instances'])+' 个预测实例包含入选边段，入选 '+str(result['routing_coverage']['selected_segments'])+' 段；不等于全部实例已修改或缺陷召回率。',
        '若智能体尚未运行，主实验共同完成数为0；本地结果不能冒充真实GPT效果。',
        '所有已提交、保持、放弃和拒绝事件均在OPERATIONS.csv中；工具参数、坐标和预览在各场景run目录中。',
        'GT仅用于保存矢量后的离线评价和独立人工核验材料，不能用于运行时选结果。',
        '人工复核界面记录的是复核时间；人工编辑节省需要另行实际修图计时。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def review_kit(manifest,out):
    from cvr_agent import read_items
    from run_boundary_tools import load_image
    from fieldbqc_direct_edit import ring_records
    from evaluate_mechanism_metrics import read_yolo_masks
    from application_experiment import MAIN_METHODS,ABLATION,STRONG_DP
    folder=out/'human_review';folder.mkdir(exist_ok=True);cases=[]
    for e in manifest['entries']:
        scene=out/'scenes'/e['scene']
        if not (scene/'diagnosis.json').exists(): continue
        image=load_image(e['image']);h,w=image.shape[:2];gt=None
        if e.get('label'):
            if digest(e['label'])!=e['label_sha256']: raise ValueError('Offline label changed')
            gt=read_yolo_masks(Path(e['label']),h,w)
        methods=[m for m in MAIN_METHODS[1:]+([STRONG_DP] if 'strong_dp_tolerance_px' in manifest else [])+[ABLATION] if (scene/m/'completed.json').exists()]
        items={m:read_items(scene/m/'final.pixel.geojson') for m in methods}
        events={m:{v['segment_id']:v for v in load(scene/m/'trace.json').get('events',[])} for m in methods}
        for s in load(scene/'diagnosis.json')['selected']:
            coords=np.array(s['raw_arc']+s['endpoints'],float)
            x0=max(0,int(coords[:,0].min())-12);y0=max(0,int(coords[:,1].min())-12)
            x1=min(w,int(np.ceil(coords[:,0].max()))+13);y1=min(h,int(np.ceil(coords[:,1].max()))+13)
            crop=image[y0:y1,x0:x1];frames=[('Original',crop.copy()),('GT reference (offline)',crop.copy())]
            if gt is not None:
                for mask in gt.masks:
                    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frames[1][1],[(c-np.array([[[x0,y0]]])).astype(np.int32) for c in contours],-1,(255,255,0),1)
            for m in methods:
                frame=crop.copy()
                for p in items[m]:
                    if p.identifier!=s['instance_id']: continue
                    for ring in ring_records(p.geometry):
                        points=np.round(np.array(ring['coordinates'])-[x0,y0]).astype(np.int32)
                        cv2.polylines(frame,[points],True,(0,255,255),1,cv2.LINE_AA)
                        for point in points: cv2.circle(frame,tuple(point),1,(0,255,255),-1)
                frames.append((m,frame))
            panels=[]
            for name,frame in frames:
                side=224;scale=min(side/frame.shape[0],side/frame.shape[1]);board=np.full((side+30,side,3),245,np.uint8)
                resized=cv2.resize(frame,(max(1,round(frame.shape[1]*scale)),max(1,round(frame.shape[0]*scale))))
                board[30:30+resized.shape[0],:resized.shape[1]]=resized
                cv2.putText(board,name,(3,19),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),1);panels.append(board)
            path=folder/'images'/e['scene']/(s['segment_id']+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(cv2.imencode('.png',np.hstack(panels))[1].tobytes())
            cases.append(dict(scene=e['scene'],segment_id=s['segment_id'],instance_id=s['instance_id'],
                suspected_task=s['suspected_task'],confirmed_problem='unknown',bounds=[x0,y0,x1,y1],
                image=path.relative_to(folder).as_posix(),methods=methods,
                vector_sha256={m:load(scene/m/'completed.json')['final_sha256'] for m in methods},
                statuses={m:events[m].get(s['segment_id'],{}).get('status','baseline' if m=='dp' else 'unknown') for m in methods},
                annotation_scope='Selected risk windows only; not exhaustive whole-scene defect labels'))
    save(folder/'cases.json',dict(manifest_sha256=digest(out/'manifest.json'),cases=cases))
    data=json.dumps(dict(manifest_sha256=digest(out/'manifest.json'),cases=cases),ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>地块边界人工核验</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f6f8}article{background:white;padding:18px;margin:16px 0;border-radius:8px}img{max-width:100%;display:block}select,input{margin:7px;padding:5px}small{color:#555}button{padding:10px}</style>
<h1>地块边界人工核验</h1><p>疑似类型尚未确认。参考原图和GT，分别判断修改是否有益；GT标注也可能存在误差。此处仅覆盖被选中的风险窗口。</p>
<p>先运行gpt/ablation，再重新生成本页，可核验实际修改。页面只导出人工记录，不修改矢量；复核时间不代表修图时间。</p>
<label>核验人<input id="reviewer"></label><button id="download">下载人工核验JSON</button><div id="cases"></div><script>
const DATA=__DATA__;const rows=[];const labels={unknown:'未标注',none:'没有问题',remove_redundancy:'冗余点',restore_corner:'拐角丢失',realign_edge:'边界偏移',uncertain:'无法确认',unjudged:'未判定',improved:'修改有益',harmful:'修改有害',neutral:'无明显变化',yes:'是',no:'否'};
function select(options,cb){const e=document.createElement('select');for(const key of options){let o=document.createElement('option');o.value=key;o.textContent=labels[key]||key;e.append(o)}e.onchange=()=>cb(e.value);return e}
for(const c of DATA.cases){const a=document.createElement('article');const title=document.createElement('p');title.textContent=`${c.scene} / ${c.segment_id} / 实例 ${c.instance_id}；疑似 ${c.suspected_task}`;a.append(title);let img=new Image();img.src=c.image;img.loading='lazy';a.append(img);
for(const m of c.methods){let row={scene:c.scene,segment_id:c.segment_id,method:m,vector_sha256:c.vector_sha256[m],problem:'unknown',judgement:'unjudged',real_corner_deleted:'unknown',review_seconds:null,reviewer:'',notes:''};rows.push(row);let div=document.createElement('div');let label=document.createElement('span');label.textContent=m+' ['+c.statuses[m]+']';div.append(label);div.append(select(['unknown','none','remove_redundancy','restore_corner','realign_edge','uncertain'],v=>row.problem=v));div.append(select(['unjudged','improved','harmful','neutral','uncertain'],v=>row.judgement=v));let corner=document.createElement('span');corner.textContent='误删真实拐角：';div.append(corner,select(['unknown','yes','no'],v=>row.real_corner_deleted=v));let seconds=document.createElement('input');seconds.type='number';seconds.min=0;seconds.placeholder='实际复核秒数';seconds.oninput=()=>row.review_seconds=seconds.value===''?null:Number(seconds.value);div.append(seconds);let note=document.createElement('input');note.placeholder='备注';note.oninput=()=>row.notes=note.value;div.append(note);a.append(div)}document.querySelector('#cases').append(a)}
document.querySelector('#download').onclick=()=>{const reviewer=document.querySelector('#reviewer').value.trim();const annotations=rows.filter(r=>r.problem!=='unknown'||r.judgement!=='unjudged'||r.real_corner_deleted!=='unknown').map(r=>({...r,reviewer}));const blob=new Blob([JSON.stringify({manifest_sha256:DATA.manifest_sha256,annotations},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human_annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script>'''.replace('__DATA__',data)
    (folder/'index.html').write_text(page,encoding='utf-8')
    print('Human review windows:',len(cases),str(folder/'index.html'),flush=True)
