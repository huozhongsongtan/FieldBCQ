"""Offline GT evaluation and independent integrity audit; never plans edits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import csv
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw,ImageFont
from cvr_agent import HOST, dump, sha, read_items, geometry_mask, scene_metrics, polygons
from fieldbqc_geometry import to_parcels
from evaluate_agent_experiments import measure, instances
from evaluate_mechanism_metrics import read_yolo_masks,instance_boundary_union
from evaluate_goal_repair import frozen_ownership,ownership_score
from run_agent_experiments import load_parcels

def audit_calls(path):
    # Stream large audit logs rather than loading every mask/check event.
    started={};finished=set();errors=[];records=0
    for line in path.read_text(encoding='utf-8').splitlines():
        r=json.loads(line);records+=1;uid=r['call_id']
        if r['status']=='started':
            if uid in started:errors.append('duplicate start '+uid)
            parent=r.get('parent_call_id')
            if parent and parent not in started:errors.append('missing parent '+parent)
            started[uid]=r
        elif r['status'] in ('completed','failed'):
            if uid not in started:errors.append('terminal without start '+uid)
            if uid in finished:errors.append('duplicate terminal '+uid)
            finished.add(uid)
    errors+=['incomplete '+i for i in set(started)-finished]
    return dict(calls=len(started),records=records,errors=errors)

def get_metrics(items,gt,shape,baseline,frozen):
    pred=instances(to_parcels(items,shape));geom=scene_metrics(items)
    results={}
    for threshold in (.5,.75,.9):
        d,_=measure(gt,pred,shape,iou_threshold=threshold)
        results['instance_f1_'+str(threshold)]=d['instance_f1'] or 0.
        if threshold==.5:
            results.update(boundary_f1=d['boundary_f1'],union_iou=d['union_iou'],pq=d['pq'],
                 matched_iou=(d['matched_iou_sum']/d['tp'] if d['tp'] else None),matched_instances=d['tp'],unmatched_predictions=d['fp'],unmatched_gt=d['fn'])
    results['instance_f1_mean']=float(np.mean([results['instance_f1_'+str(t)] for t in (.5,.75,.9)]))
    pb=instance_boundary_union(pred.masks);gb=instance_boundary_union(gt.masks)
    if pb.any() and gb.any():
        pg=distance_transform_edt(~gb)[pb];gp=distance_transform_edt(~pb)[gb]
        results['pixel_boundary_chamfer']=float((pg.mean()+gp.mean())/2)
        results['pixel_boundary_p95']=float(np.percentile(np.r_[pg,gp],95))
    else:results.update(pixel_boundary_chamfer=None,pixel_boundary_p95=None)
    results.update(geom,**ownership_score(items,frozen,shape))
    u0=shapely_union(baseline);u1=shapely_union(items)
    results['coverage_loss_px2']=u0.difference(u1).area
    results['coverage_added_px2']=u1.difference(u0).area
    old={p.identifier:p.geometry for p in baseline};new={p.identifier:p.geometry for p in items}
    shared=0;lost=0;maxshift=0.
    ids=list(old)
    for k,a in enumerate(ids):
        for b in ids[k+1:]:
            if old[a].intersection(old[b]).area>1e-7:continue
            edge=old[a].boundary.intersection(old[b].boundary)
            if edge.length<=1e-7:continue
            shared+=1;current=new[a].boundary.intersection(new[b].boundary)
            if current.length<=1e-7:lost+=1
            else:maxshift=max(maxshift,edge.hausdorff_distance(current))
    results.update(initial_shared_pairs=shared,lost_shared_pairs=lost,max_shared_boundary_shift_px=maxshift,
                   id_preservation=float(set(old)==set(new)))
    return results

def shapely_union(items):
    from shapely.ops import unary_union
    return unary_union([p.geometry for p in items])

def preview(image,items,title,side=300):
    h,w=image.height,image.width;s=min(side/w,side/h)
    ox,oy=(side-round(w*s))//2,(side-round(h*s))//2
    frame=Image.new('RGB',(side,side+45),'white')
    frame.paste(image.resize((round(w*s),round(h*s))),(ox,oy+45));draw=ImageDraw.Draw(frame)
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',15)
    draw.text((5,5),title,font=font,fill='black')
    m=scene_metrics(items);draw.text((5,25),f'N={m["vertices"]}  E={m["overlap_excess_area"]:.1f}',font=font,fill='black')
    for k,p in enumerate(items):
        color=[(20,255,80),(40,190,255),(255,200,30),(255,80,200)][k%4]
        for poly in polygons(p.geometry):
            for r in [poly.exterior,*poly.interiors]:
                draw.line([(ox+x*s,oy+45+y*s) for x,y in r.coords],fill=color,width=1)
    return frame

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args();root=args.data;out=args.runs
    if not (out/'completed.json').exists():raise ValueError('Incomplete parent; no efficacy report')
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'));sig=manifest['signature']
    for path,digest in sig['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed input '+path)
    for path,digest in sig['source_sha256'].items():
        if sha(out/'source_snapshot'/Path(path).name)!=digest:raise ValueError('Changed snapshot')
    pairs=json.loads((root/'paired_inputs.json').read_text(encoding='utf-8'))['pairs']
    methods=sig['methods'];rows=[];integrity=[];prefix=[]
    evalroot=out/'evaluation';evalroot.mkdir(exist_ok=False)
    dump(evalroot/'protocol.json',dict(instance_iou_thresholds=[.5,.75,.9],boundary_tolerance_px=2,
       ownership='Frozen best-IoU proxy, min=.30, margin=.10; NOT independent adjudication of conflicts',
       confidence_intervals='Descriptive paired bootstrap by development scene; not independent-test inference',
       gt_sha256={r['ground_truth_original_label']:sha(root/r['ground_truth_original_label']) for r in pairs}))
    for pair in pairs:
        stem=pair['image_id'];image=Image.open(root/pair['image']).convert('RGB');shape=(image.height,image.width)
        baseline=read_items(root/pair['prediction_vector'])
        label=root/pair['ground_truth_original_label']
        if sha(label)!=pair['ground_truth_label_sha256']:raise ValueError('GT changed')
        gt=read_yolo_masks(label,*shape);frozen=frozen_ownership(baseline,gt,shape)
        rawmetrics=get_metrics(baseline,gt,shape,baseline,frozen)
        rows.append(dict(scene=stem,method='raw',**rawmetrics,candidates=0,commits=0,seconds=0.,review_pairs=0,model_calls=0))
        visual_items=[('原始预测',baseline)]
        for method in methods:
            folder=out/'runs'/method/stem
            if not (folder/'completed.json').exists():raise ValueError('Incomplete method')
            trace=json.loads((folder/'trace.json').read_text(encoding='utf-8'))
            current=read_items(folder/'final/prediction.repaired.pixel.geojson')
            parcels,rastershape=load_parcels(folder/'final/prediction.repaired_masks.npz')
            original_ids=[p.identifier for p in baseline]
            assert original_ids==[p.identifier for p in current]
            assert [p.parcel_id for p in parcels]==original_ids
            assert rastershape==shape
            assert all(p.geometry.is_valid and not p.geometry.is_empty for p in current)
            assert all(np.array_equal(p.mask,geometry_mask(g.geometry,shape)) for p,g in zip(parcels,current))
            measured=scene_metrics(current)
            assert measured==trace['final'],(method,stem,measured,trace['final'])
            assert trace['candidates']<=sig['config']['budget']
            if (folder/'frozen_policy.json').exists():
                policy=json.loads((folder/'frozen_policy.json').read_text())
                assert stem not in policy['training_scenes']
            checks=0
            for h in trace['history']:
                for item in [h['candidate_file'],*h['attempted_files']]:
                    assert sha(item['path'])==item['sha256'];checks+=1
            if method!='legacy_rule':
                audit=audit_calls(folder/'calls.jsonl');assert not audit['errors'],audit
            else:audit=dict(calls=None,errors=[],note='Legacy audit uses different start/end event IDs; preserved separately, not claimed to meet CVR schema')
            integrity.append(dict(scene=stem,method=method,identity=True,legal=True,raster_roundtrip=True,
                    heldout_excluded=True,candidate_artifacts_verified=checks,audit=audit))
            d=get_metrics(current,gt,shape,baseline,frozen)
            rows.append(dict(scene=stem,method=method,**d,candidates=trace['candidates'],commits=trace['commits'],
                    seconds=trace['actual_wall_seconds'],review_pairs=len(trace['reviews']),model_calls=trace['model_calls']))
            if method in ('fixed','greedy','adaptive'):visual_items.append((method,current))
            if method in ('fixed','greedy','adaptive','no_feedback'):
                for budget in (8,16,32,sig['config']['budget']):
                    n=min(budget,len(trace['history']));state=trace['history'][n-1]['after'] if n else trace['baseline']
                    prefix.append(dict(scene=stem,method=method,budget=budget,**state,
                          note='Prefix of a fixed-max-budget trajectory, not a separately reoptimized budget policy'))
        panel=Image.new('RGB',(300*len(visual_items),345),'white')
        for k,(name,items) in enumerate(visual_items):panel.paste(preview(image,items,name),(k*300,0))
        panel.save(evalroot/(stem+'.comparison.png'))
        print('EVALUATED '+stem,flush=True)
    with (evalroot/'per_scene.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregates=[]
    for method in ['raw',*methods]:
        group=[r for r in rows if r['method']==method]
        aggregates.append(dict(method=method,scenes=len(group),vertices=sum(r['vertices'] for r in group),
          overlap=sum(r['overlap_excess_area'] for r in group),invalid=sum(r['invalid_geometries'] for r in group),
          mean_instance_f1=float(np.mean([r['instance_f1_mean'] for r in group])),
          mean_boundary_f1=float(np.mean([r['boundary_f1'] for r in group])),
          mean_chamfer=float(np.mean([r['pixel_boundary_chamfer'] for r in group])),
          candidates=sum(r['candidates'] for r in group),commits=sum(r['commits'] for r in group),
          seconds=sum(r['seconds'] for r in group),review_pairs=sum(r['review_pairs'] for r in group),
          ownership_proxy_judgeable=sum(r['ownership_proxy_judgeable'] for r in group),
          ownership_proxy_wrong_unique=sum(r['ownership_proxy_wrong_unique'] for r in group),
          ownership_proxy_uniquely_assigned=sum(r['ownership_proxy_uniquely_assigned'] for r in group)))
    rng=np.random.default_rng(20261021);comparisons=[]
    for control in [m for m in methods if m!='adaptive']:
        if 'adaptive' not in methods:break
        for key in ['vertices','overlap_excess_area','instance_f1_mean','boundary_f1']:
            deltas=np.array([next(r[key] for r in rows if r['scene']==s and r['method']=='adaptive')-
                              next(r[key] for r in rows if r['scene']==s and r['method']==control) for s in sorted({r['scene'] for r in rows})])
            boot=np.mean(rng.choice(deltas,(2000,len(deltas)),replace=True),axis=1)
            comparisons.append(dict(control=control,key=key,mean_delta=float(deltas.mean()),
                  ci95=np.quantile(boot,[.025,.975]).tolist(),development_only=True))
    dump(evalroot/'summary.json',dict(aggregate=aggregates,paired_comparisons=comparisons,
         conclusion_scope='Development only. No GT-trained semantic risk model or independent ownership truth; no proof of industrial generalization.',
         gt_used_only_here=True,api_model_calls=0))
    dump(evalroot/'integrity_audit.json',dict(passed=True,runs=len(integrity),records=integrity))
    dump(evalroot/'budget_prefix_curves.json',prefix)
    lines=['# CVR-Agent 六景开发实验结果','',
      '所有结果来自同一批冻结预测。每景训练数据排除当前景；这仍是开发交叉验证，不是独立测试。',
      '策略是本地 ridge 动作价值估计器，不是 GPT。训练只用其他景的验证器反馈、几何收益和耗时；未使用真实标注。',
      '', '|方法|总顶点|重叠面积 px²|实例 F1 均值|边界 F1 均值|候选次数|提交|秒|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:lines.append(f'|{r["method"]}|{r["vertices"]}|{r["overlap"]:.2f}|{r["mean_instance_f1"]:.4f}|{r["mean_boundary_f1"]:.4f}|{r["candidates"]}|{r["commits"]}|{r["seconds"]:.1f}|')
    lines+=['','实例 F1 是 IoU 阈值 0.5/0.75/0.9 的平均，再按场景等权汇总；边界容差为原图 2 px。',
      'fixed/greedy 与 adaptive 使用相同扩展工具集、前置条件和验证器。legacy_rule 是原 repair_scenes.py 的预算匹配复现，只使用原来的五类工具，不作为新增工具贡献已被控制的主对照。',
      'single_* 是预设参数域内、预算限制下的单工具策略，不宣称是穷举最优。',
      'no_graph/no_image/no_feedback 移除策略输入特征并重新拟合；执行器仍须保留图/影像验证。no_joint 禁用邻接联合 VW；no_review 禁用近包含复核拦截；no_cost 去掉预计成本分母。',
      'near-containment 复核是显式阈值规则，不是学得的语义判断。ownership 是冻结关联代理，不能用来声称已证明真实所有权正确。',
      '预算曲线为相同最大预算轨迹的前缀；错误自动提交率和风险覆盖曲线需要独立人工编辑真值，当前未伪造该指标。',
      '', '完整候选、验收与方法日志见各 runs/<方法>/<图像>/；源码和配置已冻结在 source_snapshot 和 manifest.json。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(aggregates,ensure_ascii=False))

if __name__=='__main__':main()
