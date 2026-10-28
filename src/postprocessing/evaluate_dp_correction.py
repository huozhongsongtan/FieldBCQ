"""Offline-only supplemental diagnostics after vector files are finalized.

Reads GT only here. Fixed DP-stage windows and exported vector hashes make
local boundary-set comparisons auditable; they are not semantic parcel matches.
"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from fieldagent import activate_engine
from fieldagent.io import load,save,digest
from dp_correction_run import METHODS,report


def boundary_metrics(pb,gb,tolerances=(1,2,3),window=None):
    from evaluate_mechanism_metrics import precision_recall_f1
    if window is None:window=np.ones(pb.shape,bool)
    pred=pb & window;gt=gb & window
    # Distances to full-scene boundary avoid artificial crop-edge boundaries.
    pg=distance_transform_edt(~gb)[pred] if gb.any() else np.full(np.count_nonzero(pred),np.inf)
    gp=distance_transform_edt(~pb)[gt] if pb.any() else np.full(np.count_nonzero(gt),np.inf)
    result={}
    for t in tolerances:
        # Match the existing BF1 implementation exactly: OpenCV's discrete
        # elliptical dilation differs slightly from an exact Euclidean disk.
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*t+1,2*t+1))
        near_gt=cv2.dilate(gb.astype(np.uint8),kernel).astype(bool)
        near_pred=cv2.dilate(pb.astype(np.uint8),kernel).astype(bool)
        result[f'BF1_{t}px']=precision_recall_f1(int((pred & near_gt).sum()),len(pg),int((gt & near_pred).sum()),len(gp))[2]
    result['chamfer_px']=float((pg.mean()+gp.mean())/2) if len(pg) and len(gp) and np.isfinite(pg).all() and np.isfinite(gp).all() else None
    result['p95_px']=float(np.percentile(np.r_[pg,gp],95)) if len(pg) and len(gp) and np.isfinite(pg).all() and np.isfinite(gp).all() else None
    return result


def bootstrap(values,seed=20261025):
    values=np.asarray(values,float)
    if not len(values):return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(values,(10000,len(values)),replace=True),axis=1)
    return dict(n=len(values),mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
        note='Exploratory image bootstrap; development reuse, spatial dependence and one API realization limit inference')


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);out=args.output.resolve();activate_engine()
    from fieldagent.experiment import csv_file,completed_method
    from cvr_agent import read_items,vertices
    from fieldbqc_geometry import to_parcels
    from evaluate_agent_experiments import instances
    from evaluate_mechanism_metrics import read_yolo_masks,instance_boundary_union
    from run_boundary_tools import load_image
    manifest=load(out/'manifest.json')
    if not all(completed_method(out,e,m) for e in manifest['entries'] for m in METHODS):
        raise ValueError('All four arms must be finalized before supplemental GT evaluation')
    result=report(manifest,out);global_rows=[];local_rows=[];hashes={};risks=[]
    figures=out/'figures';figures.mkdir(exist_ok=True)
    for entry in manifest['entries']:
        shape=load_image(entry['image']).shape[:2];scene=out/'scenes'/entry['scene']
        if digest(entry['label'])!=entry['label_sha256']:raise ValueError('GT fingerprint changed')
        gt=read_yolo_masks(Path(entry['label']),*shape)
        gb=instance_boundary_union(gt.masks)
        if not gt.masks:gb=np.zeros(shape,bool)
        diag=load(scene/'diagnosis.json');risk=dict(scene=entry['scene'],detected=diag['detected'],selected=diag['selected_count'],skipped_rings=len(diag['skipped_rings']))
        risks.append(risk)
        for m in METHODS:
            vector=scene/m/'final.pixel.geojson';hashes[str(vector)]=digest(vector)
            items=read_items(vector)
            predicted=instances(to_parcels(items,shape));pb=instance_boundary_union(predicted.masks)
            if not predicted.masks:pb=np.zeros(shape,bool)
            rec=load(scene/m/'completed.json')
            measured=boundary_metrics(pb,gb)
            if abs(measured['BF1_2px']-rec['evaluation']['metrics']['boundary_f1'])>1e-12:
                raise ValueError('Supplemental BF1 disagrees with the frozen primary evaluator')
            global_rows.append(dict(scene=entry['scene'],domain=entry['domain'],method=m,triggered=bool(diag['selected']),
                vertices=sum(vertices(p.geometry) for p in items),**measured))
            # All selected windows, including unchanged and rejected proposals.
            for segment in diag['selected']:
                coords=np.asarray(segment['raw_arc'],float)
                x0=max(0,int(np.floor(coords[:,0].min()))-8);y0=max(0,int(np.floor(coords[:,1].min()))-8)
                x1=min(shape[1],int(np.ceil(coords[:,0].max()))+9);y1=min(shape[0],int(np.ceil(coords[:,1].max()))+9)
                window=np.zeros(shape,bool);window[y0:y1,x0:x1]=True
                local_rows.append(dict(scene=entry['scene'],segment_id=segment['segment_id'],method=m,
                    instance_id=segment['instance_id'],window_bounds=[x0,y0,x1,y1],**boundary_metrics(pb,gb,window=window)))
    csv_file(out/'BOUNDARY_TOLERANCES.csv',global_rows);csv_file(out/'FIXED_WINDOW_METRICS.csv',local_rows)
    csv_file(out/'RISK_COVERAGE.csv',risks)
    save(out/'VECTOR_HASHES.json',hashes)
    lookup={(r['scene'],r['method']):r for r in global_rows};window_lookup={(r['scene'],r['segment_id'],r['method']):r for r in local_rows}
    diagnostics=dict(detected_segments=sum(r['detected'] for r in risks),selected_segments=sum(r['selected'] for r in risks),
        triggered_images=sum(r['selected']>0 for r in risks),all_images=len(risks),image_bootstrap={},window_outcomes={})
    for reference in ['dp','dp_rule']:
        a=[];b=[]
        for e in manifest['entries']:
            d=lookup[e['scene'],'dp_visual_agent']['BF1_2px']-lookup[e['scene'],reference]['BF1_2px'];a.append(d)
            if lookup[e['scene'],'dp_visual_agent']['triggered']:b.append(d)
        diagnostics['image_bootstrap'][reference]=dict(all=bootstrap(a),triggered=bootstrap(b),
            improved=sum(x>1e-8 for x in a),tied=sum(abs(x)<=1e-8 for x in a),degraded=sum(x< -1e-8 for x in a))
    for m in ['dp_rule','dp_visual_agent']:
        rows=[]
        for r in local_rows:
            if r['method']!=m:continue
            a=window_lookup[r['scene'],r['segment_id'],'dp'];delta=r['BF1_2px']-a['BF1_2px']
            rows.append(dict(scene=r['scene'],segment_id=r['segment_id'],method=m,delta_BF1_2px=delta,
                outcome='improved' if delta>1e-8 else 'degraded' if delta< -1e-8 else 'tied'))
        diagnostics['window_outcomes'][m]=dict(improved=sum(r['outcome']=='improved' for r in rows),
            degraded=sum(r['outcome']=='degraded' for r in rows),tied=sum(r['outcome']=='tied' for r in rows),
            denominator='All selected fixed windows, not only committed changes; windows overlap and are dependent')
    diagnostics['supplemental_source_sha256']=digest(__file__)
    save(out/'SUPPLEMENTAL.json',diagnostics)
    # Explicitly choose one best and one worst visual-vs-rule BF1 difference.
    candidates=[e for e in manifest['entries'] if lookup[e['scene'],'dp_visual_agent']['triggered']]
    ranked=sorted(candidates,key=lambda e:lookup[e['scene'],'dp_visual_agent']['BF1_2px']-lookup[e['scene'],'dp_rule']['BF1_2px'])
    selected=[('largest_gain',ranked[-1]),('largest_loss',ranked[0])] if ranked else []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from fieldbqc_direct_edit import ring_records
    for kind,entry in selected:
        image=load_image(entry['image'])
        fig,axes=plt.subplots(1,4,figsize=(17,5),layout='constrained')
        for ax,m in zip(axes,METHODS):
            ax.imshow(image[:,:,::-1]);ax.axis('off')
            for p in read_items(out/'scenes'/entry['scene']/m/'final.pixel.geojson'):
                for ring in ring_records(p.geometry):
                    xy=np.asarray(ring['coordinates']+ring['coordinates'][:1]);ax.plot(xy[:,0]-.5,xy[:,1]-.5,lw=.7,color='#ffff00')
                    if m!='raw':ax.scatter(xy[:-1,0]-.5,xy[:-1,1]-.5,s=3,c='#00ffff')
            row=lookup[entry['scene'],m]
            ax.set_title(f'{m}\n{row["vertices"]} vertices | BF1={row["BF1_2px"]:.4f}',fontsize=10)
        fig.suptitle(entry['scene']+' | '+kind+' (selected by observed visual-minus-rule BF1)',fontsize=11)
        fig.savefig(figures/(kind+'.png'),dpi=150);plt.close(fig)
    note=['','## 离线补充评价','',
        f'疑似边段{diagnostics["detected_segments"]}个，实际送审{diagnostics["selected_segments"]}个；触发图像{diagnostics["triggered_images"]}/{diagnostics["all_images"]}。只对送审边段做纠错，不能声称所有疑似边都已处理。','',
        '`BOUNDARY_TOLERANCES.csv`：BF1@1/2/3px与距离；BF1沿用原评价器的OpenCV离散椭圆膨胀，2px逐图已核对一致。Chamfer/P95使用欧氏距离。`FIXED_WINDOW_METRICS.csv`：固定窗口的集合边界指标，全部送审窗口包括未修改者。窗口可能重叠且不代表实例语义正确率。','',
        '探索性逐图配对结果：','']
    for ref,stats in diagnostics['image_bootstrap'].items():
        note.append(f'- 视觉组相对{ref}：BF1改善/持平/退化={stats["improved"]}/{stats["tied"]}/{stats["degraded"]}；均值差={stats["all"]["mean"]:.6f}；图像bootstrap 95%区间={stats["all"]["ci95"]}。')
    note+=['','以上区间受20张开发复用、切片相关及单次模型运行影响，不能解释为正式盲测证明。','',
        '示例按视觉组相对规则组最大改善/最大退化选择，同时展示，不作为平均效果证据。','',
        '注意：下列补充报告已实现多容差与固定窗口集合指标；此前基础表中的“待补充”只指基础入口。预算曲线、实例固定匹配、独立测试与正式消融尚未完成。','']
    with (out/'RESULTS.md').open('a',encoding='utf-8') as f:f.write('\n'.join(note))
    print(json.dumps(diagnostics,ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
