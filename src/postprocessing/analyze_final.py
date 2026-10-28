"""Audit actual DP-point deletion and draw a local before/after example."""
from pathlib import Path
import json
import sys
import numpy as np
from fieldagent import activate_engine
from fieldagent.io import load,save,digest


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();activate_engine()
    from fieldagent.experiment import completed_method,csv_file
    from cvr_agent import read_items,vertices
    from fieldbqc_direct_edit import ring_records
    from run_boundary_tools import load_image
    manifest=load(out/'manifest.json')
    if manifest.get('methods')!=['raw','dp','dp_visual_agent']:raise ValueError('Requires final three-arm manifest')
    if not all(completed_method(out,e,m) for e in manifest['entries'] for m in manifest['methods']):raise ValueError('Finish all three arms first')
    operations=[];coverage=[];provenance=[]
    for e in manifest['entries']:
        scene=out/'scenes'/e['scene'];diag=load(scene/'diagnosis.json')
        coverage.append(dict(scene=e['scene'],detected=diag['detected'],selected=diag['selected_count'],
            removal_arcs=diag['selected_removal_arcs'],potential_DP_points=diag['selected_removable_DP_points']))
        for m in ['dp_visual_agent']:
            trace=load(scene/m/'trace.json')
            for event in trace['events']:
                committed=event.get('status')=='committed'
                if not committed:continue
                op=event['operation'];candidate=event['candidate']
                operations.append(dict(scene=e['scene'],method=m,segment_id=event['segment_id'],
                    task=op['task'],geometry_method=candidate['method'],
                    deleted_existing_DP_points=len(op['deleted_DP_coordinates']),added_control_points=len(op['added_control_coordinates']),
                    net_vertex_change=op['net_vertex_change'],path_before=op['path_points_before'],path_after=op['path_points_after'],
                    chosen_by='GPT',
                    direct_model_path=candidate['method']=='literal_model_authored_path',
                    reason=event.get('decision',{}).get('reason','Rule image/complexity score'),
                    actual_path=candidate['path']))
        for m in manifest['methods']:
            folder=scene/m;record=load(folder/'completed.json')
            assert digest(folder/'final.pixel.geojson')==record['final_sha256']
            provenance.append(dict(scene=e['scene'],method=m,sha256=record['final_sha256']))
    csv_file(out/'DP_POINT_OPERATIONS.csv',operations);csv_file(out/'STRUCTURE_COVERAGE.csv',coverage)
    totals={}
    for m in ['dp_visual_agent']:
        rows=[r for r in operations if r['method']==m]
        totals[m]=dict(deleted_existing_DP_points=sum(r['deleted_existing_DP_points'] for r in rows),
            added_control_points=sum(r['added_control_points'] for r in rows),net_vertex_change=sum(r['net_vertex_change'] for r in rows),
            removal_commits=sum(r['task']=='remove_DP_redundancy' for r in rows),correction_commits=sum(r['task']=='correct_edge' for r in rows),
            direct_model_path_commits=sum(r['direct_model_path'] for r in rows))
        dp_total=sum(load(out/'scenes'/e['scene']/'dp/completed.json')['vertices'] for e in manifest['entries'])
        total=sum(load(out/'scenes'/e['scene']/m/'completed.json')['vertices'] for e in manifest['entries'])
        assert total-dp_total==totals[m]['net_vertex_change']
        assert totals[m]['added_control_points']-totals[m]['deleted_existing_DP_points']==totals[m]['net_vertex_change']
    audit=dict(operations=totals,coverage=coverage,source_sha256=digest(__file__),vector_provenance=provenance,
        interpretation='Gross deletion and net reduction differ because image-supported corrections may add controls. Main arms are raw, DP and DP plus visual agent. Classical geometric edit operations are not claimed novel.')
    save(out/'DP_POINT_AUDIT.json',audit)
    # Choose by actual deleted DP points, never by GT quality, and disclose it.
    candidates=[r for r in operations if r['method']=='dp_visual_agent' and r['net_vertex_change']<0]
    candidates.sort(key=lambda r:(r['net_vertex_change'],r['scene'],r['segment_id']))
    if candidates:
        row=candidates[0];entry=next(e for e in manifest['entries'] if e['scene']==row['scene'])
        diag=load(out/'scenes'/row['scene']/'diagnosis.json');segment=next(s for s in diag['selected'] if s['segment_id']==row['segment_id'])
        image=load_image(entry['image']);old=np.asarray(segment['dp_anchor_path']);new=np.asarray(row['actual_path'])
        points=np.asarray(segment['raw_arc']+segment['dp_anchor_path']+row['actual_path'])
        x0=max(0,points[:,0].min()-12);y0=max(0,points[:,1].min()-12)
        x1=min(image.shape[1],points[:,0].max()+12);y1=min(image.shape[0],points[:,1].max()+12)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,4,figsize=(14,4.5),layout='constrained')
        titles=['Original imagery',f'DP arc: {len(old)} points',f'GPT arc: {len(new)} points','Raw cyan + GPT green']
        for ax,title in zip(axes,titles):
            ax.imshow(image[:,:,::-1]);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.axis('off');ax.set_title(title)
        for ax,xy,color in [(axes[1],old,'#ffff00'),(axes[2],new,'#00ff00')]:
            ax.plot(xy[:,0]-.5,xy[:,1]-.5,color=color,lw=1.5);ax.scatter(xy[:,0]-.5,xy[:,1]-.5,c=color,s=18)
            for k,p in enumerate(xy):ax.annotate(str(k),(p[0]-.5,p[1]-.5),xytext=(3,4),textcoords='offset points',fontsize=9,color='white',bbox=dict(facecolor='black',alpha=.5,pad=.4))
        raw=np.asarray(segment['raw_arc']);axes[3].plot(raw[:,0]-.5,raw[:,1]-.5,color='cyan',lw=1)
        axes[3].plot(new[:,0]-.5,new[:,1]-.5,color='#00ff00',lw=1.5)
        dp=read_items(out/'scenes'/row['scene']/'dp/final.pixel.geojson')
        final=read_items(out/'scenes'/row['scene']/'dp_visual_agent/final.pixel.geojson')
        uid=segment['instance_id'];before=vertices(next(p for p in dp if p.identifier==uid).geometry)
        after=vertices(next(p for p in final if p.identifier==uid).geometry)
        fig.suptitle(f'{row["scene"]}\nLocal arc {len(old)} -> {len(new)} points | parcel ID {uid}: {before} -> {after} total vertices | {row["geometry_method"]}',fontsize=10)
        (out/'figures').mkdir(exist_ok=True)
        fig.savefig(out/'figures/DP_REDUNDANT_POINT_EXAMPLE.png',dpi=160);plt.close(fig)
        save(out/'DP_EXAMPLE_SELECTION.json',dict(selection='Largest committed local DP-point reduction, not selected by GT',operation=row,segment=segment,
            whole_parcel_before=before,whole_parcel_after=after))
    note=['','## 实际DP残留点删除审计','',
        '|分支|删除已有DP点|新增控制点|净点数变化|跨DP边合并提交|纠偏/恢复提交|模型直接路径提交|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,r in totals.items():note.append(f'|{m}|{r["deleted_existing_DP_points"]}|{r["added_control_points"]}|{r["net_vertex_change"]}|{r["removal_commits"]}|{r["correction_commits"]}|{r["direct_model_path_commits"]}|')
    note+=['','`DP_POINT_OPERATIONS.csv`逐项记录删除与新增，不把局部弧段的两端点描述为整个地块只有两个点。',
        '不能把经典端点直连操作当作原创。需要通过多容差DP曲线检验智能体相对传统简化的点数—精度增量。',
        '本轮每图最多12段、每实例3段，上一版为6段/2段；版本之间还改变了观察与操作空间，不能将差值单独归因于提示词。','']
    with (out/'RESULTS.md').open('a',encoding='utf-8') as stream:stream.write('\n'.join(note))
    print(json.dumps(totals,ensure_ascii=False))


if __name__=='__main__':main()
