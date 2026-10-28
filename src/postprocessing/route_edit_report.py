"""Offline attribution of tool choice and subsequent model-authored edits."""
from pathlib import Path
from collections import Counter
import json
from fieldagent.io import load,save,digest


def report(m,source,out):
    from fieldagent.experiment import completed_method,aggregate,csv_file
    from route_edit_experiment import BASE_METHODS,NEW_METHODS,NAMES
    from evaluate_final import bootstrap
    from cvr_agent import read_items
    from fieldbqc_direct_edit import ring_records
    from run_boundary_tools import load_image
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=BASE_METHODS+NEW_METHODS;entries=m['entries'];records={};per_image=[];parcels=[];choices=[];ops=[];calls=[];hashes={};coverage=[]
    for e in entries:
        scene=e['scene'];route_trace=load(out/'scenes'/scene/'tool_routed/trace.json')
        routed={d['instance_id']:d for d in route_trace['events']}
        for method in methods:
            root=out if method in NEW_METHODS else source
            if not completed_method(root,e,method):raise ValueError('All 50 images in all arms required for final comparison')
            folder=root/'scenes'/scene/method;r=load(folder/'completed.json');records[scene,method]=r
            path=folder/'final.pixel.geojson';hashes[str(path)]=digest(path)
            q=r['evaluation']['metrics'];per_image.append(dict(scene=scene,method=method,
                vertices=r['vertices'],boundary_f1=q['boundary_f1'],chamfer_px=q['pixel_boundary_chamfer']))
        n={method:{p['instance_id']:p['after'] for p in records[scene,method]['parcel_vertices']} for method in methods}
        for ident in n['raw']:
            d=routed[ident]
            parcels.append(dict(scene=scene,instance_id=ident,chosen_tool=d['tool'],reason=d['reason'],
                **{method:n[method][ident] for method in methods},
                route_extra_removed=n['dp'][ident]-n['tool_routed'][ident],
                edit_extra_removed=n['tool_routed'][ident]-n['tool_routed_edit'][ident]))
            choices.append(dict(scene=scene,**d))
        trace=load(out/'scenes'/scene/'tool_routed_edit/trace.json')
        pool=load(out/'scenes'/scene/'candidate_pool.json')
        prep=load(out/'scenes'/scene/'preparation.json');coverage.append(dict(scene=scene,**prep))
        for ev in trace['events']:
            c=ev.get('candidate',{});operation=ev.get('operation',{});path=c.get('path',[])
            ops.append(dict(scene=scene,instance_id=ev['instance_id'],segment_id=ev['segment_id'],
                task=ev['task'],action=ev['decision']['action'],status=ev['status'],
                method=c.get('method'),reason=ev['decision']['reason'],
                path_matches_shared_candidate=any(path==p['path'] for p in pool[ev['segment_id']]) if path else False,
                deleted=len(operation.get('deleted_DP_coordinates',[])),added=len(operation.get('added_control_coordinates',[]))))
        folders=list((out/'scenes'/scene/'tool_routed/run').glob('page_*'))+list((out/'scenes'/scene/'tool_routed_edit/run').glob('*'))
        for folder in folders:
            meta=folder/'metadata.json'
            if not meta.exists():continue
            d=load(meta);context=load(folder/'context.json')
            assert context['runtime_GT_access'] is False
            text=(folder/'request.json').read_text(encoding='utf-8')
            assert e['label'] not in text and e['label'].replace('\\','\\\\') not in text
            assert 'boundary_f1' not in text and 'gt_instances' not in text
            assert d['vision'] is True and d['images_sent']>0
            calls.append(dict(scene=scene,stage='routing' if folder.name.startswith('page_') else 'refinement',
                phase=folder.name,seconds=d['seconds'],images_sent=d['images_sent'],response_model=d.get('response_model'),
                input_tokens=d.get('usage',{}).get('input_tokens',0),output_tokens=d.get('usage',{}).get('output_tokens',0)))
    aggregates={method:aggregate([records[e['scene'],method] for e in entries]) for method in methods}
    paired=[];comparisons=[]
    for method,ref in [('tool_routed','dp'),('tool_routed_edit','tool_routed'),('tool_routed_edit','dp'),('tool_routed_edit','dp_visual_agent')]:
        ds=[];nv=0
        for e in entries:
            a=records[e['scene'],ref];b=records[e['scene'],method]
            delta=b['evaluation']['metrics']['boundary_f1']-a['evaluation']['metrics']['boundary_f1'];ds.append(delta)
            difference=b['vertices']-a['vertices'];nv+=difference
            paired.append(dict(scene=e['scene'],method=method,reference=ref,delta_vertices=difference,delta_BF1=delta))
        comparisons.append(dict(method=method,reference=ref,extra_removed=-nv,BF1_delta=bootstrap(ds),
            images_improved=sum(x>1e-8 for x in ds),images_tied=sum(abs(x)<=1e-8 for x in ds),images_worse=sum(x< -1e-8 for x in ds)))
    committed=[r for r in ops if r['status']=='committed'];tool_counts=dict(Counter(c['tool'] for c in choices))
    sessions=[load(p) for p in out.glob('EXECUTION_SESSION_*.json')]
    failed=list((out/'scenes').glob('*/tool_routed/run/*/failed_transport_attempt_*/error.json'))+list((out/'scenes').glob('*/tool_routed_edit/run/*/failed_transport_attempt_*/error.json'))
    api=dict(successful_requests=len(calls),successful_by_stage=dict(Counter(c['stage'] for c in calls)),
        failed_transport_attempts=len(failed),total_attempts=len(calls)+len(failed),
        input_tokens_reported=sum(c['input_tokens'] for c in calls),output_tokens_reported=sum(c['output_tokens'] for c in calls),
        successful_request_seconds=sum(c['seconds'] for c in calls),
        accumulated_execution_seconds=sum(s['wall_seconds'] for s in sessions),workers=m['workers'],
        response_model_fields=sorted({c['response_model'] for c in calls}),
        scope='Execution sessions include routing/crops, local diagnosis, editing and offline evaluation; exclude report/plots and inter-session gaps.',
        costs=None)
    assert len(calls)<=m['max_successful_requests']
    operation_summary=dict(decisions=len(ops),statuses=dict(Counter(o['status'] for o in ops)),
        committed=len(committed),commits_by_task=dict(Counter(o['task'] for o in committed)),
        committed_by_method=dict(Counter(o['method'] for o in committed)),
        deleted=sum(o['deleted'] for o in committed),added=sum(o['added'] for o in committed),
        direct_paths_different_from_candidates=sum(o['method']=='literal_model_authored_path' and not o['path_matches_shared_candidate'] for o in committed))
    assert operation_summary['deleted']-operation_summary['added']==aggregates['tool_routed']['vertices_after']-aggregates['tool_routed_edit']['vertices_after']
    result=dict(images=len(entries),instances=aggregates['raw']['instances'],development_only=True,
        aggregates=aggregates,comparisons=comparisons,tool_choices=tool_counts,
        non_DP_choices=sum(c['tool']!='dp' for c in choices),
        route_point_reduced_parcels=sum(p['route_extra_removed']>0 for p in parcels),
        edit_point_reduced_parcels=sum(p['edit_extra_removed']>0 for p in parcels),
        operations=operation_summary,API=api,coverage=dict(selected_instances=sum(p['selected_instances'] for p in coverage),selected_arcs=sum(p['selected_arcs'] for p in coverage)),
        runtime_GT_access=False,GT_path_and_quality_score_export=False,source_controls_unchanged=True,
        manual_annotation_rows=0,
        interpretation='Tool routing and actual geometry editing are separate evaluated stages. No forced diversity, oracle choice, new parameter tuning, hidden DP or automatic rule finish.',
        limitations=['Same 50 reused validation images, one API realization, no independent test-set superiority.',
            'Initial candidate parameters differ in meaning; route previews are actual outputs of fixed tools.',
            'The editor still observes at most 12 arcs per scene / 3 per instance with fixed endpoints.',
            'Third-party requested/response model aliases do not verify the underlying model identity.',
            'No human adjudication or measured manual editing time.'])
    save(out/'summary.json',result);save(out/'FINAL_VECTOR_HASHES.json',hashes)
    csv_file(out/'METRICS.csv',[dict(method=method,**v) for method,v in aggregates.items()])
    for name,rows in [('PER_IMAGE_METRICS',per_image),('PARCEL_COUNTS',parcels),('TOOL_CHOICES',choices),('EDIT_OPERATIONS',ops),('PAIRED_DELTAS',paired),('API_TIMINGS',calls),('COVERAGE',coverage)]:csv_file(out/(name+'.csv'),rows)
    figs=out/'figures';figs.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    labels={'nth_point':'Nth point','radial_distance':'Radial distance','dp':'DP 1px','dp_visual_agent':'DP + edit (old)',
        'tool_routed':'GPT tool choice','tool_routed_edit':'GPT tool choice + edit'}
    for method in methods[1:]:
        a=aggregates[method];ax.scatter(a['mean_vertices'],a['boundary_f1_macro'],s=85,label=labels[method])
    ax.axhline(aggregates['raw']['boundary_f1_macro'],ls='--',color='grey',label='Raw BF1')
    ax.set(xlabel='Mean vertices per instance',ylabel='Boundary F1 @ 2px',title='Validation 50: tool choice then visual editing')
    ax.legend(fontsize=8);fig.savefig(figs/'ROUTE_EDIT_COMPARISON.png',dpi=180);plt.close(fig)
    pairs=[r for r in paired if r['method']=='tool_routed_edit' and r['reference']=='dp']
    examples={'largest_BF1_gain':max(pairs,key=lambda p:p['delta_BF1'])['scene'],
        'largest_BF1_loss':min(pairs,key=lambda p:p['delta_BF1'])['scene']}
    shown=['raw','dp','dp_visual_agent','tool_routed','tool_routed_edit']
    for label,sname in examples.items():
        entry=next(e for e in entries if e['scene']==sname);image=load_image(entry['image'])
        fig,axes=plt.subplots(1,5,figsize=(21,5),layout='constrained')
        for ax,method in zip(axes,shown):
            root=out if method in NEW_METHODS else source;ax.imshow(image[:,:,::-1]);record=records[sname,method]
            for item in read_items(root/'scenes'/sname/method/'final.pixel.geojson'):
                for ring in ring_records(item.geometry):
                    xy=np.asarray(ring['coordinates']+[ring['coordinates'][0]])
                    ax.plot(xy[:,0],xy[:,1],'.-',color='yellow',lw=.7,markersize=1.4)
            ax.set_title(f"{method}\nN={record['vertices']}; BF1={record['evaluation']['metrics']['boundary_f1']:.4f}",fontsize=9);ax.axis('off')
        fig.suptitle(sname+' / '+label+' (selected illustration)',fontsize=10)
        fig.savefig(figs/(label+'.png'),dpi=170);plt.close(fig)
    for path,h in hashes.items():assert digest(path)==h
    for relative,h in m['source_outputs_sha256'].items():assert digest(source/relative)==h
    save(out/'FINAL_AUDIT.json',dict(new_final_vectors=2*len(entries),baseline_vectors=5*len(entries),
        total_final_vectors_verified=len(hashes),source_files_verified=len(m['source_outputs_sha256']),
        all_source_and_output_hashes_unchanged=True,all_requests_checked_for_GT_export=True,
        successful_requests=len(calls),failed_transport_attempts=len(failed)))
    write_report(m,out,result)
    print(json.dumps(dict(images=len(entries),tool_choices=tool_counts,API=api,
        methods={method:{k:v[k] for k in ['vertices_after','mean_vertices','boundary_f1_macro']} for method,v in aggregates.items()},
        operations=operation_summary,output=str(out)),ensure_ascii=False,indent=2),flush=True)


def write_report(m,out,r):
    from route_edit_experiment import NAMES
    a=r['aggregates'];api=r['API'];op=r['operations']
    lines=['# 智能体逐地块选择减点工具，再进行视觉编辑：真实实验','',
        '同一批50张验证图、1231个预测地块。三种工具的参数固定沿用五组实验：固定抽点k=12、径向距离8px、DP 1px。模型先查看原图与三种实际工具预览，逐地块选择，再对实际混合结果提案—预览—提交。GT仅离线评价。',
        '以下五组基线均复用已校验结果；最后两组为本轮新真实GPT阶段。没有使用历史GPT回答作为新决策。','',
        '|方法|总点数|平均点数|BF1@2px↑|Chamfer px↓|','|---|---:|---:|---:|---:|']
    for method,v in a.items():lines.append(f"|{NAMES[method]}|{v['vertices_after']}|{v['mean_vertices']:.4f}|{v['boundary_f1_macro']:.6f}|{v['chamfer_macro_px']:.6f}|")
    lines+=['','## 两阶段各自的贡献','']
    for c in r['comparisons']:
        lines.append(f"- {NAMES[c['method']]}相对{NAMES[c['reference']]}：净减点{c['extra_removed']:+d}，平均BF1变化{c['BF1_delta']['mean']:+.6f}；逐图{c['images_improved']}升/{c['images_tied']}平/{c['images_worse']}降。")
    lines+=['','负净减点表示点数增加。置信区间为探索性图像bootstrap，验证集复用、空间相关和单次API实现限制推断。较少点但更差精度属于取舍，不自动算综合提升。',
        f"实际工具选择：{json.dumps(r['tool_choices'],ensure_ascii=False)}；非DP选择{r['non_DP_choices']}个。没有强制工具多样化；允许模型全部选DP。TOOL_CHOICES.csv保存每块选择及图像依据，PARCEL_COUNTS.csv保存所有七组点数。",
        f"后续编辑：{op['decisions']}个局部决策、{op['committed']}次提交；任务分布{json.dumps(op['commits_by_task'],ensure_ascii=False)}，方法分布{json.dumps(op['committed_by_method'],ensure_ascii=False)}。删除{op['deleted']}个当前基线坐标，新增{op['added']}个控制点。模型直接写出且不同于候选的提交路径{op['direct_paths_different_from_candidates']}条。",
        '不能将“选中已有工具输出”描述为模型自己创造了几何算法。后续直接坐标编辑、实际搜索和候选选择分别记录；未调用的网络工具不写入贡献。','',
        '## 实现与限制','',
        '本地先生成并展示三种真实预览，模型选择一个完整地块输出；选择后按该输出精确执行。参数不由GT或模型临时改变。初始工具非法结果沿用上一轮回退原始地块的规则，并向模型展示回退记录。',
        f"精修沿用每图12段/每实例3段、固定观察端点和相同累计验收，实际覆盖{r['coverage']['selected_instances']}/1231个地块、{r['coverage']['selected_arcs']}个局部边段。共享代码内部DP命名表示选中的混合BASE，提示词已明确不是所有地块都采用DP。",
        '拒绝精修时保留选中的BASE边段，不回填另一工具或隐藏的第二次DP。原始、工具预览及混合BASE都不是GT。几何通过不是语义修复正确。',
        '本轮没有人工正确性判定、实际人工编辑计时、跨新测试集验证或随机重复API。结果只支持当前验证开发配置。','',
        '## 调用与耗时','',
        f"服务{m['endpoint']}，请求模型别名{m['model_alias']}。共{api['successful_requests']}次成功请求，阶段分布{json.dumps(api['successful_by_stage'],ensure_ascii=False)}；外部失败{api['failed_transport_attempts']}次，总尝试{api['total_attempts']}次。第三方模型别名和响应字段不能验证底层模型身份。",
        f"{api['workers']}任务并发，执行会话累计{api['accumulated_execution_seconds']:.2f}秒（{api['accumulated_execution_seconds']/60:.2f}分钟）；包括工具路由、裁剪、局部诊断、精修和离线评价，不含最终作图/报告和会话间空档。请求累计响应时间{api['successful_request_seconds']:.2f}秒，不能当作并发墙钟。",
        f"服务返回token：输入{api['input_tokens_reported']}，输出{api['output_tokens_reported']}；费用未知，不虚构价格。每次输入、图像、方法、理由、路径、反馈、原始响应和token均保存在对应run目录。",'',
        '## 结果文件','',
        '- summary.json / METRICS.csv：七组总表与两阶段归因。',
        '- TOOL_CHOICES.csv / PARCEL_COUNTS.csv：逐地块工具、理由、点数。',
        '- EDIT_OPERATIONS.csv / API_TIMINGS.csv / PAIRED_DELTAS.csv：编辑、调用与配对统计。',
        '- scenes/<scene>/tool_routed/run/page_*/：第一阶段选择上下文、图像、响应。',
        '- scenes/<scene>/tool_routed_edit/run/：后续编辑提案、实际预览和确认。',
        '- 两个方法各50份final.pixel.geojson；五组基线仍在manifest.source_run。',
        '- source_snapshot / manifest.json / FINAL_AUDIT.json：代码、参数、来源和指纹。','']
    (out/'工具路由与视觉精修实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
