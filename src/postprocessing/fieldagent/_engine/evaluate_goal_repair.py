"""Offline evaluation of v9; only this stage reads reference annotations."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_agent_experiments import instances, measure, summary, write_csv
from evaluate_geometry_boundary_iou import matches, measure_boundary, summarize as boundary_summary
from evaluate_mechanism_metrics import pairwise_overlap, read_yolo_masks
from evaluate_selective_repair import reference_vectors, shape_metrics
from fieldbqc_geometry import geometry_mask, scene_metrics, vertices
from fieldbqc_goal_repair import conflict_components
from run_agent_experiments import fingerprint, load_parcels, read_json, write_json
from run_geometry_experiments import vector_input


def identity_audit(initial, final, trace):
    old={p.identifier:p for p in initial}
    new={p.identifier:p for p in final}
    removed=set(old)-set(new)
    lineage={r["original_id"]:r for r in trace["lineage"]}
    if len(lineage)!=len(old) or set(lineage)!=set(old) or set(new)-set(old):
        raise ValueError("Missing/duplicate lineage or invented IDs")
    for uid,p in old.items():
        r=lineage[uid]
        if uid in removed:
            if r.get("active") is not False or r.get("operation")!="overlap_suppress" or r.get("replaced_by") not in new:
                raise ValueError("Unexplained instance loss or representative missing")
        elif r.get("active") is not True:
            raise ValueError("Active lineage inconsistent with output")
        if uid in new and (p.cls,p.score,p.source_ids)!=(new[uid].cls,new[uid].score,new[uid].source_ids):
            raise ValueError("Unexplained metadata change")
    survivor_original=sum(vertices(old[i].geometry) for i in new)
    suppressed=sum(vertices(old[i].geometry) for i in removed)
    if survivor_original+suppressed!=sum(vertices(p.geometry) for p in initial):
        raise ValueError("Vertex accounting failed")
    return dict(identity_lineage_ok=True,removed=len(removed),survivor_original_vertices=survivor_original,
                suppressed_original_vertices=suppressed)


def frozen_ownership(initial, gt, shape):
    """Exploratory best-IoU association (duplicates allowed), frozen at G0.

    It is NOT independently annotated conflict truth. Report threshold, margin
    and judgeable coverage; no ground truth enters the repair process.
    """
    masks=[geometry_mask(p.geometry,shape) for p in initial]
    classes=np.asarray([p.cls for p in initial])
    iou,_,_=pairwise_overlap(gt.masks,masks,gt.classes,classes)
    mapping={}
    associations=[]
    for c,p in enumerate(initial):
        scores=iou[:,c]
        order=np.argsort(-scores)
        best=int(order[0]) if len(order) else -1
        value=float(scores[best]) if best>=0 else 0.
        second=float(scores[order[1]]) if len(order)>1 else 0.
        qualified=value>=.30 and value-second>=.10
        if qualified:
            mapping[p.identifier]=best
        associations.append(dict(prediction_id=p.identifier,best_reference=best,best_iou=value,
                                 runner_up_iou=second,qualified=qualified))
    occupied=sum((m.astype(np.int32) for m in masks),np.zeros(shape,np.int32))
    gt_count=sum((m.astype(np.int32) for m in gt.masks),np.zeros(shape,np.int32))
    gt_index=np.full(shape,-1,np.int32)
    for index,mask in enumerate(gt.masks):
        gt_index[mask]=index
    roi=occupied>1
    unqualified=np.zeros(shape,bool)
    correct_initial_support=np.zeros(shape,bool)
    for p,mask in zip(initial,masks):
        if p.identifier not in mapping:
            unqualified|=mask
        else:
            correct_initial_support|=mask & (gt_index==mapping[p.identifier])
    judgeable=roi & (gt_count==1) & ~unqualified & correct_initial_support
    original_owner=np.full(shape,-1,np.int32)
    original_indices={p.identifier:index for index,p in enumerate(initial)}
    for p,mask in zip(initial,masks):
        original_owner[mask]=original_indices[p.identifier]
    return dict(mapping=mapping,associations=associations,roi=roi,judgeable=judgeable,gt_index=gt_index,
                original_indices=original_indices,original_owner=original_owner,non_disputed=occupied==1)


def ownership_score(final,frozen,shape):
    occupied=np.zeros(shape,np.int32)
    owner=np.full(shape,-1,np.int32)
    original_owner=np.full(shape,-1,np.int32)
    for p in final:
        mask=geometry_mask(p.geometry,shape)
        occupied+=mask
        owner[mask]=frozen["mapping"].get(p.identifier,-2)
        original_owner[mask]=frozen["original_indices"][p.identifier]
    denominator=int(frozen["judgeable"].sum())
    correct=int(np.count_nonzero(frozen["judgeable"] & (occupied==1) & (owner==frozen["gt_index"])))
    assigned=int(np.count_nonzero(frozen["judgeable"] & (occupied==1)))
    return dict(ownership_proxy_correct=correct,ownership_proxy_judgeable=denominator,
                ownership_proxy_uniquely_assigned=assigned,
                ownership_proxy_wrong_unique=assigned-correct,
                ownership_proxy_remaining_overlap=int(np.count_nonzero(frozen["judgeable"] & (occupied>1))),
                ownership_proxy_uncovered=int(np.count_nonzero(frozen["judgeable"] & (occupied==0))),
                non_disputed_pixels_before=int(frozen["non_disputed"].sum()),
                non_disputed_pixels_uncovered=int(np.count_nonzero(frozen["non_disputed"] & (occupied==0))),
                non_disputed_pixels_identity_changed=int(np.count_nonzero(frozen["non_disputed"] & ((occupied!=1) | (original_owner!=frozen["original_owner"])))),
                initial_disputed_pixels=int(frozen["roi"].sum()),
                ownership_proxy_accuracy=correct/denominator if denominator else None)


def make_visuals(root,rows,entries,methods,references=None):
    output=root/"visualizations"
    output.mkdir(exist_ok=True)
    index={(r["method"],r["image_id"]):r for r in rows}
    ids=[e["image_id"] for e in entries]
    # User's two discussed examples first, then largest initial overlap and
    # largest goal-versus-rule vertex difference. This is illustration selection.
    preferred=["DDFHAPD_HuN_id_5_8_8_add_0_0_256_256","DDFHAPD_HuN_id_8_11_0_dia_0_0_256_256"]
    selected=[i for i in preferred if i in ids]
    ranked=sorted(ids,key=lambda i:-index[(methods[0],i)]["overlap_before"])
    if "A1-GOAL-LOCAL" in methods and "B1-RULE" in methods:
        ranked+=sorted(ids,key=lambda i:-abs(index[("A1-GOAL-LOCAL",i)]["vertices_after"]-index[("B1-RULE",i)]["vertices_after"]))
    for i in ranked:
        if i not in selected:
            selected.append(i)
        if len(selected)>=5:
            break
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",16)
    small=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",13)
    panels=["image"]+(["reference"] if references else [])+[m for m in ("B0-RAW","B1-RULE","A1-GOAL-LOCAL","A2-GOAL-OPEN","D1-LLM-ONCE","D2-AGENT-EDIT") if m in methods]
    canvas=Image.new("RGB",(320*len(panels),370*len(selected)),"white")
    draw=ImageDraw.Draw(canvas)
    colors=[(0,255,255),(255,0,255),(0,230,0),(255,120,0),(255,70,70),(50,160,255)]
    for row,stem in enumerate(selected):
        entry=next(e for e in entries if e["image_id"]==stem)
        image=cv2.imread(entry["source"])
        initial_ids=[p.identifier for p in vector_input(root/"initial"/(stem+".pixel.geojson"))]
        color_by_id={uid:colors[k%len(colors)] for k,uid in enumerate(initial_ids)}
        for col,method in enumerate(panels):
            frame=image.copy()
            if method=="reference":
                for k,mask in enumerate(references[stem].masks):
                    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame,contours,-1,(255,255,0),1)
                    if contours:
                        x,y,_,_=cv2.boundingRect(max(contours,key=cv2.contourArea))
                        cv2.putText(frame,"g"+str(k),(x,max(9,y)),cv2.FONT_HERSHEY_SIMPLEX,.28,(255,255,0),1)
            elif method!="image":
                parcels,_=load_parcels(root/method/"masks"/(stem+".npz"))
                occupied=sum((p.mask.astype(np.int16) for p in parcels),np.zeros(image.shape[:2],np.int16))
                frame[occupied>1]=(frame[occupied>1]*.4+np.array([0,0,255])*.6).astype(np.uint8)
                vectors=vector_input(root/method/"vectors"/(stem+".pixel.geojson"))
                for k,(p,v) in enumerate(zip(parcels,vectors)):
                    color=color_by_id[p.parcel_id]
                    from fieldbqc_geometry import polygons
                    for part in polygons(v.geometry):
                        for ring in [part.exterior]+list(part.interiors):
                            points=np.rint(np.asarray(ring.coords)).astype(np.int32)
                            cv2.polylines(frame,[points],True,color,1)
                            for x,y in points[:-1]:
                                cv2.circle(frame,(int(x),int(y)),1,color,-1)
                    if not v.geometry.is_empty:
                        x,y,_,_=v.geometry.bounds
                        cv2.putText(frame,str(p.parcel_id),(int(x),max(9,int(y))),cv2.FONT_HERSHEY_SIMPLEX,.28,color,1)
            picture=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((310,310))
            x,y=col*320,row*370
            canvas.paste(picture,(x+5,y+36))
            draw.text((x+5,y+5),method,font=font,fill="black")
            if method not in {"image","reference"}:
                r=index[(method,stem)]
                draw.text((x+5,y+348),f"N={r['vertices_after']}; E={r['overlap_after']:.1f}; F1={r['mean_f1']:.4f}",font=small,fill="black")
            elif method=="image":
                draw.text((x+5,y+348),stem[:39],font=small,fill="black")
            else:
                draw.text((x+5,y+348),"Rasterized reference; g IDs differ from predictions",font=small,fill="black")
    canvas.save(output/"paired_examples.png")
    write_json(output/"selection.json",dict(selected=selected,
        rule="Two previously discussed overlap examples, then largest G0 overlap, then largest goal-rule vertex difference; no claim of representativeness",
        red="remaining overlapping pixels; prediction colors stable by original ID; reference g IDs are independent"))


def evaluate(args):
    root=args.runs
    manifest=read_json(root/"manifest.json")
    complete=(root/"completed.json").exists()
    methods=args.methods or manifest["methods"]
    if not set(methods)<=set(manifest["methods"]):
        raise ValueError("Requested methods not part of original experiment")
    requested_complete=all((root/m/"completed.json").exists() for m in methods)
    if not complete and not requested_complete and not args.allow_partial:
        raise ValueError("Incomplete requested groups; --allow-partial required to disclose paired coverage")
    eligible=[]
    excluded=[]
    for entry in manifest["inputs"]:
        missing=[]
        for method in methods:
            p=root/method/"traces"/(entry["image_id"]+".json")
            if not p.exists() or read_json(p).get("planner_failed"):
                missing.append(method)
        if missing:
            excluded.append(dict(image_id=entry["image_id"],missing_or_failed=missing))
        else:
            eligible.append(entry)
    if not eligible:
        raise ValueError("No complete paired images available")
    ordinary_selection=bool(args.methods) and not any(m.startswith(("A1-", "A2-", "D1-", "D2-")) for m in methods)
    output=root/(("ordinary_evaluation" if ordinary_selection else "selected_methods_evaluation") if args.methods else "evaluation" if complete else "paired_partial_evaluation")
    output.mkdir(exist_ok=True)
    analysis_snapshot=output/"analysis_source_snapshot"
    analysis_snapshot.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__),analysis_snapshot/Path(__file__).name)
    analysis_core=Path(__file__).parent/"fieldbqc_goal_repair.py"
    shutil.copyfile(analysis_core,analysis_snapshot/analysis_core.name)
    write_json(output/"coverage.json",dict(parent_complete=complete,requested_methods_complete=requested_complete,
        methods=methods,paired_images=len(eligible),
        intended_images=len(manifest["inputs"]),excluded=excluded,selection="all methods complete; no metric-based exclusion"))
    data={}
    labels={}
    mapping_rows=[]
    for entry in eligible:
        stem,shape=entry["image_id"],tuple(entry["shape"])
        label=args.dataset_root/"labels"/args.split/(stem+".txt")
        labels[stem]=fingerprint(label)
        gt=read_yolo_masks(label,*shape)
        reference,reference_counts=reference_vectors(label,shape)
        initial=vector_input(root/"initial"/(stem+".pixel.geojson"))
        frozen=frozen_ownership(initial,gt,shape)
        mapping_rows.extend(dict(image_id=stem,**row) for row in frozen["associations"])
        data[stem]=(gt,reference,reference_counts,initial,frozen)
    rows=[]
    audits=[]
    all_summaries=[]
    for method in methods:
        totals=Counter()
        bt=Counter()
        thresholds={t:Counter() for t in (.5,.75,.9)}
        geometry_values=[]
        for entry in eligible:
            stem,shape=entry["image_id"],tuple(entry["shape"])
            gt,reference,reference_counts,initial,frozen=data[stem]
            final=vector_input(root/method/"vectors"/(stem+".pixel.geojson"))
            parcels,actual_shape=load_parcels(root/method/"masks"/(stem+".npz"))
            trace=read_json(root/method/"traces"/(stem+".json"))
            for kind,suffix in (("vectors",".pixel.geojson"),("masks",".npz")):
                if fingerprint(root/method/kind/(stem+suffix))!=trace["output_sha256"][kind]:
                    raise ValueError("Output hash mismatch")
            if actual_shape!=shape or [p.parcel_id for p in parcels]!=[p.identifier for p in final]:
                raise ValueError("Output IDs/shape mismatch")
            if not all(np.array_equal(p.mask,geometry_mask(v.geometry,shape)) for p,v in zip(parcels,final)):
                raise ValueError("Vector/raster mismatch")
            audit=identity_audit(initial,final,trace)
            audits.append(dict(method=method,image_id=stem,roundtrip_ok=True,**audit))
            predicted=instances(parcels)
            result,counts=measure(gt,predicted,shape)
            totals.update(counts)
            boundary=measure_boundary(gt,predicted)
            bt.update(boundary)
            iou,_,_=pairwise_overlap(gt.masks,predicted.masks,gt.classes,predicted.classes)
            f1s=[]
            row=dict(method=method,image_id=stem)
            for t in thresholds:
                n=len(matches(iou,t))
                thresholds[t].update(tp=n,fp=len(parcels)-n,fn=len(gt.masks)-n)
                f=2*n/max(1,len(parcels)+len(gt.masks))
                row[f"f1_{int(t*100)}"]=f
                f1s.append(f)
            values=[]
            for r,c in matches(iou,.5):
                v=shape_metrics(final[c].geometry,reference[r],vertices(final[c].geometry),reference_counts[r],iou[r,c])
                if v:
                    values.append(v)
            geometry_values.extend(values)
            plans=[e for e in trace["action_trace"] if e["action"]=="plan"]
            usages=[e.get("billed_usage",{}) for e in plans]
            removed=[r for r in trace["lineage"] if r.get("active") is False]
            same,different,unknown=0,0,0
            for r in removed:
                a,b=frozen["mapping"].get(r["original_id"]),frozen["mapping"].get(r["replaced_by"])
                if a is None or b is None:
                    unknown+=1
                elif a==b:
                    same+=1
                else:
                    different+=1
            row.update(mean_f1=float(np.mean(f1s)),tolerant_boundary_f1_2px=result["boundary_f1"],
                matched_pairs=len(values),
                pq_like=result["pq"],union_iou=result["union_iou"],**boundary_summary(boundary),
                vertices_before=trace["initial"]["vertices"],vertices_after=trace["final"]["vertices"],
                survivor_original_vertices=audit["survivor_original_vertices"],
                survivor_compression=1-trace["final"]["vertices"]/max(1,audit["survivor_original_vertices"]),
                suppressed_original_vertices=audit["suppressed_original_vertices"],
                suppressed_instances=audit["removed"],suppression_same_reference_proxy=same,
                suppression_different_reference_proxy=different,suppression_unresolved_proxy=unknown,
                instances_before=len(initial),instances_after=len(final),gt_instances=len(gt.masks),
                overlap_before=trace["initial"]["overlap_excess_area"],overlap_after=trace["final"]["overlap_excess_area"],
                union_area_before=trace["initial"]["union_area"],invalid_after=trace["final"]["invalid_geometries"],
                initial_conflict_components=len(conflict_components(initial)),
                remaining_conflict_components=len(conflict_components(final)),
                reviewed_instances=len(trace["review_ids"]),tool_checks=trace["candidate_evaluations"],
                api_calls=sum(e.get("api_attempts",int(e.get("api_request",False))) for e in plans),
                input_tokens=sum(u.get("input_tokens",u.get("prompt_tokens",0)) for u in usages),
                output_tokens=sum(u.get("output_tokens",u.get("completion_tokens",0)) for u in usages),
                local_discovery_calls=sum(e["action"] in {"find_tools","discover_local"} for e in trace["action_trace"]),
                web_search_calls=sum(e["action"]=="search_web" for e in trace["action_trace"]),
                source_read_calls=sum(e["action"]=="read_source" for e in trace["action_trace"]),
                build_attempts=sum(e["action"]=="build_tool" for e in trace.get("extension_events",[])),
                registered_tools=len(trace.get("generated_tool_ids",[])),
                generated_tool_trials=trace.get("generated_tool_trials",0),
                generated_tool_commits=trace.get("generated_tool_commits",0),
                build_test_checks=trace.get("build_test_checks",0),
                inspect_calls=sum(e["action"]=="inspect" for e in trace["action_trace"]),
                restores=sum(e["action"]=="restore" and e.get("accepted",False) for e in trace["action_trace"]),
                seconds=trace["postprocess_seconds"],**ownership_score(final,frozen,shape))
            if values:
                row.update({k:float(np.mean([v[k] for v in values])) for k in values[0]})
            rows.append(row)
        selected=[r for r in rows if r["method"]==method]
        base=summary(totals)
        s=dict(method=method,images=len(selected),predicted_instances=int(base["pred_count"]),
               matched_pairs=len(geometry_values),reference_instances=sum(r["gt_instances"] for r in selected),
               matched_prediction_coverage=len(geometry_values)/max(1,int(base["pred_count"])),
               matched_reference_coverage=len(geometry_values)/max(1,sum(r["gt_instances"] for r in selected)),
               mean_f1=float(np.mean([2*c["tp"]/max(1,2*c["tp"]+c["fp"]+c["fn"]) for c in thresholds.values()])),
               tolerant_boundary_f1_2px=base["boundary_f1"],pq_like=base["pq"],union_iou=base["union_iou"],
               **boundary_summary(bt))
        for t,c in thresholds.items():
            s.update({f"precision_{int(t*100)}":c["tp"]/max(1,c["tp"]+c["fp"]),
                      f"recall_{int(t*100)}":c["tp"]/max(1,c["tp"]+c["fn"]),
                      f"f1_{int(t*100)}":2*c["tp"]/max(1,2*c["tp"]+c["fp"]+c["fn"])})
        sum_keys=("vertices_before","vertices_after","survivor_original_vertices","suppressed_original_vertices",
                  "suppressed_instances","suppression_same_reference_proxy","suppression_different_reference_proxy",
                  "suppression_unresolved_proxy","overlap_before","overlap_after","union_area_before",
                  "invalid_after","tool_checks","api_calls","input_tokens","output_tokens","local_discovery_calls",
                  "web_search_calls","source_read_calls","build_attempts","registered_tools",
                  "generated_tool_trials","generated_tool_commits","build_test_checks",
                  "inspect_calls","restores","ownership_proxy_correct","ownership_proxy_judgeable","initial_disputed_pixels",
                  "ownership_proxy_uniquely_assigned","ownership_proxy_wrong_unique","ownership_proxy_remaining_overlap","ownership_proxy_uncovered",
                  "non_disputed_pixels_before","non_disputed_pixels_uncovered","non_disputed_pixels_identity_changed",
                  "initial_conflict_components","remaining_conflict_components","reviewed_instances")
        for k in sum_keys:
            s[k]=sum(r[k] for r in selected)
        s.update(total_compression=1-s["vertices_after"]/max(1,s["vertices_before"]),
                 survivor_compression=1-s["vertices_after"]/max(1,s["survivor_original_vertices"]),
                 overlap_reduction=1-s["overlap_after"]/max(1e-9,s["overlap_before"]),
                 overlap_over_initial_union=s["overlap_after"]/max(1e-9,s["union_area_before"]),
                 ownership_proxy_accuracy=s["ownership_proxy_correct"]/s["ownership_proxy_judgeable"] if s["ownership_proxy_judgeable"] else None,
                 ownership_proxy_coverage=s["ownership_proxy_judgeable"]/s["initial_disputed_pixels"] if s["initial_disputed_pixels"] else None,
                 ownership_proxy_accuracy_when_unique=s["ownership_proxy_correct"]/s["ownership_proxy_uniquely_assigned"] if s["ownership_proxy_uniquely_assigned"] else None,
                 non_disputed_uncovered_fraction=s["non_disputed_pixels_uncovered"]/max(1,s["non_disputed_pixels_before"]),
                 mean_seconds=float(np.mean([r["seconds"] for r in selected])),
                 semantic_correct_repair_rate=None,normal_region_damage_rate=None,
                 independently_verified_duplicate_precision=None,shared_edge_recall=None)
        if geometry_values:
            s.update({k:float(np.mean([v[k] for v in geometry_values])) for k in geometry_values[0]})
        all_summaries.append(s)
        print(f"[evaluation] {method}: N={s['vertices_after']} survivor-compression={s['survivor_compression']:.2%} "
              f"F1={s['mean_f1']:.6f} BF1={s['tolerant_boundary_f1_2px']:.6f} E={s['overlap_after']:.2f}",flush=True)
    comparisons=[]
    rng=np.random.default_rng(20261021)
    for left,right in (("B1-RULE","B0-RAW"),("A1-GOAL-LOCAL","B0-RAW"),("A1-GOAL-LOCAL","B1-RULE"),("A2-GOAL-OPEN","B0-RAW"),("A2-GOAL-OPEN","B1-RULE"),
                       ("D1-LLM-ONCE","B0-RAW"),("D1-LLM-ONCE","B1-RULE"),("D2-AGENT-EDIT","B0-RAW"),("D2-AGENT-EDIT","B1-RULE"),("D2-AGENT-EDIT","D1-LLM-ONCE")):
        if left not in methods or right not in methods:
            continue
        a={r["image_id"]:r for r in rows if r["method"]==left}
        b={r["image_id"]:r for r in rows if r["method"]==right}
        for metric in ("mean_f1","tolerant_boundary_f1_2px","pq_like","vertices_after","overlap_after","seconds"):
            delta=np.asarray([a[i][metric]-b[i][metric] for i in a],float)
            means=rng.choice(delta,size=(2000,len(delta)),replace=True).mean(axis=1)
            comparisons.append(dict(comparison=left+" minus "+right,metric=metric,mean_delta=float(delta.mean()),
                                    ci95_low=float(np.quantile(means,.025)),ci95_high=float(np.quantile(means,.975)),
                                    unit="image; exploratory, scene independence unverified"))
    write_json(output/"summary.json",all_summaries)
    write_csv(output/"summary.csv",all_summaries)
    write_csv(output/"per_image.csv",rows)
    write_csv(output/"output_audit.csv",audits)
    write_csv(output/"paired_comparisons.csv",comparisons)
    write_csv(output/"frozen_reference_associations.csv",mapping_rows)
    common_totals={m:dict(method=m,original_vertices=0,final_vertices=0,common_instances=0) for m in methods}
    for entry in eligible:
        stem=entry["image_id"]
        lookup={m:{p.identifier:p for p in vector_input(root/m/"vectors"/(stem+".pixel.geojson"))} for m in methods}
        common=set.intersection(*(set(v) for v in lookup.values()))
        initial={p.identifier:p for p in data[stem][3]}
        for method in methods:
            common_totals[method]["original_vertices"]+=sum(vertices(initial[i].geometry) for i in common)
            common_totals[method]["final_vertices"]+=sum(vertices(lookup[method][i].geometry) for i in common)
            common_totals[method]["common_instances"]+=len(common)
    for r in common_totals.values():
        r["compression"]=1-r["final_vertices"]/max(1,r["original_vertices"])
    write_csv(output/"common_survivor_complexity.csv",list(common_totals.values()))
    cost_rows=[]
    for method in manifest["methods"]:
        traces=[]
        for folder in (root/method/"traces",root/method/"failed_attempts"):
            if folder.exists():
                traces.extend(read_json(p) for p in sorted(folder.glob("*.json")))
        plans=[e for t in traces for e in t["action_trace"] if e["action"]=="plan"]
        usages=[e.get("billed_usage",{}) for e in plans]
        cost_rows.append(dict(method=method,saved_attempts=len(traces),
            failed_attempts=sum(t["planner_failed"] for t in traces),
            request_attempts=sum(e.get("api_attempts",int(e.get("api_request",False))) for e in plans),
            requests_without_returned_usage=sum(e.get("api_request",False) and not e.get("billed_usage") for e in plans),
            known_input_tokens=sum(u.get("input_tokens",u.get("prompt_tokens",0)) for u in usages),
            known_output_tokens=sum(u.get("output_tokens",u.get("completion_tokens",0)) for u in usages),
            all_attempt_seconds=sum(t["postprocess_seconds"] for t in traces),
            cost_scope="All intended images and archived failed attempts, including cases excluded from paired evaluation"))
    write_csv(output/"all_attempt_costs.csv",cost_rows)
    write_json(output/"protocol.json",dict(evaluator_sha256=fingerprint(Path(__file__)),labels=labels,
        evaluator_snapshot=str((analysis_snapshot/Path(__file__).name).relative_to(root)),
        analysis_goal_core_sha256=fingerprint(analysis_core),
        repair_time_evaluator_sha256=manifest["signature"]["code_sha256"]["evaluate_goal_repair.py"],
        scope="Reused development images, not independent test. A1 uses known tools; A2 discovers/authors tools; D1/D2 author direct geometry edits with single-shot/actual feedback respectively. Audit actual actions separately; failed groups are not efficacy baselines.",
        instance_matching="class-aware one-to-one max valid cardinality then IoU at .50/.75/.90",
        mean_f1="Arithmetic mean of dataset F1 at three thresholds, not mean of per-image F1",
        tolerant_bf1="Legacy pixel boundary matching tolerance=2, separate from inner-band BIoU",
        conditional_geometry="IoU>=0.5 matched pairs; PoLiS/MTA/P95 sampled <=2px; coverage must accompany",
        vertices="Actual vector outer+inner, no duplicated closure; deactivation vertices separated from survivor compression",
        overlap="sum area minus union area; fixed initial union denominator",
        ownership_proxy="Frozen best-reference association at initial IoU>=.30 and best-minus-runnerup>=.10, duplicates allowed; initial multiply-covered cells with unique GT owner, all initial covering predictions qualified, and that owner supported",
        ownership_limit="Exploratory reference-derived association, NOT independently annotated semantic conflict truth. Compare only fixed judgeable denominator with disclosed coverage",
        unavailable="Independent semantic correct-repair/duplicate precision, normal-damage and SER remain N/A",
        partial=not complete))
    make_visuals(root,rows,eligible,methods,{stem:data[stem][0] for stem in data})
    title=("# 普通双任务后处理：开发实验" if ordinary_selection else "# 指定已完成方法：开发实验") if args.methods else "# 双任务后处理与目标驱动智能体：开发实验"
    method_description=("D1为单次大模型直接几何编辑；D2为实际候选反馈驱动的直接编辑智能体。失败组不能用保留的原始输出代替有效性结果；单次判断有效不等于闭环已验证。"
                        if manifest['signature'].get('goal_profile')=='direct-edit' else
                        "A1为本地已知工具下的目标驱动控制器：自主诊断、选对象/算法/参数、试运行、反馈后提交、回退与停止；不包含网络新工具发现。")
    lines=[title, "",
           f"本表所列方法完整配对 {len(eligible)}/{len(manifest['inputs'])} 图；所列方法全部完成：{requested_complete}；包含其他方法的父实验完成：{complete}。所有组使用同一冻结分割结果。", "",
           "研究任务仅为轮廓简化和重叠消解，几何合法化为共用支撑。不主动填孔，不设固定点数，不做末尾共同减点。", "",
           method_description, "",
           "| 方法 | 活动实例 | 顶点 | 存续实例压缩率 | 去重移除顶点 | 平均F1 | BF1@2px | PQ-like | 重叠超额面积 | 秒/图 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in all_summaries:
        lines.append(f"| {s['method']} | {s['predicted_instances']} | {s['vertices_after']} | {s['survivor_compression']:.2%} | {s['suppressed_original_vertices']} | {s['mean_f1']:.6f} | {s['tolerant_boundary_f1_2px']:.6f} | {s['pq_like']:.6f} | {s['overlap_after']:.2f} | {s['mean_seconds']:.2f} |")
    lines += ["", "| 方法 | 重叠下降 | 参考关联归属代理 | 可判定覆盖 | 停用实例 | 同参考/不同参考/未定代理 | 模型请求 | 输入/输出token |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in all_summaries:
        accuracy=f"{s['ownership_proxy_accuracy']:.2%}" if s["ownership_proxy_accuracy"] is not None else "N/A"
        coverage=f"{s['ownership_proxy_coverage']:.2%}" if s["ownership_proxy_coverage"] is not None else "N/A"
        lines.append(f"| {s['method']} | {s['overlap_reduction']:.2%} | {accuracy} | {coverage} | {s['suppressed_instances']} | {s['suppression_same_reference_proxy']}/{s['suppression_different_reference_proxy']}/{s['suppression_unresolved_proxy']} | {s['api_calls']} | {s['input_tokens']}/{s['output_tokens']} |")
    idx={s["method"]:s for s in all_summaries}
    if "A2-GOAL-OPEN" in idx:
        lines += ["", "本轮包含A2开放能力原型：可检索实际资料并构建几何函数；是否真实检索、注册、试验、提交，须逐项核对extension_events及dynamic_tools，不能由能力可用推断已使用。新增工具与决策的贡献尚需冻结扩展工具库后的等工具对照。"]
    if "A1-GOAL-LOCAL" in idx and "B1-RULE" in idx:
        a,b=idx["A1-GOAL-LOCAL"],idx["B1-RULE"]
        lines += ["",f"智能体相对规则：平均F1差 {a['mean_f1']-b['mean_f1']:+.6f}，BF1差 {a['tolerant_boundary_f1_2px']-b['tolerant_boundary_f1_2px']:+.6f}，PQ-like差 {a['pq_like']-b['pq_like']:+.6f}，重叠超额面积差 {a['overlap_after']-b['overlap_after']:+.2f}，存续压缩率差 {a['survivor_compression']-b['survivor_compression']:+.2%}。"]
    lines += ["", "| 方法 | IoU≥0.5条件匹配数 | 预测匹配覆盖 | 参考匹配覆盖 | 复核实例数 |", "|---|---:|---:|---:|---:|"]
    for s in all_summaries:
        lines.append(f"| {s['method']} | {s['matched_pairs']} | {s['matched_prediction_coverage']:.2%} | {s['matched_reference_coverage']:.2%} | {s['reviewed_instances']} |")
    lines += ["", "| 方法 | 所有方法共同存续实例数 | 同一批实例原始顶点 | 最终顶点 | 共同存续集合压缩率 |", "|---|---:|---:|---:|---:|"]
    for method,r in common_totals.items():
        lines.append(f"| {method} | {r['common_instances']} | {r['original_vertices']} | {r['final_vertices']} | {r['compression']:.2%} |")
    lines += ["", "共同存续集合使顶点比较的实例集合一致，但仍是条件性分析，不能代替停用错误与全体实例评价。裁剪也可能改变存续实例的顶点，因此本表不是纯简化算法实验。", "",
        "| 方法 | 固定可判定争议像素 | 正确单一归属 | 错误单一归属 | 仍重叠 | 留空 | 初始非争议区覆盖丢失 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in all_summaries:
        lines.append(f"| {s['method']} | {s['ownership_proxy_judgeable']} | {s['ownership_proxy_correct']} | {s['ownership_proxy_wrong_unique']} | {s['ownership_proxy_remaining_overlap']} | {s['ownership_proxy_uncovered']} | {s['non_disputed_uncovered_fraction']:.4%} |")
    lines += ["", "上表区分未处理和错误分配；代理无法确定身份的初始区域不进入可判定分母，但仍计入整体重叠、实例质量和覆盖率。非争议区为G0仅覆盖一次的固定像素集合，其覆盖丢失不是人工标注的正常结构损伤率。", "",
        "归属与停用的‘同参考’统计是冻结初始预测与已有标注的探索性关联代理，不是人工确认的语义真值；条件覆盖和阈值见protocol.json。独立正确修复率、正常区域损伤率、去重精确率及SER仍为N/A，不能由几何通过率推算。", "",
        "点数分开记账：总减少可能包含停用实例移除；存续压缩率只对各方法存续集合计算，集合不同使其具有条件性。共同存续集合已单列；禁止停用和裁剪的固定实例轮廓专项仍需补充。", "",
        "数据是反复使用的开发图，参数未在独立场景冻结标定，模型仅一次重复；图像自助区间仅为探索性描述，不宣称场景级显著性或论文创新已证实。", "",
        "费用只记录真实返回用量，无可靠单价不估算金额。外部失败与未返回用量不可按零费用处理。主表统计配对样本最终轨迹；all_attempt_costs.csv另统计全部图像和已归档失败尝试，避免遗漏配对集外成本。失败轨迹及覆盖排除单独保存，不做静默规则回退。", "",
        "[并列案例](visualizations/paired_examples.png)。案例优先选取此前讨论的两个重叠样例，再取初始重叠最大和方法差异较大的图；选择依据已保存，不代表随机样本或整体结论。", "",
        f"离线评价目录：{output.name}。包括完整指标、逐图结果、配对差值、栅矢/身份审计、冻结参考关联及评价协议。逐步计划、工具反馈、试验矢量、停用谱系、源码快照与输入哈希另存于运行目录。"]
    if (root/"execution_notes.json").exists():
        lines += ["", "耗时为共享工作站上的描述性结果；辅助1像素运行与末尾失败补跑存在重叠，不是隔离条件的延迟基准，详见execution_notes.json。"]
    if (root/"post_run_maintenance.json").exists():
        lines += ["", "本轮修复使用冻结v9.1源码。工作区v9.2仅修正多次停用后的回退谱系快照隔离并增加测试；本轮实际回退0次，保存输出未改动。维护记录见post_run_maintenance.json，复现本轮须使用source_snapshot。"]
    report=root/(("ORDINARY_RESULTS.md" if ordinary_selection else "SELECTED_METHOD_RESULTS.md") if args.methods else "RESULTS.md" if complete else "PAIRED_PARTIAL_RESULTS.md")
    report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(output/"completed.json",dict(paired_images=len(eligible),methods=methods,audits=len(audits),parent_complete=complete))
    print("REPORT",report,flush=True)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--split",default="val")
    p.add_argument("--allow-partial",action="store_true")
    p.add_argument("--methods",nargs="+",help="Explicit completed-method analysis; saved separately, with parent completion and failure coverage disclosed")
    evaluate(p.parse_args())
