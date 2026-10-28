"""Offline v5 evaluation. Reference labels are read here, never by repair code."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np
from shapely.geometry import Polygon

from evaluate_agent_experiments import instances, measure, summary as legacy_summary, write_csv
from evaluate_geometry_boundary_iou import inner_band, matches, measure_boundary, summarize as boundary_summary
from evaluate_mechanism_metrics import instance_boundary_union, pairwise_overlap, read_yolo_masks
from fieldbqc_geometry import geometry_mask, normalized, polygons, scene_metrics, vertices
from run_agent_experiments import fingerprint, load_parcels, read_json, write_json
from run_geometry_experiments import vector_input


def reference_vectors(path, shape):
    """Retain original annotation vertices; do not simplify the reference."""
    result, counts = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        coords = np.asarray([float(x) for x in fields[1:]], float).reshape(-1, 2) * [shape[1], shape[0]]
        mask = np.zeros(shape, np.uint8)
        quantized = np.clip(np.rint(coords), [0,0], [shape[1]-1,shape[0]-1]).astype(np.int32)
        cv2.fillPoly(mask, [quantized], 1)
        if mask.any():
            geom = Polygon(coords)
            result.append(normalized(geom))
            counts.append(len(coords) - int(np.array_equal(coords[0], coords[-1])))
    return result, counts


def boundary_samples(geometry, step=2.):
    points, directions, starts, ends = [], [], [], []
    for part in polygons(geometry):
        for ring in [part.exterior] + list(part.interiors):
            coords = np.asarray(ring.coords, float)
            for a, b in zip(coords[:-1], coords[1:]):
                length = np.linalg.norm(b-a)
                if length < 1e-12:
                    continue
                count = max(1, int(np.ceil(length / step)))
                fractions = (np.arange(count) + .5) / count
                points.extend(a + fractions[:,None] * (b-a))
                directions.extend([((b-a) / length)] * count)
                starts.append(a)
                ends.append(b)
    return (np.asarray(points).reshape(-1,2), np.asarray(directions).reshape(-1,2),
            np.asarray(starts).reshape(-1,2), np.asarray(ends).reshape(-1,2))


def directed_distance(points, tangents, target):
    starts, ends = target[2], target[3]
    lines = ends - starts
    lengths2 = np.einsum("ij,ij->i", lines, lines)
    distances, angles = [], []
    for offset in range(0, len(points), 256):
        p = points[offset:offset+256]
        displacement = p[:,None,:] - starts[None,:,:]
        t = np.clip(np.einsum("ijk,jk->ij", displacement, lines) / lengths2, 0, 1)
        residual = displacement - t[:,:,None] * lines[None,:,:]
        d2 = np.einsum("ijk,ijk->ij", residual, residual)
        nearest = d2.argmin(axis=1)  # deterministic ring/edge order breaks ties
        distances.extend(np.sqrt(d2[np.arange(len(p)),nearest]))
        directions = lines[nearest] / np.sqrt(lengths2[nearest,None])
        cosine = np.abs(np.einsum("ij,ij->i", tangents[offset:offset+256], directions))
        angles.extend(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
    return np.asarray(distances), np.asarray(angles)


def shape_metrics(prediction, reference, prediction_n, reference_n, iou):
    p, g = boundary_samples(prediction), boundary_samples(reference)
    if not len(p[0]) or not len(g[0]):
        return None
    pd, pa = directed_distance(p[0], p[1], g)
    gd, _ = directed_distance(g[0], g[1], p)
    return dict(polis_sampled_2px=float((pd.mean()+gd.mean())*.5),
                mta_sampled_2px_degrees=float(pa.max()),
                symmetric_distance_p95=float(np.percentile(np.r_[pd,gd], 95)),
                n_ratio=prediction_n/reference_n,
                complexity_iou=float(iou * (1-abs(prediction_n-reference_n)/(prediction_n+reference_n))))


def band_counts(gt, predicted, shape, width=2):
    a = instance_boundary_union(gt.masks) if gt.masks else np.zeros(shape, bool)
    b = instance_boundary_union(predicted.masks) if predicted.masks else np.zeros(shape, bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*width+1, 2*width+1))
    a = cv2.dilate(a.astype(np.uint8), kernel).astype(bool)
    b = cv2.dilate(b.astype(np.uint8), kernel).astype(bool)
    return dict(band_intersection=int(np.count_nonzero(a & b)), band_pred=int(b.sum()), band_gt=int(a.sum()))


def summarize_bands(counts):
    i, p, g = counts["band_intersection"], counts["band_pred"], counts["band_gt"]
    return dict(boundary_band_precision=i/p if p else None, boundary_band_recall=i/g if g else None,
                boundary_band_f1=2*i/(p+g) if p+g else None,
                boundary_band_iou=i/(p+g-i) if p+g-i else None)


def evaluate(args):
    root = args.runs
    if not (root / "completed.json").exists():
        raise ValueError("Incomplete run must not be evaluated as a completed experiment")
    manifest = read_json(root / "manifest.json")
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    rows, summaries, pair_rows, annotation_template, audits = [], [], [], [], []
    gt_cache = {}
    for entry in manifest["inputs"]:
        stem = entry["image_id"]
        before = vector_input(root / "initial" / (stem + ".pixel.geojson"))
        for i, p in enumerate(before):
            for q in before[i+1:]:
                intersection = normalized(p.geometry).intersection(normalized(q.geometry))
                if intersection.area > 1e-8:
                    annotation_template.append(dict(image_id=stem, roi_id=f"overlap_{p.identifier}_{q.identifier}",
                        instance_ids=[p.identifier,q.identifier], issue_type="overlap", initial_geometry=intersection.__geo_interface__,
                        initial_status="unreviewed", judgeable=None, reference_geometry=None, reviewer=None))
            for j, part in enumerate(polygons(p.geometry)):
                for k, ring in enumerate(part.interiors):
                    annotation_template.append(dict(image_id=stem, roi_id=f"hole_{p.identifier}_{j}_{k}",
                        instance_ids=[p.identifier], issue_type="hole_candidate", initial_geometry=Polygon(ring).__geo_interface__,
                        initial_status="unreviewed", judgeable=None, reference_geometry=None, reviewer=None))
        # Initial complexity flags are not error labels. Include normal-control
        # proposals before viewing any method output, requiring human confirmation.
        for p in before[:2]:
            annotation_template.append(dict(image_id=stem, roi_id=f"control_{p.identifier}", instance_ids=[p.identifier],
                issue_type="normal_control_candidate", initial_geometry=p.geometry.__geo_interface__,
                initial_status="unreviewed", judgeable=None, reference_geometry=None, reviewer=None))
    for method in manifest["methods"]:
        method_rows, totals, btotal, band_total = [], None, None, None
        threshold_counts = {t: Counter() for t in (.5,.75,.9)}
        all_shape = []
        for entry in manifest["inputs"]:
            stem, shape = entry["image_id"], tuple(entry["shape"])
            label = args.dataset_root / "labels" / args.split / (stem + ".txt")
            if not label.exists():
                raise FileNotFoundError(label)
            if stem not in gt_cache:
                gt_cache[stem] = (read_yolo_masks(label,*shape), *reference_vectors(label,shape))
            gt, reference, reference_counts = gt_cache[stem]
            if len(reference) != len(gt.masks):
                raise ValueError("Reference vector/mask instance ordering mismatch")
            parcels, actual_shape = load_parcels(root / method / "masks" / (stem + ".npz"))
            predicted = instances(parcels)
            vector = vector_input(root / method / "vectors" / (stem + ".pixel.geojson"))
            initial = vector_input(root / "initial" / (stem + ".pixel.geojson"))
            trace = read_json(root / method / "traces" / (stem + ".json"))
            for kind,suffix in (("masks",".npz"),("vectors",".pixel.geojson")):
                if fingerprint(root/method/kind/(stem+suffix)) != trace["output_sha256"][kind]:
                    raise ValueError("Saved output differs from repair-time hash")
            if trace["planner_failed"]:
                raise ValueError("Failed planner group cannot support a completed comparison")
            if actual_shape != shape or [p.parcel_id for p in parcels] != [p.identifier for p in vector]:
                raise ValueError("Output identity or shape mismatch")
            roundtrip_ok = all(np.array_equal(p.mask,geometry_mask(v.geometry,shape)) for p,v in zip(parcels,vector))
            identity_ok = [(p.identifier,p.cls,p.source_ids) for p in initial] == [(p.identifier,p.cls,p.source_ids) for p in vector]
            if not roundtrip_ok or not identity_ok:
                raise ValueError("Raster/vector consistency or instance preservation failed")
            audits.append(dict(method=method,image_id=stem,roundtrip_ok=roundtrip_ok,identity_ok=identity_ok))
            result, counts = measure(gt,predicted,shape)
            totals = Counter(counts) if totals is None else totals + Counter(counts)
            boundary = measure_boundary(gt,predicted)
            btotal = Counter(boundary) if btotal is None else btotal + Counter(boundary)
            bands = band_counts(gt,predicted,shape)
            band_total = Counter(bands) if band_total is None else band_total + Counter(bands)
            iou,_,_ = pairwise_overlap(gt.masks,predicted.masks,gt.classes,predicted.classes)
            threshold_results = {}
            for threshold in threshold_counts:
                pairs = matches(iou,threshold)
                threshold_counts[threshold].update(tp=len(pairs),fp=len(parcels)-len(pairs),fn=len(gt.masks)-len(pairs))
                threshold_results[f"f1_{int(threshold*100)}"] = 2*len(pairs)/max(1,len(parcels)+len(gt.masks))
            geom_values = []
            for r,c in matches(iou,.5):
                metrics = shape_metrics(vector[c].geometry,reference[r],vertices(vector[c].geometry),reference_counts[r],iou[r,c])
                if metrics:
                    geom_values.append(metrics)
                    pair_rows.append(dict(method=method,image_id=stem,gt_index=r,prediction_id=vector[c].identifier,
                                          predicted_vertices=vertices(vector[c].geometry),reference_vertices=reference_counts[r],**metrics))
            all_shape.extend(geom_values)
            means = {key:float(np.mean([v[key] for v in geom_values])) for key in geom_values[0]} if geom_values else {}
            occupied = sum((p.mask.astype(np.int32) for p in parcels), np.zeros(shape,np.int32))
            boundary_occupancy = sum((inner_band(p.mask,1).astype(np.int32) for p in parcels),np.zeros(shape,np.int32))
            excess = np.maximum(occupied-1,0)
            ir = float(excess[boundary_occupancy<2].sum()/max(1,np.count_nonzero(occupied)))
            actions = [e for e in trace["action_trace"] if "accepted" in e]
            plans = [e for e in trace["action_trace"] if e["action"] == "plan"]
            billings = [e.get("billed_usage",{}) for e in plans]
            row = dict(method=method,image_id=stem,**threshold_results,pq_like=result["pq"],
                       tolerant_boundary_f1_2px=result["boundary_f1"],union_iou=result["union_iou"],
                       **boundary_summary(boundary),**summarize_bands(bands),**means,
                       matched_pairs=len(geom_values),reference_instances=len(gt.masks),
                       vertices_before=trace["initial"]["vertices"],vertices_after=trace["final"]["vertices"],
                       overlap_before=trace["initial"]["overlap_excess_area"],overlap_after=trace["final"]["overlap_excess_area"],
                       excess_over_initial_union=trace["final"]["overlap_excess_area"]/max(1,trace["initial"]["union_area"]),
                       ir_pixel_adapted=ir,invalid_after=trace["final"]["invalid_geometries"],
                       holes_before=trace["initial"]["holes"],holes_after=trace["final"]["holes"],
                       accepted=sum(e["accepted"] for e in actions),attempts=len(actions),
                       api_calls=sum(e.get("api_attempts",int(e.get("api_request",False))) for e in plans),
                       input_tokens=sum(u.get("input_tokens",u.get("prompt_tokens",0)) for u in billings),
                       output_tokens=sum(u.get("output_tokens",u.get("completion_tokens",0)) for u in billings),
                       candidate_evaluations=trace["candidate_evaluations"],seconds=trace["postprocess_seconds"],
                       reviewed_instances=len(trace["review_ids"]))
            method_rows.append(row)
            rows.append(row)
        base = legacy_summary(totals)
        n0, nf = sum(r["vertices_before"] for r in method_rows),sum(r["vertices_after"] for r in method_rows)
        final = dict(method=method,images=len(method_rows),predicted_instances=base["pred_count"],
                     pq_like=base["pq"],tolerant_boundary_f1_2px=base["boundary_f1"],union_iou=base["union_iou"],
                     vertices_before=n0,vertices_after=nf,compression=1-nf/max(1,n0),
                     **boundary_summary(btotal),**summarize_bands(band_total))
        f1s, ps, rs = [],[],[]
        for t,c in threshold_counts.items():
            precision = c["tp"]/max(1,c["tp"]+c["fp"])
            recall = c["tp"]/max(1,c["tp"]+c["fn"])
            f1 = 2*c["tp"]/max(1,2*c["tp"]+c["fp"]+c["fn"])
            final.update({f"precision_{int(t*100)}":precision,f"recall_{int(t*100)}":recall,f"f1_{int(t*100)}":f1})
            f1s.append(f1);ps.append(precision);rs.append(recall)
        final.update(mean_f1=float(np.mean(f1s)),mean_precision=float(np.mean(ps)),mean_recall=float(np.mean(rs)),
                     matched_pairs=len(all_shape),reference_instances=sum(r["reference_instances"] for r in method_rows))
        if all_shape:
            final.update({key:float(np.mean([v[key] for v in all_shape])) for key in all_shape[0]})
        for key in ("overlap_before","overlap_after","invalid_after","holes_before","holes_after","accepted","attempts",
                    "api_calls","input_tokens","output_tokens","candidate_evaluations","reviewed_instances"):
            final[key] = sum(r[key] for r in method_rows)
        final["mean_seconds"] = float(np.mean([r["seconds"] for r in method_rows]))
        final["ir_pixel_adapted_image_mean"] = float(np.mean([r["ir_pixel_adapted"] for r in method_rows]))
        counts = [vertices(p.geometry) for entry in manifest["inputs"] for p in vector_input(root/method/"vectors"/(entry["image_id"]+".pixel.geojson"))]
        final.update(mean_vertices=float(np.mean(counts)) if counts else None,
                     median_vertices=float(np.median(counts)) if counts else None,
                     p90_vertices=float(np.percentile(counts,90)) if counts else None,
                     semantic_correct_repair_recall=None,normal_region_damage_rate=None,true_hole_false_fill_rate=None,
                     shared_edge_recall=None)
        summaries.append(final)
        print(f"[evaluation] {method}: compression={final['compression']:.2%} BF1={base['boundary_f1']:.6f} meanF1={final['mean_f1']:.6f}",flush=True)
    comparisons = []
    baseline = {r["image_id"]:r for r in rows if r["method"]=="G0"}
    rng = np.random.default_rng(20261020)
    for left,right in (("G1-RULE","G0"),("G2-ONCE","G1-RULE"),("G2-LOOP","G1-RULE"),("G2-LOOP","G2-ONCE")):
        a = {r["image_id"]:r for r in rows if r["method"]==left}
        b = {r["image_id"]:r for r in rows if r["method"]==right}
        if not a or not b:
            continue
        for key in ("tolerant_boundary_f1_2px","pq_like","f1_50","vertices_after","overlap_after"):
            delta = np.array([a[k][key]-b[k][key] for k in sorted(a)])
            boot = rng.choice(delta,size=(2000,len(delta)),replace=True).mean(axis=1)
            comparisons.append(dict(comparison=left+" minus "+right,metric=key,mean_delta=float(delta.mean()),
                                    ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),
                                    unit="image; exploratory, scene independence unverified"))
    write_csv(out/"summary.csv",summaries)
    write_json(out/"summary.json",summaries)
    write_csv(out/"per_image.csv",rows)
    write_csv(out/"matched_geometry.csv",pair_rows)
    write_csv(out/"paired_comparisons.csv",comparisons)
    write_csv(out/"output_audit.csv",audits)
    write_json(out/"roi_annotation_template.json",dict(status="UNREVIEWED; not ground truth",regions=annotation_template))
    reference_counts = [n for _,_,counts in gt_cache.values() for n in counts]
    write_json(out/"reference_vertex_audit.json",dict(instances=len(reference_counts),
        total=sum(reference_counts),median=float(np.median(reference_counts)) if reference_counts else None,
        p90=float(np.percentile(reference_counts,90)) if reference_counts else None,
        vertex_semantics="Original annotation vertices, no simplification or densification. Key-corner quality is not certified.",
        holes_supported=False))
    write_json(out/"protocol.json",dict(evaluator_sha256=fingerprint(Path(__file__)),split=args.split,
        labels={entry["image_id"]:fingerprint(args.dataset_root/"labels"/args.split/(entry["image_id"]+".txt")) for entry in manifest["inputs"]},
        geometry_matching="class-aware, max valid cardinality then max IoU >=0.5; rematched per method",
        instance_thresholds=[.5,.75,.9],band_width_pixels=2,
        boundary_band="union of per-instance boundaries, dilated by radius-2 elliptical kernel; pixel adaptation, not 1m protocol",
        inner_boundary="padded 3x3 erosion, 2 iterations; conditional pair BIoU accompanied by instance recall",
        polis="mean of two directed sampled-point-to-full-segment means; <=2px equal-arclength midpoints per edge; includes holes",
        mta="max predicted sample tangent versus nearest reference segment tangent; orientation-free [0,90] degrees; first segment breaks ties",
        geometry_aggregation="pair-weighted mean; per-image metrics also saved; conditional on IoU>=0.5 matching",
        vertex_count="saved vector exterior + interior vertices, excluding closing duplicate; original YOLO annotation vertices for references",
        reference_limit="YOLO single-ring labels; reference vertex density not independently certified. N-ratio/C-IoU exploratory only. Raw floating annotation geometry and cv2-rasterized GT are separate representations.",
        ir="pixel adaptation: sum(max(occupancy-1,0)) outside pixels with >=2 one-pixel internal boundary bands / current occupied area",
        unavailable="SER and semantic defect/normal-control/hole correctness require independent reference; no invented values",
        hypothesis_testing="image-bootstrap descriptive only; no independent test or scene-level significance claim"))
    lines = ["# 四工具选择性修复：本地开发实验", "", "同一G0独立启动，无固定十点目标，无末尾共同减点。现有开发图复用；不作为独立测试。", "",
             "| 方法 | 顶点数 | 压缩率 | 多阈值平均F1 | 容差BF1@2px | PQ-like | 重叠超额面积 | API请求 | 秒/图 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['method']} | {s['vertices_after']} | {s['compression']:.2%} | {s['mean_f1']:.6f} | {s['tolerant_boundary_f1_2px']:.6f} | {s['pq_like']:.6f} | {s['overlap_after']:.2f} | {s['api_calls']} | {s['mean_seconds']:.2f} |")
    lines += ["", "程序验收通过不等于语义修复正确。孔洞误填、正常区域损伤、重叠归属与SER暂不可评价；固定初始ROI候选已输出，需独立人工标注。",
              "", "线段规整为借鉴线段表达思想的确定性工具，并非B2PNet网络或完整复现。PoLiS/MTA采用已披露采样适配；N-ratio与C-IoU仅作探索性对照。",
              "", "原始矢量、最终矢量、逐步候选与验收、API原始回复/用量、代码快照和输入哈希均已保存。费用无可靠单价，报告真实token而不猜测金额。"]
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(out/"completed.json",dict(images=len(manifest["inputs"]),methods=manifest["methods"],audits=len(audits)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--split",default="val")
    evaluate(parser.parse_args())
