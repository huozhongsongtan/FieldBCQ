"""CSV and Markdown layouts corresponding exactly to manuscript Tables 4--10."""

import csv
from pathlib import Path

from paper_metrics import aggregate

MODELS = ("Baseline", "A", "B", "C", "A+B", "A+C", "B+C", "FieldBCQ")
TITLES = {4: "消融实验分割结果", 5: "边界与结构指标", 6: "同权重关闭与开启质量评分",
          7: "固定阈值检出与数量面积结果", 8: "分来源结果", 9: "分辨率分组结果", 10: "模型复杂度与耗时"}
FIXED_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1", "boundary_f1": "Boundary F1",
                "merge_rate": "合并率", "oversegmentation_rate": "过分割率", "small_parcel_recall": "小地块 Recall",
                "score_iou_ece": "分数–IoU ECE", "parcel_count_mae_per_image": "逐图实例数量MAE／个",
                "matched_parcel_area_mape": "成功匹配实例面积MAPE／%", "foreground_area_wape": "全前景面积WAPE／%",
                "foreground_area_net_bias": "全前景面积净偏差／%"}


def scaled(value, key):
    return None if value is None else value * (1 if key in {"score_iou_ece", "parcel_count_mae_per_image"} else 100)


def table_rows(reports, records, timing, selected):
    tables = {}
    if 4 in selected:
        tables[4] = [{"配置": m, **{k.replace("_", "–"): v*100 for k, v in reports[m]["native"].items()}} for m in MODELS]
    if 5 in selected:
        keys = ("boundary_f1", "merge_rate", "oversegmentation_rate", "score_iou_ece")
        tables[5] = [{"配置": m, **{FIXED_LABELS[k]: scaled(reports[m]["fixed"][k], k) for k in keys}} for m in MODELS]
    if 6 in selected:
        if reports["FieldBCQ"]["checkpoint_sha256"] != reports["class"]["checkpoint_sha256"]:
            raise ValueError("Table 6 must compare the same checkpoint SHA256")
        rows = []
        for key in ("AP75", "AP50_95", "precision", "recall", "f1", "boundary_f1", "merge_rate",
                    "oversegmentation_rate", "small_parcel_recall", "score_iou_ece"):
            section = "native" if key.startswith("AP") else "fixed"
            c, j = [scaled(reports[m][section][key], key) for m in ("class", "FieldBCQ")]
            rows.append({"指标": FIXED_LABELS.get(key, key.replace("_", "–")), "关闭评分 class": c,
                         "联合评分 joint": j, "差值": None if c is None or j is None else j-c})
        tables[6] = rows
    if 7 in selected:
        tables[7] = [{"指标": FIXED_LABELS[k], **{m: scaled(reports[m]["fixed"][k], k) for m in ("Baseline", "FieldBCQ")}}
                     for k in ("precision", "recall", "f1", "boundary_f1", "small_parcel_recall",
                               "parcel_count_mae_per_image", "matched_parcel_area_mape", "foreground_area_wape", "foreground_area_net_bias")]
    for number, column in ((8, "source"), (9, "resolution_m")):
        if number not in selected:
            continue
        groups = reports["groups"][column]
        rows = []
        for group in sorted(groups, key=(float if number == 9 else str)):
            b, j = groups[group]["Baseline"], groups[group]["FieldBCQ"]
            if any(b["counts"][k] != j["counts"][k] for k in ("images", "gt_instances")):
                raise ValueError(f"Group count mismatch: {group}")
            row = {"来源" if number == 8 else "标称分辨率／m": group,
                   "影像数": b["counts"]["images"], "实例数": b["counts"]["gt_instances"],
                   "Baseline F1": scaled(b["f1"], "f1"), "FieldBCQ F1": scaled(j["f1"], "f1")}
            if number == 8:
                row.update({m+" Boundary F1": scaled(v["boundary_f1"], "boundary_f1") for m, v in (("Baseline", b), ("FieldBCQ", j))})
            else:
                row["差值／百分点"] = None if b["f1"] is None or j["f1"] is None else (j["f1"]-b["f1"])*100
            rows.append(row)
        tables[number] = rows
    if 10 in selected:
        tables[10] = [{"配置": m, "参数量 M": timing[m]["parameters_total"]/1e6,
                       "GFLOPs": timing[m]["gflops_per_image"], "原始forward ms": timing[m]["raw_forward_ms"],
                       "预测流程 ms": timing[m]["prediction_pipeline_ms"]} for m in ("Baseline", "FieldBCQ")]
    return tables


def grouped_results(reports, records, metadata, selected):
    reports["groups"] = {}
    for number, column in ((8, "source"), (9, "resolution_m")):
        if number not in selected:
            continue
        grouped = {}
        for model in ("Baseline", "FieldBCQ"):
            buckets = {}
            for image, record in records[model].items():
                keys = {"images", "gt_instances", "pred_instances", "tp"}
                if number == 8:
                    keys.update(("boundary_pred_pixels", "boundary_gt_pixels", "boundary_correct_pred", "boundary_correct_gt"))
                buckets.setdefault(metadata[image][column], []).append({k: record[k] for k in keys})
            for value, rows in buckets.items():
                grouped.setdefault(value, {})[model] = aggregate(rows)
        reports["groups"][column] = grouped


def write_tables(output, tables):
    output = Path(output)
    lines = ["# FieldBCQ 论文表4—10", "", "P、R、F1、AP、召回率、合并率、过分割率与面积相对误差均以百分数表示；ECE使用0—1尺度。差值为百分点，ECE差值保留原尺度；空值记为 —。", ""]
    for number, rows in sorted(tables.items()):
        if not rows:
            raise ValueError(f"Table {number} has no rows")
        columns = list(rows[0])
        with (output / f"table{number}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, columns)
            writer.writeheader()
            writer.writerows(rows)
        lines += [f"## 表{number} {TITLES[number]}", "", "| " + " | ".join(columns) + " |",
                  "| " + " | ".join("---" for _ in columns) + " |"]
        for row in rows:
            values = ["—" if v is None else f"{v:.5f}" if isinstance(v, float) else str(v) for v in row.values()]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    (output / "tables4_10.md").write_text("\n".join(lines), encoding="utf-8")
