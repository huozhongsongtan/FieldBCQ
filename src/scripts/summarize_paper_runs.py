"""Summarize reports by seed and protocol; compute baseline gains only for paired seeds."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from project_paths import project_path
from paper_runtime import write_json


def flat_values(report):
    values = {"mask_" + key: value for key, value in report["native"]["mask"].items()}
    values.update({"fixed_" + key: value for key, value in report["fixed"].items() if isinstance(value, (float, int))})
    efficiency = report.get("efficiency") or {}
    for key in ("parameters_total", "gflops_per_image"):
        if efficiency.get(key) is not None:
            values[key] = efficiency[key]
    # Batch/hardware differences must not be averaged as a single timing measurement.
    return values


def summarize(reports):
    groups, seen = {}, set()
    for report in reports:
        protocol = {"dataset": report["dataset_content_sha256"], "metadata": report.get("metadata_csv_sha256"),
                    "evaluation": report["evaluation_protocol"], "code": report["code"]["source_sha256"]}
        signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        config = report["configuration"]
        if config == "fieldbcq":
            config += "_c0" if report.get("quality_blend") == 0 else "_c2"
        seed = report.get("seed")
        identity = (signature, config, seed)
        if seed is None:
            identity += (report["checkpoint_sha256"],)
        if identity in seen:
            raise ValueError(f"Duplicate configuration/seed/protocol: {config} seed={seed}")
        seen.add(identity)
        groups.setdefault((signature, config), []).append(report)
    summary = []
    for (signature, config), records in groups.items():
        metrics = [flat_values(record) for record in records]
        result = {"configuration": config, "protocol_sha256": signature, "runs": len(records),
                  "seeds": [record.get("seed") for record in records], "metrics": {}, "paired_gain_over_baseline": {}}
        for name in sorted(set.intersection(*(set(row) for row in metrics))):
            values = [row[name] for row in metrics]
            result["metrics"][name] = {"mean": statistics.mean(values),
                                       "sample_std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}
        baseline = {row["seed"]: row for row in groups.get((signature, "baseline"), []) if row.get("seed") is not None}
        if config != "baseline":
            for metric in ("map50", "map75", "map50_95"):
                gains, seeds = [], []
                for row in records:
                    seed = row.get("seed")
                    if seed in baseline:
                        gains.append(100 * (row["native"]["mask"][metric] - baseline[seed]["native"]["mask"][metric]))
                        seeds.append(seed)
                if gains:
                    result["paired_gain_over_baseline"][metric] = {
                        "unit": "AP points", "paired_seeds": seeds, "n": len(gains),
                        "mean": statistics.mean(gains), "sample_std": statistics.stdev(gains) if len(gains) > 1 else None,
                    }
        summary.append(result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, help="Report files or directories searched recursively")
    parser.add_argument("--output", default="results/paper_summary")
    args = parser.parse_args()
    paths = set()
    for entry in args.reports:
        path = project_path(entry)
        paths.update(path.rglob("paper_report.json") if path.is_dir() else [path])
    if not paths:
        parser.error("No paper_report.json files found")
    result = summarize([json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)])
    base = project_path(args.output)
    write_json(base.with_suffix(".json"), result)
    with base.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration", "protocol_sha256", "metric", "n", "mean", "sample_std"])
        for group in result:
            for name, value in group["metrics"].items():
                writer.writerow([group["configuration"], group["protocol_sha256"], name, value["n"], value["mean"], value["sample_std"]])
    print(f"Saved {base.with_suffix('.json')} and {base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
