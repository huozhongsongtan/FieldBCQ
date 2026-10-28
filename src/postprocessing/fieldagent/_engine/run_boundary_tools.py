"""Run each expanded tool independently on frozen prediction scenes, no GT/GPT.

Default parameters are a tool smoke baseline, not validation-selected optima.
Every method starts from the same raw vectors; instances are attempted once in
input order, with current-neighbour safety gates. No hidden optimizing finish.
"""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from boundary_executor import BoundaryExecutor, BoundaryConfig
from boundary_tools import REGISTRY, VERSION, catalog
from cvr_agent import dump, sha, read_items, HOST


def load_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read image: '+str(path))
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tools', nargs='+', choices=list(REGISTRY), default=[k for k, v in REGISTRY.items() if v.task != 'legalize'])
    p.add_argument('--scenes', nargs='+', help='Optional exact image stems')
    p.add_argument('--budget', type=int, default=128)
    p.add_argument('--seconds', type=float, default=120.)
    p.add_argument('--max-targets', type=int, help='Optional smoke cap per scene/tool')
    args = p.parse_args()
    if len(set(args.tools)) != len(args.tools):
        p.error('Repeated tools')
    if args.max_targets is not None and args.max_targets < 1:
        p.error('--max-targets must be positive')
    root = args.data.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = BoundaryConfig(budget=args.budget, seconds=args.seconds)
    manifest_path = root/'postprocess_inputs.json'
    entries = json.loads(manifest_path.read_text(encoding='utf-8'))['inputs']
    if args.scenes:
        known = {Path(r['image']).stem for r in entries}
        if set(args.scenes)-known:
            raise ValueError('Unknown scenes')
        entries = [r for r in entries if Path(r['image']).stem in args.scenes]
    input_hashes = {str((root/v).resolve()): sha(root/v) for row in entries for v in row.values()}
    input_hashes[str(manifest_path)] = sha(manifest_path)
    source_hashes = {}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if not path or not str(path).endswith('.py'):
            continue
        path = Path(path).resolve()
        if path.parent not in (Path(__file__).parent.resolve(), HOST.resolve()):
            continue
        relative = Path('host' if path.parent == HOST.resolve() else 'cvr')/path.name
        destination = out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source_hashes[str(path)] = dict(sha256=sha(path), snapshot=str(Path('source_snapshot')/relative))
    dump(out/'manifest.json', dict(version=VERSION, runtime_gt_access=False, model_calls=0,
        data_root=str(root), entries=entries, methods=args.tools, config=asdict(config),
        max_targets=args.max_targets, input_sha256=input_hashes, sources=source_hashes,
        catalog=catalog(), comparison='Each tool independently from raw; one default-parameter attempt per target in input order; shared gates, no hidden finish',
        limitation='Development smoke baseline; no parameter tuning, independent test or GPT efficacy claim'))
    rows = []; errors = []; began = time.monotonic()
    for entry in entries:
        scene = Path(entry['image']).stem
        image = load_image(root/entry['image']); items = read_items(root/entry['prediction_vector'])
        targets = items[:args.max_targets] if args.max_targets else items
        for tool in args.tools:
            folder = out/'runs'/tool/scene
            try:
                executor = BoundaryExecutor(image, items, folder, config)
                stop = 'one_pass_complete'
                for target in targets:
                    if len(executor.history) >= config.budget or time.monotonic()-executor.started >= config.seconds:
                        stop = 'budget_exhausted'; break
                    executor.attempt(tool, [target.identifier], expected_state=executor.state_id, source='single_tool_default_probe')
                trace = executor.finish(stop)
                reasons = Counter(r for h in trace['history'] for r in h['verdict']['reasons'])
                row = dict(scene=scene, method=tool, vertices_before=trace['baseline']['vertices'],
                    vertices=trace['final']['vertices'], reduction=1-trace['final']['vertices']/max(1, trace['baseline']['vertices']),
                    overlap_before=trace['baseline']['overlap_excess_area'], overlap=trace['final']['overlap_excess_area'],
                    invalid=trace['final']['invalid_geometries'], candidates=trace['candidates'], commits=trace['commits'],
                    tool_errors=reasons.get('tool_error', 0), seconds=trace['seconds'], stop=stop)
                rows.append(row)
                dump(folder/'completed.json', row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            except Exception as error:
                record = dict(scene=scene, tool=tool, error=f'{type(error).__name__}: {error}')
                errors.append(record); dump(folder/'execution_error.json', record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    unchanged = all(sha(path) == digest for path, digest in input_hashes.items())
    if rows:
        with (out/'per_scene.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(out/'completed.json', dict(runs=len(rows), expected=len(entries)*len(args.tools), errors=errors,
        input_hashes_unchanged=unchanged, model_calls=0, runtime_gt_access=False,
        seconds=time.monotonic()-began, efficacy_evaluated=False))
    if errors or not unchanged:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
