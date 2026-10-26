"""Compute only manuscript Tables 4--10 with the eight round-2 best.pt checkpoints."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from project_paths import project_path
from paper_data import load_metadata, resolve_images
from paper_runtime import benchmark, code_provenance, environment, sha256, write_json
from paper_tables import MODELS, grouped_results, table_rows, write_tables
from paper_validator import evaluate

FOLDERS = dict(zip(MODELS, ('baseline', 'a', 'b', 'c', 'ab', 'ac', 'bc', 'abc')))
CONFIGURATIONS = dict(zip(MODELS, ('baseline', 'pcsc', 'baps', 'rbqs', 'baps_pcsc', 'bds_rbqs', 'baps_rbqs', 'fieldbcq')))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-root', default='model')
    parser.add_argument('--weights', action='append', default=[], metavar='MODEL=PATH', help='Override one named checkpoint')
    parser.add_argument('--data', default='data/paper_test.yaml')
    parser.add_argument('--metadata-csv', default='data/paper_test_metadata.csv')
    parser.add_argument('--project', default='results/tables4_10', help='New output directory; existing directories are rejected')
    parser.add_argument('--device', default='0', help='CUDA index, or cpu for a separately labelled CPU run')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--tables', nargs='+', type=int, choices=range(4, 11), default=list(range(4, 11)))
    parser.add_argument('--check-only', action='store_true', help='Reject missing/empty/LFS-pointer weights; verify metadata and image/label hashes without inference')
    args = parser.parse_args(argv)
    if args.imgsz <= 0 or args.imgsz % 32 or args.batch < 1 or args.workers < 0:
        parser.error('imgsz must be a positive multiple of 32; batch positive; workers nonnegative')
    args.tables = sorted(set(args.tables))
    return args


def check_weight_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('rb') as handle:
        header = handle.read(256)
    if not header:
        raise ValueError(f'Checkpoint is empty: {path}; download or copy the complete best.pt')
    if header.lstrip().startswith(b'version https://git-lfs.github.com/spec/'):
        raise ValueError(
            f'Checkpoint is a Git LFS pointer, not downloaded model weights: {path}\n'
            'Run these commands from the repository root, then rerun --check-only:\n'
            '  git lfs install --local\n'
            '  git lfs fetch origin main --include="model/**/*.pt" --exclude=""\n'
            '  git lfs checkout\n'
            'If LFS objects are unavailable, copy the original round-2 best.pt files into model/.')


def models_to_run(args):
    required = set(MODELS) if set(args.tables) & {4, 5} else set()
    if set(args.tables) & {7, 8, 9, 10}:
        required.update(('Baseline', 'FieldBCQ'))
    if 6 in args.tables:
        required.add('FieldBCQ')
    paths = {m: project_path(args.weights_root) / FOLDERS[m] / 'best.pt' for m in required}
    overrides = set()
    for value in args.weights:
        model, separator, path = value.partition('=')
        if not separator or model not in MODELS or not path or model in overrides:
            raise ValueError('--weights requires a unique MODEL=PATH; MODEL is one of ' + ', '.join(MODELS))
        if model not in required:
            raise ValueError(f'Unused checkpoint override for selected tables: {model}')
        overrides.add(model)
        paths[model] = project_path(path)
    for path in paths.values():
        check_weight_file(path)
    return [(m, paths[m]) for m in MODELS if m in required]


def prepare_dataset(args):
    from ultralytics.data.utils import img2label_paths
    data_path, config, root, images = resolve_images(args.data)
    labels = {p: Path(label).resolve() for p, label in zip(images, img2label_paths([str(p) for p in images]))}
    metadata = load_metadata(args.metadata_csv, root, images) if set(args.tables) & {8, 9} else {}
    manifest = []
    for image in images:
        label = labels[image]
        if not label.is_file():
            raise FileNotFoundError(f'Missing label: {label}; negatives need an explicit empty file')
        try:
            image_id = image.relative_to(root).as_posix()
        except ValueError:
            image_id = str(image)
        row = dict(image=image_id, image_sha256=sha256(image), label_sha256=sha256(label))
        meta = metadata.get(image, {})
        for key in ('image_sha256', 'label_sha256'):
            if meta.get(key) and meta[key] != row[key]:
                raise ValueError(f'Frozen test manifest hash mismatch: {key}: {image}')
        if meta.get('instances'):
            count = sum(bool(line.strip()) for line in label.read_text(encoding='utf-8-sig').splitlines())
            if count != int(meta['instances']):
                raise ValueError(f'GT count differs from manifest: {label}')
        manifest.append(row)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return data_path, config, images, labels, metadata, manifest, digest


def main(argv=None):
    args = parse_args(argv)
    models = models_to_run(args)
    data_path, config, images, labels, metadata, manifest, digest = prepare_dataset(args)
    if args.check_only:
        print(f'Checked {len(models)} checkpoints, {len(images)} test images and labels; tables {args.tables}')
        for name, weights in models:
            print(f'{name}: {weights}')
        return
    output = project_path(args.project)
    if output.exists():
        raise FileExistsError(f'Choose a new --project directory: {output}')
    output.mkdir(parents=True)
    provenance = dict(version='fieldbcq-tables4-10-v2', status='running', arguments=vars(args),
                      environment=environment(), code=code_provenance(), data_yaml_sha256=sha256(data_path),
                      dataset_content_sha256=digest, dataset_manifest=manifest,
                      metadata_sha256=sha256(project_path(args.metadata_csv)) if metadata else None,
                      protocol=dict(precision='fp32', rect=True, augment=False, native_conf=0.001, fixed_conf=0.25,
                                    nms_iou=0.70, max_det=710, match_iou=0.5, boundary_tolerance_px=2,
                                    small_area_px=1024, calibration_bins=10,
                                    fixed_grid='square letterbox; full-resolution binary masks; integer GT polygons',
                                    boundary='two-sided 3x3 morphology, square tolerance; unmatched boundaries in denominators',
                                    matching='descending score; highest available same-class mask IoU >= 0.5',
                                    ece='all retained predictions; score vs best same-class mask IoU; 10 equal-width bins',
                                    area='matched MAPE; union WAPE and signed bias; count MAE includes negatives'), runs=[])
    reports, records, timing = {}, {}, {}
    write_json(output / 'provenance.json', provenance)
    try:
        # Exact normalized image list also prevents implicit YAML-root relocation by Ultralytics.
        with tempfile.TemporaryDirectory(prefix='paper_', dir=output) as work:
            import yaml
            work = Path(work)
            listing = work / 'images.txt'
            listing.write_text('\n'.join(str(p) for p in images) + '\n', encoding='utf-8')
            normalized = work / 'data.yaml'
            normalized.write_text(yaml.safe_dump(dict(path=str(work), train=str(listing), val=str(listing),
                                                       test=str(listing), names=config.get('names', ['field'])),
                                                   allow_unicode=True), encoding='utf-8')
            if set(args.tables) - {10}:
                for name, weights in models:
                    print(f'Evaluating {name}', flush=True)
                    mode = 'joint' if 'C' in name or name == 'FieldBCQ' else None
                    report, rows = evaluate(weights, CONFIGURATIONS[name], args, normalized, labels, work, mode)
                    reports[name], records[name] = report, rows
                    provenance['runs'].append({k: v for k, v in report.items() if k not in ('native', 'fixed')})
                    write_json(output / 'provenance.json', provenance)
                if 6 in args.tables:
                    print('Evaluating FieldBCQ class (same checkpoint)', flush=True)
                    weights = dict(models)['FieldBCQ']
                    report, rows = evaluate(weights, 'fieldbcq', args, normalized, labels, work, 'class')
                    reports['class'], records['class'] = report, rows
                    provenance['runs'].append({k: v for k, v in report.items() if k not in ('native', 'fixed')})
                grouped_results(reports, records, metadata, args.tables)
            if 10 in args.tables:
                import cv2
                image = cv2.imread(str(images[0]))
                if image is None:
                    raise ValueError(f'Cannot decode benchmark image: {images[0]}')
                provenance['benchmark_image'] = manifest[0]
                for name in ('Baseline', 'FieldBCQ'):
                    print(f'Benchmarking {name}: FP32, batch=1, warmup=50, iterations=200', flush=True)
                    weights = dict(models)[name]
                    timing[name] = benchmark(weights, imgsz=args.imgsz, device=args.device, image=image,
                                             quality_mode='joint' if name == 'FieldBCQ' else None,
                                             expected=CONFIGURATIONS[name])
                    timing[name]['checkpoint_sha256'] = sha256(weights)
                provenance['benchmark'] = timing
            write_tables(output, table_rows(reports, records, timing, args.tables))
        provenance['status'] = 'complete'
    except Exception as exc:
        provenance.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output / 'provenance.json', provenance)
    print(f'Saved tables {args.tables} to {output}', flush=True)


if __name__ == '__main__':
    main()
