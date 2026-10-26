"""Paper table schemas, model selection, scope and report consistency."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import val_paper
from paper_tables import MODELS, grouped_results, table_rows, write_tables
from paper_metrics import aggregate
from paper_validator import metric_scope
from test_paper_metrics import masks, measure


class MainValidationTests(unittest.TestCase):
    def test_lfs_pointer_is_rejected_before_dataset_scanning_or_model_loading(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            weight = Path(tmp) / 'abc/best.pt'
            weight.parent.mkdir()
            weight.write_text('version https://git-lfs.github.com/spec/v1\n'
                              'oid sha256:' + 'a'*64 + '\nsize 128644434\n', encoding='utf-8')
            for flags in ([], ['--check-only']):
                with self.subTest(flags=flags), patch.object(val_paper, 'prepare_dataset') as dataset:
                    with patch.object(val_paper, 'evaluate') as evaluation:
                        with self.assertRaisesRegex(ValueError, 'Git LFS pointer') as error:
                            val_paper.main(['--tables', '6', '--weights-root', tmp] + flags)
                    dataset.assert_not_called()
                    evaluation.assert_not_called()
                    self.assertIn('git lfs fetch', str(error.exception))

    def test_empty_missing_and_materialized_checkpoint_headers(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            weight = Path(tmp) / 'best.pt'
            with self.assertRaises(FileNotFoundError):
                val_paper.check_weight_file(weight)
            weight.touch()
            with self.assertRaisesRegex(ValueError, 'empty'):
                val_paper.check_weight_file(weight)
            for header in (b'PK\x03\x04archive fixture', b'\x80\x02legacy pickle fixture'):
                weight.write_bytes(header)
                val_paper.check_weight_file(weight)

    def test_default_paths_and_requested_models(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            for folder in val_paper.FOLDERS.values():
                (root / folder).mkdir()
                (root / folder / 'best.pt').write_bytes(b'fixture')
            args = val_paper.parse_args(['--weights-root', str(root)])
            self.assertEqual([m for m, _ in val_paper.models_to_run(args)], list(MODELS))
            args.tables = [6]
            self.assertEqual([m for m, _ in val_paper.models_to_run(args)], ['FieldBCQ'])
            args.tables = [10]
            self.assertEqual([m for m, _ in val_paper.models_to_run(args)], ['Baseline', 'FieldBCQ'])
            args.weights = ['B=' + str(root / 'b/best.pt')]
            with self.assertRaisesRegex(ValueError, 'Unused'):
                val_paper.models_to_run(args)

    def test_only_requested_metrics_for_each_model(self):
        scope, fixed, native = metric_scope([5], 'pcsc', None)
        self.assertEqual(scope, {'boundary', 'structure', 'calibration'})
        self.assertTrue(fixed)
        self.assertFalse(native)
        self.assertEqual(metric_scope([4], 'baseline', None), (set(), False, True))
        self.assertEqual(metric_scope([9], 'baseline', None), (set(), True, False))
        scope, _, _ = metric_scope(list(range(4, 11)), 'fieldbcq', 'class')
        self.assertNotIn('area', scope)
        self.assertNotIn('small', metric_scope([5], 'pcsc', None)[0])

    def test_all_seven_tables_have_exact_rows_units_and_same_weight_guard(self):
        native = dict(P=0.5, R=0.5, F1=0.5, AP50=0.5, AP75=0.4, AP50_95=0.3)
        square = masks((4, 4, 20, 20))
        fixed = aggregate([measure(square, square, [0.8])])
        reports = {m: dict(native=dict(native), fixed=dict(fixed), checkpoint_sha256=m) for m in MODELS}
        reports['class'] = dict(native=dict(native, AP75=0.39), fixed=dict(fixed), checkpoint_sha256='FieldBCQ')
        metadata, rows = {}, {}
        for i, resolution in enumerate(('0.3', '0.5', '1', '1.2', '2', '3', '10')):
            image = Path(f'{i}.png')
            metadata[image] = dict(source=('FBIS', 'FHAPD', 'FTW')[i % 3], resolution_m=resolution)
            rows[image] = measure(square, square, [0.8])
        records = {m: rows for m in ('Baseline', 'FieldBCQ')}
        selected = list(range(4, 11))
        grouped_results(reports, records, metadata, selected)
        timing = {m: dict(parameters_total=2e6, gflops_per_image=3, raw_forward_ms=4,
                          prediction_pipeline_ms=5) for m in ('Baseline', 'FieldBCQ')}
        tables = table_rows(reports, records, timing, selected)
        self.assertEqual([len(tables[n]) for n in selected], [8, 8, 10, 9, 3, 7, 2])
        self.assertEqual(tables[4][0]['P'], 50)
        self.assertAlmostEqual(tables[6][0]['差值'], 1)
        self.assertAlmostEqual(tables[5][0]['分数–IoU ECE'], 0.2)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            write_tables(tmp, tables)
            self.assertEqual({p.name for p in Path(tmp).iterdir()},
                             {f'table{n}.csv' for n in selected} | {'tables4_10.md'})
            with (Path(tmp)/'table8.csv').open(encoding='utf-8-sig') as fi:
                self.assertEqual(sum(int(r['影像数']) for r in csv.DictReader(fi)), 7)
        reports['class']['checkpoint_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'same checkpoint'):
            table_rows(reports, records, timing, selected)

    def test_legacy_unrelated_cli_switches_are_rejected(self):
        for flags in (['--quality-mode', 'all'], ['--precision', 'fp16'], ['--plots'], ['--tables', '11']):
            with self.subTest(flags=flags), patch('sys.stderr'), self.assertRaises(SystemExit):
                val_paper.parse_args(flags)
