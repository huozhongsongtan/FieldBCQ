"""Verify scoring equations, unchanged geometry, rect grids and real evaluation propagation."""

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

import collect_paper_metrics as collector
import val_paper as validation
from quality_scoring import QUALITY_MODES, set_quality_scoring
from summarize_paper_runs import summarize
from test_quality_ablation import small_model
from ultralytics.nn.modules.head import BASMQCSegment, Detect


class QualityScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_four_modes_change_only_scores_and_keep_historical_default(self):
        head = BASMQCSegment(nc=2, nm=8, npr=16, ch=(16, 32, 64)).eval()
        head.stride = torch.tensor([8., 16., 32.])
        for branch in head.cv5:
            branch[-1].bias.data.copy_(torch.tensor([-1., 2.]))
        model = type("Container", (), {"model": [head]})()
        features = [torch.randn(1, c, s, s) for c, s in ((16, 8), (32, 4), (64, 2))]
        weights = copy.deepcopy(head.state_dict())
        with torch.no_grad():
            raw = Detect.forward(head, list(features))[0]
            historical = head(list(features))
            qr, qb = torch.sigmoid(torch.tensor([-1., 2.])).tolist()
            factors = {"class": 1., "region": qr, "boundary": qb, "joint": math.sqrt(qr * qb)}
            for mode in QUALITY_MODES:
                state = set_quality_scoring(model, mode)
                output, auxiliary = head(list(features))
                torch.testing.assert_close(output[:, 4:6], raw[:, 4:6] * factors[mode])
                torch.testing.assert_close(output[:, :4], raw[:, :4], rtol=0, atol=0)
                torch.testing.assert_close(output[:, 6:], historical[0][:, 6:], rtol=0, atol=0)
                torch.testing.assert_close(auxiliary[2], historical[1][2], rtol=0, atol=0)
                self.assertEqual(state["quality_mode"], mode)
            torch.testing.assert_close(output, historical[0], rtol=0, atol=0)
            for blend in (0, 1):
                state = set_quality_scoring(model, blend=blend)
                expected = raw if blend == 0 else historical[0][:, :6]
                torch.testing.assert_close(head(list(features))[0][:, :6], expected)
                self.assertEqual(state["quality_mode"], "class" if blend == 0 else "joint")
            for key, value in head.state_dict().items():
                torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "quality_mode"):
            set_quality_scoring(model, "invalid")
        with self.assertRaises(ValueError):
            set_quality_scoring(model, "region", 1)
        plain = type("Container", (), {"model": [object()]})()
        self.assertIsNone(set_quality_scoring(plain)["quality_mode"])
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            set_quality_scoring(plain, "class")

    def test_rectangular_masks_align_with_square_gt_without_resampling(self):
        from ultralytics.data.augment import LetterBox
        for shape in ((31, 64), (64, 31), (29, 47), (47, 29), (64, 64)):
            with self.subTest(shape=shape):
                image = np.zeros((*shape, 3), dtype=np.uint8)
                image[5:20, 5:20] = 255
                rect = LetterBox((64, 64), auto=True, stride=8)(image=image)
                square = LetterBox((64, 64), auto=False, stride=8)(image=image)
                masks = rect[None, :, :, 0] > 127
                restored = collector.square_evaluation_masks(masks, shape, 64)
                np.testing.assert_array_equal(restored[0], square[:, :, 0] > 127)
                empty = collector.square_evaluation_masks(np.zeros((0, *rect.shape[:2]), bool), shape, 64)
                self.assertEqual(empty.shape, (0, 64, 64))
        with self.assertRaises(ValueError):
            collector.square_evaluation_masks(np.zeros((1, 16, 16)), (32, 64), 64)

    def test_cli_all_and_mutually_exclusive_legacy_switch(self):
        for module in (collector, validation):
            with patch.object(sys, "argv", ["script", "--weights", "x.pt", "--quality-mode", "all"]):
                self.assertEqual(module.parse_args().quality_mode, "all")
            with patch.object(sys, "argv", ["script", "--weights", "x.pt", "--quality-mode", "region", "--quality-blend", "1"]):
                with self.assertRaises(SystemExit):
                    module.parse_args()
        with patch.object(sys, "argv", ["script", "--report", "x.json", "--quality-mode", "all"]):
            with self.assertRaises(SystemExit):
                validation.parse_args()

    def test_summary_pairs_modes_by_weight_hash_and_keeps_legacy_reports(self):
        reports = []
        for seed in (0, 1):
            for mode, ap in zip(QUALITY_MODES, (0.40, 0.42, 0.41, 0.43)):
                reports.append({"configuration": "fieldbcq", "seed": 0,
                                "checkpoint_sha256": f"weights{seed}", "quality_mode": mode,
                                "quality_blend": 1, "dataset_content_sha256": "data",
                                "evaluation_protocol": {"rect": True}, "code": {"source_sha256": "code"},
                                "native": {"mask": dict(map50=ap, map75=ap, map50_95=ap)},
                                "fixed": {"f1": ap}})
        # A new training seed without the identical reference weights must not be paired.
        unmatched = copy.deepcopy(reports[-1])
        unmatched.update(seed=2, checkpoint_sha256="unpaired")
        reports.append(unmatched)
        result = summarize(reports)
        self.assertEqual(len(result), 4)
        joint = next(row for row in result if row["configuration"] == "fieldbcq_qjoint")
        self.assertEqual(joint["runs"], 3)
        for mode, gain in (("class", 3), ("region", 1), ("boundary", 2)):
            metric = joint["paired_gain_over_quality_modes"][mode]["map50_95"]
            self.assertAlmostEqual(metric["mean"], gain)
            self.assertEqual(metric["n"], 2)
            self.assertAlmostEqual(metric["sample_std"], 0)
        legacy = copy.deepcopy(reports[0])
        del legacy["quality_mode"]
        legacy["quality_blend"] = 0
        self.assertEqual(summarize([legacy])[0]["configuration"], "fieldbcq_c0")
        duplicate = copy.deepcopy(reports[0])
        duplicate["seed"] = 99
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize([reports[0], duplicate])

    def test_real_checkpoint_native_and_structural_evaluation_preserve_modes(self):
        """Run real YOLO loading, backend/fusion, NMS, rect masks and report writing."""
        import cv2
        from paper_runtime import benchmark, sha256
        from ultralytics.models.yolo.segment import SegmentationValidator
        native_scores = []
        original_postprocess = SegmentationValidator.postprocess

        def capture_native_scores(validator, predictions):
            native_scores.append(float(predictions[0][:, 4].max()))
            return original_postprocess(validator, predictions)

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temp = Path(temporary)
            (temp / "images/test").mkdir(parents=True)
            (temp / "labels/test").mkdir(parents=True)
            for index in range(2):
                cv2.imwrite(str(temp / f"images/test/{index}.png"), np.zeros((31, 64, 3), np.uint8))
                (temp / f"labels/test/{index}.txt").write_text(
                    "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n" if index == 0 else "", encoding="utf-8")
            data = temp / "data.yaml"
            data.write_text("train: images/test\nval: images/test\ntest: images/test\nnc: 1\nnames: [field]\n", encoding="utf-8")
            model = small_model().eval()
            for branch in model.model[-1].cv3:
                branch[-1].weight.data.zero_()
                branch[-1].bias.data.fill_(2.)
            for branch in model.model[-1].cv5:
                branch[-1].bias.data.copy_(torch.tensor([-1., 2.]))
            checkpoint = temp / "best.pt"
            torch.save({"model": model, "train_args": {**vars(model.args), "task": "segment", "seed": 7}}, checkpoint)
            original_hash = sha256(checkpoint)
            argv = ["collector", "--weights", "same=" + str(checkpoint), "--data", str(data),
                    "--quality-mode", "all", "--project", str(temp / "reports"),
                    "--imgsz", "64", "--batch", "2", "--workers", "0", "--max-det", "20", "--skip-benchmark"]
            with patch.object(sys, "argv", argv), \
                 patch.object(collector, "environment", return_value={}), \
                 patch.object(collector, "code_provenance", return_value={"source_sha256": "tested"}), \
                 patch.object(SegmentationValidator, "postprocess", capture_native_scores):
                collector.main()
            reports = []
            for mode in QUALITY_MODES:
                report = json.loads((temp / "reports" / ("same_q" + mode) / "paper_report.json").read_text())
                self.assertEqual(report["quality_mode"], mode)
                self.assertEqual(report["checkpoint_sha256"], original_hash)
                self.assertTrue(report["evaluation_protocol"]["rect"])
                self.assertEqual(report["fixed"]["counts"]["images"], 2)
                reports.append(report)
            self.assertGreater(reports[0]["fixed"]["counts"]["pred_instances"], 0)
            self.assertEqual(reports[1]["fixed"]["counts"]["pred_instances"], 0)
            qr, qb = torch.sigmoid(torch.tensor([-1., 2.])).tolist()
            expected_scores = [torch.sigmoid(torch.tensor(2.)).item() * factor
                               for factor in (1, qr, qb, math.sqrt(qr * qb))]
            np.testing.assert_allclose(native_scores, expected_scores, rtol=1e-5)
            self.assertEqual(len(summarize(reports)), 4)
            self.assertEqual(sha256(checkpoint), original_hash)
            # The compact AP entry point must also preserve the intervention through AutoBackend.
            with patch.object(sys, "argv", ["validation", "--weights", "same=" + str(checkpoint), "--data", str(data),
                                            "--imgsz", "64", "--batch", "2", "--workers", "0", "--max-det", "20",
                                            "--quality-mode", "all", "--project", str(temp / "compact"), "--name", "four"]), \
                 patch.object(SegmentationValidator, "postprocess", capture_native_scores):
                validation.main()
            config = json.loads((temp / "compact/four/validation_config.json").read_text())
            self.assertEqual([run["quality_mode"] for run in config["runs"]], list(QUALITY_MODES))
            self.assertTrue(all(run["evaluation_options"]["rect"] for run in config["runs"]))
            self.assertTrue(all(run["checkpoint_sha256"] == original_hash for run in config["runs"]))
            np.testing.assert_allclose(native_scores[-4:], expected_scores, rtol=1e-5)
            timing = benchmark(checkpoint, imgsz=64, batch=1, warmup=0, iterations=1,
                               max_det=20, images=[np.zeros((31, 64, 3), np.uint8)], quality_mode="boundary")
            self.assertEqual(timing["quality_mode"], "boundary")
            self.assertTrue(timing["rect"])
            self.assertGreater(timing["decoded_image_pipeline"]["batch_latency_ms_mean"], 0)


if __name__ == "__main__":
    unittest.main()
