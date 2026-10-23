"""Known geometry verifies matching, unmatched contours, structural errors and summaries."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/scripts"))
from paper_metrics import aggregate, greedy_pairs, image_metrics
from summarize_paper_runs import summarize


def evaluate(pred, gt, scores=None, classes=None, gt_classes=None):
    return image_metrics(pred, gt, np.ones(len(pred)) if scores is None else np.asarray(scores),
                         np.zeros(len(pred), dtype=int) if classes is None else np.asarray(classes),
                         np.zeros(len(gt), dtype=int) if gt_classes is None else np.asarray(gt_classes))


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((1, 32, 32), dtype=bool)
        self.gt[:, 8:24, 8:24] = True
        self.empty = np.zeros((0, 32, 32), dtype=bool)

    def test_perfect_prediction_and_negative_image(self):
        positive = evaluate(self.gt, self.gt)
        result = aggregate([positive, evaluate(self.empty, self.empty)])
        for key in ("f1", "boundary_f1", "boundary_band_iou", "small_parcel_recall"):
            self.assertAlmostEqual(result[key], 1)
        self.assertEqual(result["parcel_count_mae_per_image"], 0)
        self.assertEqual(result["foreground_area_mae_px_per_image"], 0)
        self.assertEqual(result["score_iou_ece"], 0)

    def test_unmatched_and_wrong_class_predictions_are_penalized(self):
        missed = evaluate(self.empty, self.gt)
        false = evaluate(self.gt, self.empty)
        wrong = evaluate(self.gt, self.gt, classes=[1])
        result = aggregate([missed, false, wrong])
        self.assertEqual(result["f1"], 0)
        self.assertEqual(result["boundary_f1"], 0)
        self.assertEqual(result["boundary_band_iou"], 0)
        self.assertEqual(result["score_iou_ece"], 1)

    def test_undefined_errors_are_null_instead_of_perfect_zero(self):
        result = aggregate([evaluate(self.gt, self.empty)])
        for key in ("small_parcel_recall", "matched_boundary_mean_distance_px", "matched_parcel_area_mae_px",
                    "matched_parcel_area_mape", "foreground_area_wape"):
            self.assertIsNone(result[key])
        self.assertEqual(result["parcel_count_mae_per_image"], 1)
        self.assertGreater(result["foreground_area_mae_px_per_image"], 0)

    def test_merging_two_gt_and_fragmenting_one_gt(self):
        gt = np.zeros((2, 32, 32), dtype=bool)
        gt[0, 8:24, 8:16] = True
        gt[1, 8:24, 16:24] = True
        merged = evaluate(self.gt, gt)
        self.assertEqual(merged["merged_predictions"], 1)
        self.assertEqual(merged["tp"], 1)
        fragmented = evaluate(gt, self.gt)
        self.assertEqual(fragmented["oversegmented_gt"], 1)
        self.assertEqual(fragmented["tp"], 1)

    def test_matching_prioritizes_score_and_never_reuses_gt(self):
        pairs = greedy_pairs(np.asarray([[0.9], [0.6]]), np.asarray([0.5, 0.9]), np.ones((2, 1), dtype=bool))
        self.assertEqual(pairs, [(1, 0)])

    def test_physical_distance_and_area_scaling(self):
        shifted = np.roll(self.gt, 1, axis=2)
        record = image_metrics(shifted, self.gt, np.ones(1), np.zeros(1), np.zeros(1), sampling_metres=(2, 2))
        result = aggregate([record])
        self.assertGreater(result["matched_boundary_mean_distance_px"], 0)
        self.assertAlmostEqual(result["matched_boundary_mean_distance_m"], 2 * result["matched_boundary_mean_distance_px"])
        missed = image_metrics(self.empty, self.gt, np.zeros(0), np.zeros(0), np.zeros(1), sampling_metres=(2, 2))
        self.assertEqual(missed["foreground_area_absolute_error_m2"], 256 * 4)

    def test_ece_includes_score_one_and_false_positives(self):
        result = aggregate([evaluate(self.gt, self.gt, scores=[0.5]), evaluate(self.gt, self.empty, scores=[1.0])])
        self.assertAlmostEqual(result["score_iou_ece"], 0.75)

    def test_summary_pairs_seeds_and_separates_protocols(self):
        def report(config, seed, ap, protocol="v1"):
            return {"configuration": config, "seed": seed, "checkpoint_sha256": f"{config}{seed}",
                    "dataset_content_sha256": "same", "evaluation_protocol": {"version": protocol},
                    "code": {"source_sha256": "code"}, "native": {"mask": {"map50": ap, "map75": ap, "map50_95": ap}},
                    "fixed": {"f1": ap}, "efficiency": None}
        result = summarize([report("baseline", 1, 0.4), report("baseline", 2, 0.5),
                            report("baps", 1, 0.42), report("baps", 2, 0.54), report("baps", 3, 0.7, "v2")])
        baps = next(row for row in result if row["configuration"] == "baps" and row["runs"] == 2)
        self.assertAlmostEqual(baps["paired_gain_over_baseline"]["map50_95"]["mean"], 3)
        single = next(row for row in result if row["configuration"] == "baps" and row["runs"] == 1)
        self.assertIsNone(single["metrics"]["mask_map50_95"]["sample_std"])
        self.assertFalse(single["paired_gain_over_baseline"])
        with self.assertRaises(ValueError):
            summarize([report("baps", 1, 0.4), report("baps", 1, 0.5)])

    def test_summary_separates_all_quality_configurations_and_fusion_switches(self):
        reports = []
        for config in ("rbqs", "bds_rbqs", "baps_rbqs", "fieldbcq"):
            for blend in (0, 1):
                reports.append({
                    "configuration": config, "seed": 0, "quality_blend": blend,
                    "checkpoint_sha256": config, "dataset_content_sha256": "same",
                    "evaluation_protocol": {"version": "v1"}, "code": {"source_sha256": "code"},
                    "native": {"mask": {"map50": 0.6, "map75": 0.4, "map50_95": 0.4}},
                    "fixed": {"f1": 0.6},
                })
        result = summarize(reports)
        self.assertEqual({row["configuration"] for row in result}, {
            config + suffix for config in ("rbqs", "bds_rbqs", "baps_rbqs", "fieldbcq")
            for suffix in ("_c0", "_c2")
        })


if __name__ == "__main__":
    unittest.main()
