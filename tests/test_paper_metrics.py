"""Numerical regressions for paper-only fixed-threshold metrics."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from paper_metrics import aggregate, greedy_pairs, image_metrics


def masks(*boxes):
    result = np.zeros((len(boxes), 32, 32), bool)
    for mask, (x1, y1, x2, y2) in zip(result, boxes):
        mask[y1:y2, x1:x2] = True
    return result


def measure(p, g, scores=None, **kwargs):
    return image_metrics(p, g, np.ones(len(p)) if scores is None else np.array(scores),
                         np.zeros(len(p), int), np.zeros(len(g), int), **kwargs)


class PaperMetricTests(unittest.TestCase):
    def test_perfect_and_negative_images(self):
        shape = masks((4, 4, 20, 20))
        result = aggregate([measure(shape, shape), measure(masks(), masks())])
        for key in ('precision', 'recall', 'f1', 'boundary_f1', 'small_parcel_recall'):
            self.assertEqual(result[key], 1)
        for key in ('merge_rate', 'oversegmentation_rate', 'score_iou_ece', 'parcel_count_mae_per_image',
                    'matched_parcel_area_mape', 'foreground_area_wape', 'foreground_area_net_bias'):
            self.assertEqual(result[key], 0)
        self.assertEqual(result['counts']['images'], 2)

    def test_empty_denominators_are_missing(self):
        result = aggregate([measure(masks(), masks())])
        for key in ('precision', 'recall', 'f1', 'boundary_f1', 'small_parcel_recall', 'merge_rate',
                    'oversegmentation_rate', 'score_iou_ece', 'matched_parcel_area_mape',
                    'foreground_area_wape', 'foreground_area_net_bias'):
            self.assertIsNone(result[key], key)
        self.assertEqual(result['parcel_count_mae_per_image'], 0)

    def test_unmatched_and_negative_false_positives_count(self):
        square = masks((4, 4, 20, 20))
        result = aggregate([measure(square, square), measure(square, masks())])
        self.assertAlmostEqual(result['f1'], 2/3)
        self.assertAlmostEqual(result['boundary_f1'], 2/3)
        self.assertEqual(result['parcel_count_mae_per_image'], 0.5)
        self.assertEqual(result['foreground_area_wape'], 1)
        self.assertEqual(result['foreground_area_net_bias'], 1)

    def test_merge_and_oversegmentation(self):
        whole = masks((4, 4, 20, 20))
        split = masks((4, 4, 12, 20), (12, 4, 20, 20))
        merged = aggregate([measure(whole, split)])
        fragmented = aggregate([measure(split, whole)])
        self.assertEqual(merged['merge_rate'], 1)
        self.assertEqual(fragmented['oversegmentation_rate'], 1)
        self.assertEqual(merged['counts']['tp'], 1)
        self.assertEqual(fragmented['counts']['tp'], 1)

    def test_ece_has_false_positives_and_score_one_last_bin(self):
        gt = masks((2, 2, 10, 10))
        pred = masks((2, 2, 10, 10), (20, 20, 28, 28))
        result = aggregate([measure(pred, gt, [0.5, 1.0])])
        self.assertEqual(result['score_iou_ece'], 0.75)

    def test_area_uses_union_and_keeps_signed_bias(self):
        gt = masks((2, 2, 12, 12))
        pred = masks((2, 2, 7, 12))
        result = aggregate([measure(pred, gt)])
        self.assertEqual(result['matched_parcel_area_mape'], 0.5)
        self.assertEqual(result['foreground_area_wape'], 0.5)
        self.assertEqual(result['foreground_area_net_bias'], -0.5)
        overlap = aggregate([measure(np.concatenate([gt, gt]), gt)])
        self.assertEqual(overlap['foreground_area_wape'], 0)
        self.assertEqual(overlap['parcel_count_mae_per_image'], 1)

    def test_matching_score_order_no_gt_reuse_wrong_class(self):
        iou = np.array([[0.8], [0.9]])
        self.assertEqual(greedy_pairs(iou, np.array([0.9, 0.8]), np.ones((2, 1), bool)), [(0, 0)])
        self.assertEqual(greedy_pairs(iou, np.array([0.9, 0.8]), np.zeros((2, 1), bool)), [])

    def test_f1_only_skips_contours_calibration_and_area(self):
        square = masks((4, 4, 20, 20))
        with patch('paper_metrics.boundary', side_effect=AssertionError('Unrequested boundary calculation')):
            result = aggregate([measure(square, square, scope=set())])
        self.assertEqual(set(result), {'counts', 'precision', 'recall', 'f1'})
