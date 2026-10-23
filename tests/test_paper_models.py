"""Regression checks for the public model graph, gradients, and calibration."""

import gc
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

import torch

from check_paper_models import check_configuration
from train_paper import STAGES
from ultralytics.nn.modules.head import BASMQCSegment, Detect
from ultralytics.utils.loss import v8BASMQCSegmentationLoss, v8BAPSSegmentationLoss, v8BASSegmentationLoss


class PaperModelTests(unittest.TestCase):
    def test_baps_only_matches_bce_without_the_dice_term(self):
        logits = torch.tensor([[[[-1.0, 0.2], [0.7, 2.0]]]], requires_grad=True)
        target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        actual = v8BAPSSegmentationLoss._boundary_aux_loss(logits, target)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(v8BASSegmentationLoss._boundary_aux_loss(logits, target).item(), actual.item())
        actual.backward()
        torch.testing.assert_close(logits.grad, (logits.detach().sigmoid() - target) / target.numel())

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_public_models_support_inference_and_positive_empty_training(self):
        for name in STAGES:
            with self.subTest(stage=name):
                check_configuration(name)
                gc.collect()

    def test_quality_calibration_changes_scores_only(self):
        head = BASMQCSegment(nc=1, nm=32, npr=64, ch=(32, 64, 128)).eval()
        head.stride = torch.tensor([8., 16., 32.])
        features = [torch.randn(1, channels, size, size) for channels, size in ((32, 8), (64, 4), (128, 2))]
        with torch.no_grad():
            raw = Detect.forward(head, list(features))[0]
            logits = torch.cat([head.cv5[i](feature).view(1, 2, -1) for i, feature in enumerate(features)], 2)
            expected_scores = raw[:, 4:5] * logits.sigmoid().prod(1, keepdim=True).sqrt()
            calibrated = head(list(features))[0]
            torch.testing.assert_close(calibrated[:, :4], raw[:, :4])
            torch.testing.assert_close(calibrated[:, 4:5], expected_scores)
            head.quality_blend = 0.0
            uncalibrated = head(list(features))[0]
            torch.testing.assert_close(uncalibrated[:, :5], raw)
            torch.testing.assert_close(uncalibrated[:, 5:], calibrated[:, 5:])

    def test_quality_targets_do_not_backpropagate_to_masks(self):
        logits = torch.randn(2, 16, 16, requires_grad=True)
        target = torch.zeros_like(logits)
        target[:, 4:12, 4:12] = 1
        valid = torch.ones_like(logits)
        for value in (
            v8BASMQCSegmentationLoss._soft_iou_target(logits, target, valid),
            v8BASMQCSegmentationLoss._boundary_f1_target(logits, target, valid),
        ):
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(((value >= 0) & (value <= 1)).all())


if __name__ == "__main__":
    unittest.main()
