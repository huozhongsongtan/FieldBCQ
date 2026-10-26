"""Verify scoring equations, unchanged geometry, rect grids and real evaluation propagation."""

import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

import paper_data as collector
from quality_scoring import QUALITY_MODES, set_quality_scoring
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
