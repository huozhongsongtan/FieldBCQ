"""Regression coverage for historical BAPS-only checkpoints."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/scripts'))

from paper_runtime import configuration_of
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.extra_modules import BASSegment
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8BAPSSegmentationLoss, v8BASOnlySegmentationLoss


def small_model(head, mode=None):
    config = {
        'nc': 1,
        'backbone': [[-1, 1, 'Conv', [16, 3, 2]],
                     [-1, 1, 'Conv', [32, 3, 2]],
                     [-1, 1, 'Conv', [64, 3, 2]]],
        'head': [[[0, 1, 2], 1, head, [1, 8, 16]]],
    }
    model = SegmentationModel(config, nc=1, verbose=False)
    overrides = {'boundary_mode': mode} if mode else {}
    model.args = get_cfg(overrides=overrides)
    return model


class LegacyBAPSCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_legacy_checkpoint_loads_without_changing_predictions_or_weights(self):
        model = small_model('BASSegment').eval()
        self.assertIsInstance(model.model[-1], BASSegment)
        # Historical checkpoints may omit boundary_mode and retain a loss object.
        delattr(model.args, 'boundary_mode')
        model.criterion = v8BASOnlySegmentationLoss(model)
        image = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
        with torch.no_grad():
            expected, expected_auxiliary = model(image)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'legacy_baps.pt'
            torch.save({'model': model, 'train_args': {'task': 'segment', 'seed': 7}}, checkpoint)
            original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            loaded = YOLO(str(checkpoint))
            self.assertEqual(type(loaded.model.model[-1]).__name__, 'BASSegment')
            self.assertEqual(loaded.task, 'segment')
            self.assertEqual(configuration_of(loaded.model), 'baps')
            self.assertEqual(loaded.ckpt['train_args']['seed'], 7)
            self.assertIsInstance(loaded.model.criterion, v8BASOnlySegmentationLoss)
            # The trainer converts checkpoint argument dicts to its cfg namespace.
            loaded.model.args = get_cfg(overrides=loaded.model.args)
            delattr(loaded.model.args, 'boundary_mode')
            self.assertIsInstance(loaded.model.init_criterion(), v8BAPSSegmentationLoss)
            with torch.no_grad():
                actual, actual_auxiliary = loaded.model(image)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_auxiliary[-1], expected_auxiliary[-1], rtol=0, atol=0)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), original_hash)

    def test_legacy_loss_is_bce_only(self):
        logits = torch.tensor([[[[-1.0, 0.2], [0.7, 2.0]]]], requires_grad=True)
        target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        model = small_model('BASSegment')
        delattr(model.args, 'boundary_mode')
        criterion = model.init_criterion()
        self.assertIsInstance(criterion, v8BAPSSegmentationLoss)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        actual = criterion._boundary_aux_loss(logits, target)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(v8BASOnlySegmentationLoss._boundary_aux_loss(logits, target), expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_current_configurations_keep_their_loss_and_identity(self):
        settings = [
            ('Segment', None, 'baseline', 'v8SegmentationLoss'),
            ('BoundaryDiceSegment', None, 'pcsc', 'v8BoundaryDiceSegmentationLoss'),
            ('Segment_BAS', 'bce', 'baps', 'v8BAPSSegmentationLoss'),
            ('Segment_BAS', 'dice_bce', 'baps_pcsc', 'v8BASSegmentationLoss'),
            ('BASMQCSegment', None, 'fieldbcq', 'v8BASMQCSegmentationLoss'),
        ]
        for head, mode, configuration, loss in settings:
            with self.subTest(head=head, mode=mode):
                model = small_model(head, mode)
                self.assertEqual(configuration_of(model), configuration)
                self.assertEqual(type(model.init_criterion()).__name__, loss)


if __name__ == '__main__':
    unittest.main()
