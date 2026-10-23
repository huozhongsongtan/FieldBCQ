"""Verify isolated boundary objectives, checkpoint identity, and quality-only gradients."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from check_paper_models import synthetic_batch
from paper_runtime import configuration_of
from train_paper import STAGES, parse_args, STAGE_ALIASES
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8BASSegmentationLoss


def small_model(overrides=None):
    config = {
        "nc": 1,
        "backbone": [[-1, 1, "Conv", [16, 3, 2]],
                     [-1, 1, "Conv", [32, 3, 2]],
                     [-1, 1, "Conv", [64, 3, 2]]],
        "head": [[[0, 1, 2], 1, "BASMQCSegment", [1, 8, 16]]],
    }
    model = SegmentationModel(config, nc=1, verbose=False)
    model.args = get_cfg(overrides=overrides or {})
    return model


class QualityAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_loss_values_and_gradients_match_independent_dice_and_bce(self):
        model = small_model()
        values = torch.tensor([[[[-1.0, 0.2], [0.7, 2.0]]]])
        target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        for mode in ("none", "dice", "bce", "dice_bce"):
            with self.subTest(mode=mode):
                model.args.boundary_mode = mode
                actual_logits = values.clone().requires_grad_()
                expected_logits = values.clone().requires_grad_()
                probability = expected_logits.sigmoid()
                dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
                bce = F.binary_cross_entropy_with_logits(expected_logits, target)
                expected = {
                    "none": expected_logits.sum() * 0, "dice": dice,
                    "bce": bce, "dice_bce": dice + bce,
                }[mode]
                actual = model.init_criterion()._boundary_aux_loss(actual_logits, target)
                torch.testing.assert_close(actual, expected)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(actual_logits.grad, expected_logits.grad)

    def test_quality_only_skips_boundary_targets_but_learns_quality(self):
        model = small_model(STAGES["rbqs"]["overrides"]).train()
        for empty in (False, True):
            with self.subTest(empty=empty):
                model.zero_grad(set_to_none=True)
                with patch.object(v8BASSegmentationLoss, "_boundary_target", side_effect=AssertionError("disabled target")):
                    loss, _ = model(synthetic_batch(empty))
                self.assertTrue(torch.isfinite(loss).all())
                loss.backward()
                head = model.model[-1]
                for parameter in head.boundary_head.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)
                quality_grads = [parameter.grad for branch in head.cv5 for parameter in branch.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in quality_grads))
                if not empty:
                    self.assertGreater(sum(g.abs().sum().item() for g in quality_grads), 0)
                self.assertEqual(model.criterion.last_bas_mqc_losses["boundary"], 0)

    def test_saved_checkpoints_preserve_modes_identity_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            for stage in ("rbqs", "bds_rbqs", "baps_rbqs", "fieldbcq"):
                with self.subTest(stage=stage):
                    model = small_model(STAGES[stage]["overrides"]).eval()
                    image = torch.zeros(1, 3, 32, 32)
                    with torch.no_grad():
                        expected = model(image)[0]
                    path = Path(directory) / f"{stage}.pt"
                    torch.save({"model": model, "train_args": {**vars(model.args), "task": "segment", "seed": 3}}, path)
                    loaded = YOLO(str(path)).model.eval()
                    self.assertEqual(configuration_of(loaded), stage)
                    loaded.args = get_cfg(overrides=loaded.args)
                    self.assertEqual(loaded.init_criterion().boundary_mode, STAGES[stage]["overrides"]["boundary_mode"])
                    with torch.no_grad():
                        torch.testing.assert_close(loaded(image)[0], expected, rtol=0, atol=0)

    def test_legacy_missing_mode_keeps_joint_objective_and_zero_gain_is_c(self):
        model = small_model()
        delattr(model.args, "boundary_mode")
        self.assertEqual(configuration_of(model), "fieldbcq")
        self.assertEqual(model.init_criterion().boundary_mode, "dice_bce")
        model.train()
        model.criterion = model.init_criterion()
        delattr(model.criterion, "boundary_mode")  # a historical pickled criterion
        loss, _ = model(synthetic_batch())
        self.assertTrue(torch.isfinite(loss).all())
        logits = torch.randn(1, 1, 4, 4)
        target = torch.zeros_like(logits)
        torch.testing.assert_close(model.criterion._boundary_aux_loss(logits, target),
                                   v8BASSegmentationLoss._boundary_aux_loss(logits, target))
        model.args.boundary = 0
        self.assertEqual(configuration_of(model), "rbqs")
        model.args = vars(model.args)
        self.assertEqual(configuration_of(model), "rbqs")

    def test_invalid_mode_fails_and_short_cli_aliases_resolve(self):
        model = small_model({"boundary_mode": "invalid"})
        with self.assertRaisesRegex(ValueError, "boundary_mode"):
            model.init_criterion()
        with self.assertRaisesRegex(ValueError, "boundary_mode"):
            configuration_of(model)
        for alias, stage in (("c", "rbqs"), ("ac", "bds_rbqs"), ("bc", "baps_rbqs")):
            with patch.object(sys, "argv", ["train_paper.py", "--stage", alias]):
                self.assertEqual(STAGE_ALIASES[parse_args().stage], stage)


if __name__ == "__main__":
    unittest.main()
