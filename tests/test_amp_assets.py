"""AMP checks must handle missing package assets without hiding failed comparisons."""

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/scripts"))
import project_paths
import cv2
import torch
import ultralytics
from ultralytics.utils import checks, torch_utils


class AMPAssetTests(unittest.TestCase):
    @staticmethod
    def cuda_model():
        return SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="cuda"))]),
                               stride=torch.tensor([32]))

    def test_check_image_is_bundled_and_readable(self):
        image = cv2.imread(str(checks.ASSETS / "bus.jpg"))
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (1080, 810, 3))

    def test_missing_image_disables_amp_before_loading_or_downloading_auxiliary_model(self):
        with patch.object(checks, "ASSETS", ROOT / "nonexistent_amp_assets"), \
             patch.object(ultralytics, "YOLO") as loader, patch.object(checks.LOGGER, "warning") as warning:
            self.assertFalse(checks.check_amp(self.cuda_model()))
        loader.assert_not_called()
        self.assertIn("AMP will be disabled", warning.call_args.args[0])

    def test_present_image_runs_both_precision_checks_and_preserves_failed_check(self):
        values = torch.tensor([[0., 0., 20., 20., 0.9, 0.]])
        for mismatch in (False, True):
            calls = []

            def infer(batch, **kwargs):
                self.assertEqual(batch, [checks.ASSETS / "bus.jpg"] * 8)
                calls.append(kwargs)
                boxes = values + (10 if mismatch and len(calls) == 2 else 0)
                return [SimpleNamespace(boxes=SimpleNamespace(data=boxes))]

            with self.subTest(mismatch=mismatch), \
                 patch.object(ultralytics, "YOLO", return_value=infer), \
                 patch.object(torch_utils, "autocast", return_value=nullcontext()):
                self.assertEqual(checks.check_amp(self.cuda_model()), not mismatch)
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
