"""Small CPU integration fixtures; independent of the paper dataset and archived weights."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/scripts"))
from paper_runtime import sha256
from paper_validator import evaluate
from test_quality_ablation import small_model
from ultralytics.models.yolo.segment import SegmentationValidator


class PaperEvaluationTests(unittest.TestCase):
    def test_native_and_fixed_share_pass_and_preserve_class_joint_checkpoint(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                root = Path(temporary)
                images, labels = root / "images/test", root / "labels/test"
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                paths = {}
                for i in range(2):
                    image, label = images / f"{i}.png", labels / f"{i}.txt"
                    cv2.imwrite(str(image), np.zeros((31, 64, 3), np.uint8))
                    label.write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n" if i == 0 else "", encoding="utf-8")
                    paths[image.resolve()] = label.resolve()
                data = root / "data.yaml"
                data.write_text(f"path: {root.as_posix()}\ntrain: images/test\nval: images/test\ntest: images/test\nnames: [field]\n", encoding="utf-8")
                model = small_model().eval()
                for branch in model.model[-1].cv3:
                    branch[-1].weight.data.zero_()
                    branch[-1].bias.data.fill_(2.)
                for branch in model.model[-1].cv5:
                    branch[-1].weight.data.zero_()
                    branch[-1].bias.data.fill_(-3.)
                checkpoint = root / "best.pt"
                torch.save({"model": model, "train_args": {**vars(model.args), "task": "segment", "seed": 7}}, checkpoint)
                original_hash = sha256(checkpoint)
                args = SimpleNamespace(tables=list(range(4, 10)), imgsz=64, batch=2, workers=0, device="cpu")
                raw_scores = []
                original = SegmentationValidator.postprocess

                def capture(validator, predictions):
                    raw_scores.append(float(predictions[0][:, 4].max()))
                    return original(validator, predictions)

                with patch.object(SegmentationValidator, "postprocess", capture):
                    c, cr = evaluate(checkpoint, "fieldbcq", args, data, paths, root / "class", "class")
                    j, jr = evaluate(checkpoint, "fieldbcq", args, data, paths, root / "joint", "joint")
                self.assertEqual(len(raw_scores), 2)  # one batch per mode, no second prediction pass
                self.assertAlmostEqual(raw_scores[0], torch.sigmoid(torch.tensor(2.)).item(), places=5)
                self.assertAlmostEqual(raw_scores[1] / raw_scores[0], torch.sigmoid(torch.tensor(-3.)).item(), places=5)
                for report, records in ((c, cr), (j, jr)):
                    self.assertEqual(report["checkpoint_sha256"], original_hash)
                    self.assertEqual(report["seed"], 7)
                    self.assertEqual(report["fixed"]["counts"]["images"], 2)
                    self.assertEqual(report["fixed"]["counts"]["gt_instances"], 1)
                    self.assertEqual(set(records), set(paths))
                    self.assertNotIn("box", report)
                self.assertGreater(c["fixed"]["counts"]["pred_instances"], 0)
                self.assertEqual(j["fixed"]["counts"]["pred_instances"], 0)
                self.assertEqual(sha256(checkpoint), original_hash)
        finally:
            torch.set_num_threads(old_threads)
