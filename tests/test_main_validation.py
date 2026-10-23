"""Keep native-AP summaries separate from fixed-threshold and repeated inference."""

import csv
import importlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/scripts"))
validation = importlib.import_module("val_paper")


class MainValidationTests(unittest.TestCase):
    def test_report_extraction_uses_native_mask_metrics_without_loading_a_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            report = Path(temporary) / "paper_report.json"
            report.write_text(json.dumps({
                "run_name": "baseline_seed7", "seed": 7,
                "native": {"mask": {"precision": 0.6, "recall": 0.3, "f1": 0.99,
                                    "map50": 0.5, "map75": 0.4, "map50_95": 0.2}},
                "fixed": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
            }), encoding="utf-8")
            output = Path(temporary) / "summary"
            argv = ["val_paper.py", "--report", str(report), "--project", str(output), "--name", "existing"]
            with patch.dict(sys.modules, {"torch": None, "ultralytics": None}), patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    validation.main()
            with (output / "existing/paper_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seed"], "7")
            self.assertAlmostEqual(float(rows[0]["F1"]), 0.4)
            self.assertAlmostEqual(float(rows[0]["AP50_95"]), 0.2)
            config = json.loads((output / "existing/validation_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["mode"], "extract_existing_reports")

    def test_native_evaluation_runs_once_and_records_actual_checkpoint_seed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temporary = Path(temporary)
            checkpoint = temporary / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            data = temporary / "data.yaml"
            data.write_text("names: [field]\n", encoding="utf-8")
            with patch.object(sys, "argv", ["val_paper.py", "--weights", str(checkpoint), "--data", str(data)]):
                args = validation.parse_args()
            model = Mock()
            model.model.model = [SimpleNamespace()]
            model.model.parameters.return_value = iter([SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype="float32")])
            model.ckpt = {"train_args": {"seed": 19}}
            model.val.return_value = SimpleNamespace(seg=SimpleNamespace(mp=0.6, mr=0.3,
                                                     map50=0.5, map75=0.4, map=0.2))
            ultra = SimpleNamespace(YOLO=Mock(return_value=model), __version__="test", __file__="local-fork")
            torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None))
            with patch.dict(sys.modules, {"ultralytics": ultra, "torch": torch,
                                          "cv2": SimpleNamespace(__version__="test")}):
                with patch.object(validation, "configuration_of", return_value="baseline"), \
                     patch.object(validation, "main_validator_class", return_value=object), \
                     patch.object(validation, "concise_logs", return_value=nullcontext()):
                    row, info = validation.evaluate_one(args, "example", checkpoint, temporary)
            model.val.assert_called_once()
            model.predict.assert_not_called()
            self.assertEqual(row["seed"], 19)
            self.assertAlmostEqual(row["F1"], 0.4)
            self.assertEqual(len(info["checkpoint_sha256"]), 64)
            options = model.val.call_args.kwargs
            for flag in ("plots", "save", "save_json", "save_txt", "rect", "augment", "half"):
                self.assertFalse(options[flag])
            self.assertEqual(options["conf"], 0.001)
            self.assertEqual(options["iou"], 0.7)
            self.assertEqual(options["max_det"], 710)

    def test_completed_rows_survive_a_later_model_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temporary = Path(temporary)
            checkpoint = temporary / "best.pt"
            checkpoint.touch()
            row = validation.metric_row("first", None, dict(precision=0, recall=0, map50=0, map75=0, map50_95=0))
            argv = ["val_paper.py", "--weights", "first=" + str(checkpoint),
                    "--weights", "second=" + str(checkpoint), "--project", str(temporary), "--name", "run"]
            with patch.object(sys, "argv", argv), patch.object(validation, "evaluate_one", side_effect=[
                (row, {"model": "first", "seed": None}), RuntimeError("second model failed")
            ]), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "second model failed"):
                    validation.main()
            with (temporary / "run/paper_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual([entry["model"] for entry in saved], ["first"])
            self.assertEqual(saved[0]["seed"], "")


if __name__ == "__main__":
    unittest.main()
