"""Check path portability without loading models or running inference."""

import argparse
import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch



ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "src" / "scripts"


class RelativePathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Mock())}):
            cls.paths = importlib.import_module("project_paths")
            cls.train = importlib.import_module("train_paper")

    def temporary_directory(self):
        return tempfile.TemporaryDirectory(prefix="relative_paths_", dir=ROOT)

    def test_dataset_references_without_python39_path_methods(self):
        base = ROOT / "数据集"
        inside = base / "images" / "二轮.png"
        outside = ROOT / "数据集_backup" / "二轮.png"
        with patch.object(Path, "is_relative_to", create=True, side_effect=AssertionError("Python 3.9 API used")):
            self.assertEqual(self.paths.relative_or_absolute(inside, base), str(Path("images") / "二轮.png"))
            self.assertEqual(self.paths.relative_or_absolute(outside, base), str(outside))
            self.assertEqual(self.paths.relative_or_absolute(base, base), ".")

    def test_amp_flags_without_python39_argparse_actions(self):
        with patch.object(argparse, "BooleanOptionalAction", None, create=True):
            for flags, expected in (([], True), (["--amp"], True), (["--no-amp"], False)):
                with self.subTest(flags=flags), patch.object(sys, "argv", ["train_paper.py"] + flags):
                    self.assertIs(self.train.parse_args().amp, expected)

    def test_paths_follow_relocated_script_instead_of_working_directory(self):
        with self.temporary_directory() as temporary:
            renamed = Path(temporary) / "renamed_project"
            destination = renamed / "src" / "scripts" / "project_paths.py"
            destination.parent.mkdir(parents=True)
            shutil.copyfile(SCRIPTS / "project_paths.py", destination)
            spec = importlib.util.spec_from_file_location("relocated_paths", destination)
            module = importlib.util.module_from_spec(spec)
            old_cwd = Path.cwd()
            try:
                os.chdir(temporary)
                with patch.dict(os.environ, {}, clear=True):
                    spec.loader.exec_module(module)
                    self.assertEqual(module.project_path("data/datasets.yaml"), renamed / "data/datasets.yaml")
                    self.assertEqual(module.relative_path(renamed / "model/a/best.pt"), "model/a/best.pt")
                    self.assertEqual(Path(os.environ["YOLO_CONFIG_DIR"]), renamed / "results/runtime_config")
            finally:
                os.chdir(old_cwd)

    def test_training_uses_project_relative_arguments_from_another_directory(self):
        with self.temporary_directory() as temporary:
            old_cwd = Path.cwd()
            model = Mock()
            try:
                os.chdir(temporary)
                with patch.object(self.train, "YOLO", return_value=model) as constructor:
                    with patch.object(sys, "argv", ["train_paper.py", "--stage", "fieldbcq", "--no-pretrained"]):
                        self.train.main()
                self.assertEqual(Path(constructor.call_args.args[0]), ROOT / "src/ultralytics/cfg/models/11/yolo11x-seg-bas-mqc.yaml")
                self.assertEqual(Path(model.train.call_args.kwargs["data"]), ROOT / "data/datasets.yaml")
                self.assertEqual(Path(model.train.call_args.kwargs["project"]), ROOT / "results/train")
            finally:
                os.chdir(old_cwd)

    def test_public_training_names_select_expected_model_and_losses(self):
        self.assertEqual(set(self.train.STAGES), {
            "baseline", "pcsc", "baps", "baps_pcsc", "rbqs", "bds_rbqs", "baps_rbqs", "fieldbcq",
        })
        for stage in self.train.STAGES.values():
            self.assertTrue((ROOT / "src" / stage["model"]).is_file())
        configurations = [
            ("baps", "yolo11x-seg-bas.yaml", {"boundary": 0.20, "boundary_mode": "bce"}),
            ("pcsc", "yolo11x-seg-boundary-dice.yaml", {"boundary": 0.20}),
            ("baps_pcsc", "yolo11x-seg-bas.yaml", {"boundary": 0.20}),
            ("rbqs", "yolo11x-seg-bas-mqc.yaml", {
                "boundary": 0.0, "boundary_mode": "none", "btq_quality": 0.25, "btq_boundary_width": 2,
            }),
            ("bds_rbqs", "yolo11x-seg-bas-mqc.yaml", {
                "boundary": 0.20, "boundary_mode": "dice", "btq_quality": 0.25, "btq_boundary_width": 2,
            }),
            ("baps_rbqs", "yolo11x-seg-bas-mqc.yaml", {
                "boundary": 0.20, "boundary_mode": "bce", "btq_quality": 0.25, "btq_boundary_width": 2,
            }),
            ("fieldbcq", "yolo11x-seg-bas-mqc.yaml", {
                "boundary": 0.20,
                "btq_quality": 0.25,
                "btq_boundary_width": 2,
            }),
        ]
        for name, configuration, overrides in configurations:
            with self.subTest(name=name):
                model = Mock()
                argv = ["train_paper.py", "--stage", name, "--no-pretrained", "--seed", "3"]
                with patch.object(self.train, "YOLO", return_value=model) as constructor:
                    with patch.object(sys, "argv", argv):
                        self.train.main()
                self.assertEqual(
                    Path(constructor.call_args.args[0]),
                    ROOT / "src/ultralytics/cfg/models/11" / configuration,
                )
                arguments = model.train.call_args.kwargs
                self.assertEqual(arguments["name"], name + "_seed3")
                for key, value in overrides.items():
                    self.assertEqual(arguments[key], value)
