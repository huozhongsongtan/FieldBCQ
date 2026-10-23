"""Check path portability without loading models or running inference."""

import argparse
import ast
import glob
import importlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "src" / "scripts"


class RelativePathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Mock())}):
            cls.paths = importlib.import_module("project_paths")
            cls.train = importlib.import_module("train_paper")
            cls.single = importlib.import_module("val_paper")
            cls.multi = importlib.import_module("val_paper_all_models")
            cls.source = importlib.import_module("val_by_source")

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
        self.assertEqual(set(self.train.STAGES), {"baseline", "pcsc", "baps", "baps_pcsc", "fieldbcq"})
        for stage in self.train.STAGES.values():
            self.assertTrue((ROOT / "src" / stage["model"]).is_file())
        configurations = [
            ("baps", "yolo11x-seg-bas.yaml", {"boundary": 0.20, "boundary_mode": "bce"}),
            ("pcsc", "yolo11x-seg-boundary-dice.yaml", {"boundary": 0.20}),
            ("baps_pcsc", "yolo11x-seg-bas.yaml", {"boundary": 0.20}),
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

    def test_default_models_and_checkpoint_names(self):
        defaults = self.multi.parse_models(None)
        self.assertEqual([name for name, _ in defaults], ["baseline", "pcsc", "baps_pcsc", "fieldbcq"])
        checkpoint_directories = ["baseline", "a", "ab", "abc"]
        for (_, path), directory in zip(defaults, checkpoint_directories):
            self.assertEqual(path, ROOT / "model" / directory / "best.pt")
        models = self.multi.parse_models(["model/a/best.pt", "model/ab/best.pt"])
        self.assertEqual([name for name, _ in models], ["a", "ab"])
        with self.assertRaises(ValueError):
            self.multi.parse_models(["a=model/a/best.pt", "a=model/ab/best.pt"])

    def test_multi_model_summary_stores_relative_references(self):
        with self.temporary_directory() as temporary:
            temporary = Path(temporary)
            weight = temporary / "model" / "abc" / "best.pt"
            weight.parent.mkdir(parents=True)
            weight.touch()
            output = temporary / "results"
            metric = SimpleNamespace(mp=0.7, mr=0.6, map50=0.64, map75=0.43, map=0.42)

            def validate(**kwargs):
                save_dir = Path(kwargs["project"]) / kwargs["name"]
                save_dir.mkdir(parents=True)
                return SimpleNamespace(save_dir=save_dir, box=metric, seg=metric)

            model = SimpleNamespace(val=validate)
            argv = ["val_paper_all_models.py", "--weights", "abc=" + str(weight), "--project", str(output)]
            with patch.object(self.multi, "YOLO", return_value=model), patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    self.multi.main()
            record = json.loads((output / "paper_metrics_all.json").read_text(encoding="utf-8"))[0]
            self.assertFalse(Path(record["weights"]).is_absolute())
            self.assertEqual((ROOT / record["weights"]).resolve(), weight)
            self.assertEqual(record["data"], "data/datasets.yaml")

    def test_group_yaml_and_lists_survive_project_move(self):
        with self.temporary_directory() as temporary:
            original = Path(temporary) / "original"
            data = original / "data"
            for split in ("train", "val", "test"):
                (data / "images" / split).mkdir(parents=True)
            image = data / "images/test/DDFHAPD_example.png"
            image.touch()
            config = {"train": "images/train", "val": "images/val", "test": "images/test", "nc": 1, "names": ["field"]}
            config_path = data / "datasets.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            _, loaded, images = self.source.resolve_images(str(config_path), "test")
            generated = original / "results" / "source_eval" / "group_data"
            files = self.source.write_group_data(loaded, {"DDFHAPD": images}, generated, data)
            group_path = files["DDFHAPD"]
            group = yaml.safe_load(group_path.read_text(encoding="utf-8"))
            self.assertNotIn("path", group)
            for split in ("train", "val", "test"):
                self.assertFalse(Path(group[split]).is_absolute())
            list_path = generated / group["test"]
            entry = list_path.read_text(encoding="utf-8").strip()
            self.assertTrue(entry.startswith("./"))

            renamed = original.with_name("renamed")
            original.rename(renamed)
            moved_group = renamed / group_path.relative_to(original)
            _, _, moved_images = self.source.resolve_images(str(moved_group), "test")
            expected = renamed / image.relative_to(original)
            self.assertEqual(moved_images, [expected])

            # Exercise the actual image-list reader without importing the GPU stack.
            tree = ast.parse((ROOT / "src/ultralytics/data/base.py").read_text(encoding="utf-8"))
            base_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseDataset")
            reader = next(node for node in base_class.body if isinstance(node, ast.FunctionDef) and node.name == "get_img_files")
            namespace = {"Path": Path, "glob": glob, "os": os, "IMG_FORMATS": ("png",), "FORMATS_HELP_MSG": "", "HELP_URL": ""}
            exec(compile(ast.Module(body=[reader], type_ignores=[]), "image_list_reader", "exec"), namespace)
            loader = SimpleNamespace(prefix="", fraction=1.0)
            moved_list = renamed / list_path.relative_to(original)
            actual = namespace["get_img_files"](loader, str(moved_list))
            self.assertEqual([Path(value).resolve() for value in actual], [expected])


if __name__ == "__main__":
    unittest.main()
