import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/scripts"))
from paper_data import load_metadata, resolve_images


class PaperDataTests(unittest.TestCase):
    def test_yaml_root_and_manifest_basenames_survive_relocation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            images = root / "dataset/images/test"
            images.mkdir(parents=True)
            for name in ("a.png", "b.png"):
                (images / name).write_bytes(b"path fixture")
            data = root / "data/test.yaml"
            data.write_text("path: ../dataset\ntest: images/test\nnames: [field]\n", encoding="utf-8")
            _, _, dataset, paths = resolve_images(data)
            metadata = root / "metadata.csv"
            metadata.write_text("image,source,resolution_m\na.png,DDFHAPD,2.0\nb.png,FBIS,0.3\n", encoding="utf-8")
            rows = load_metadata(metadata, dataset, paths)
            self.assertEqual(rows[paths[0]]["source"], "FHAPD")
            self.assertEqual(rows[paths[0]]["resolution_m"], "2")
            with self.assertRaisesRegex(ValueError, "match the metadata exactly"):
                load_metadata(metadata, dataset, paths[:1])

    def test_duplicate_basenames_require_relative_paths(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            images = [root / "one/a.png", root / "two/a.png"]
            metadata = root / "metadata.csv"
            metadata.write_text("image,source,resolution_m\na.png,FBIS,1\nother.png,FTW,10\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ambiguous metadata"):
                load_metadata(metadata, root, images)
