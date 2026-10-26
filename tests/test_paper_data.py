import sys
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/scripts"))
from paper_data import evaluation_content_shape, load_metadata, load_targets, resolve_images, square_evaluation_masks


class PaperDataTests(unittest.TestCase):
    def test_672_validation_grid_crops_only_extra_stride_padding(self):
        mask = np.zeros((1, 672, 672), bool)
        mask[0, 16:40, 16:56] = True
        mask[0, :16] = True  # Extra padding must not enter the 640-square result.
        restored = square_evaluation_masks(mask, (256, 256), 640,
                                           ratio_pad=((2.5, 2.5), (16, 16)))
        np.testing.assert_array_equal(restored, mask[:, 16:656, 16:656])
        self.assertEqual(restored.shape, (1, 640, 640))
        self.assertTrue(restored[0, 0:24, 0:40].all())
        self.assertEqual(restored.sum(), 24 * 40)
        empty = square_evaluation_masks(np.zeros((0, 672, 672), bool), (256, 256), 640,
                                        ratio_pad=((2.5, 2.5), (16, 16)))
        self.assertEqual(empty.shape, (0, 640, 640))

    def test_mixed_crop_and_pad_preserves_contours_and_square_padding_predictions(self):
        mask = np.zeros((1, 32, 96), bool)
        mask[0, 0:31, 16:80] = True
        # Image content is 31x64; source has less y padding and extra x padding.
        restored = square_evaluation_masks(mask, (31, 64), 64,
                                           ratio_pad=((1., 1.), (16, 0)))
        expected = np.zeros((1, 64, 64), bool)
        expected[:, 16:48] = mask[:, :, 16:80]
        np.testing.assert_array_equal(restored, expected)
        # Predictions in padding retained by the canonical square remain measurable as FP.
        padded = np.zeros((1, 96, 96), bool)
        padded[0, 16:24, 20:28] = True
        restored = square_evaluation_masks(padded, (31, 64), 64,
                                           ratio_pad=((1., 1.), (16, 32)))
        self.assertTrue(restored[0, :8, 4:12].all())

    def test_recorded_ceil_resize_aligns_gt_and_masks_for_portrait_and_landscape(self):
        import cv2
        for shape in ((29, 47), (47, 29)):
            height, width = shape
            gain = 64 / max(shape)
            content_h, content_w = min(math.ceil(height * gain), 64), min(math.ceil(width * gain), 64)
            src_h, src_w = 96, 96
            left = round((src_w-content_w)/2 - 0.1)
            top = round((src_h-content_h)/2 - 0.1)
            metadata = ((content_h / height, content_w / width), (left, top))
            with self.subTest(shape=shape), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                label = Path(temporary) / 'field.txt'
                label.write_text('0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n', encoding='utf-8')
                gt, classes, _ = load_targets(label, shape, 64, ratio_pad=metadata)
                points = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
                # Draw the polygon independently in the actual padded input grid.
                source = np.zeros((src_h, src_w), np.uint8)
                cv2.fillPoly(source, [(points * [content_w, content_h] + [left, top]).astype(np.int32)], 1)
                restored = square_evaluation_masks(source[None], shape, 64, ratio_pad=metadata)
                np.testing.assert_array_equal(restored, gt)
                self.assertEqual(classes.tolist(), [0])
                self.assertEqual(evaluation_content_shape(shape, 64, metadata), (content_h, content_w))

    def test_invalid_recorded_geometry_fails_instead_of_resizing(self):
        for metadata in (((3., 3.), (0, 0)), ((2.5, 2.5), (33, 16)), ((2.5, 2.5), (16.5, 16))):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                square_evaluation_masks(np.zeros((1, 672, 672), bool), (256, 256), 640, ratio_pad=metadata)

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
