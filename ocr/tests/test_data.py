"""Tests for loading the shared prepared-data format for OCR."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ocr.data import load_prepared_data


class PreparedDataTests(unittest.TestCase):
    def test_writer_splits_produce_correct_images_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_images = np.full((3, 8, 8), 64, dtype=np.uint8)
            val_images = np.full((3, 8, 8), 192, dtype=np.uint8)
            np.save(root / "train.npy", train_images)
            np.save(root / "val.npy", val_images)
            manifest = {
                "version": 1,
                "image_size": 8,
                "characters": ["甲", "乙", "丙"],
                "writers": [
                    {"id": "train", "file": "train.npy", "labels": [0, 1, 2], "split": "train"},
                    {"id": "val", "file": "val.npy", "labels": [2, 0, 1], "split": "val"},
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            train, val, chars, fingerprint = load_prepared_data(root)
            self.assertEqual(chars, ["甲", "乙", "丙"])
            self.assertEqual(len(train), 3)
            self.assertEqual(len(val), 3)
            train_pixels, train_label = train[1]
            val_pixels, val_label = val[0]
            self.assertEqual(train_label, 1)
            self.assertEqual(val_label, 2)
            self.assertTrue(torch.all(train_pixels == 64))
            self.assertTrue(torch.all(val_pixels == 192))
            self.assertEqual(len(fingerprint), 64)

    def test_rejects_class_missing_from_a_writer_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "train.npy", np.zeros((2, 8, 8), dtype=np.uint8))
            np.save(root / "val.npy", np.ones((1, 8, 8), dtype=np.uint8))
            manifest = {
                "version": 1,
                "image_size": 8,
                "characters": ["甲", "乙"],
                "writers": [
                    {"id": "train", "file": "train.npy", "labels": [0, 1], "split": "train"},
                    {"id": "val", "file": "val.npy", "labels": [0], "split": "val"},
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cover every OCR class"):
                load_prepared_data(root)

    def test_rejects_writer_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "version": 1,
                "image_size": 8,
                "characters": ["甲", "乙"],
                "writers": [
                    {"id": "train", "file": "../outside.npy", "labels": [0, 1], "split": "train"},
                    {"id": "val", "file": "../outside.npy", "labels": [0, 1], "split": "val"},
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_prepared_data(root)


if __name__ == "__main__":
    unittest.main()
