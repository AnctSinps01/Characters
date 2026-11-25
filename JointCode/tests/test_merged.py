"""End-to-end check of the merged-file path: prepare_merged -> MergedDataset."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from JointCode.merged import MergedDataset
from JointCode.prepare_merged import merge
from JointCode.tests.test_jointcode import fixture
from Ref2Latent.data import LatentStore
from VAE.data import GlyphStore

IMAGE = 64
LATENT = 8
CHANNELS = 2


class MergedTests(unittest.TestCase):
    def _prepare(self, root):
        fixture(root)
        store = GlyphStore(root / "data")
        content = LatentStore(root / "content")
        author = LatentStore(root / "author")
        output = root / "merged"
        output.mkdir()
        manifest = {"image_size": IMAGE,
                    "latent_shape": [CHANNELS * 2, LATENT, LATENT], "splits": {}}
        try:
            for split in ("train", "val"):
                writers = [w for w in store.writers if w["split"] == split]
                latents, images = merge(store, content, author, writers)
                np.save(output / f"{split}_latent.npy", latents)
                np.save(output / f"{split}_image.npy", images)
                manifest["splits"][split] = {"samples": int(len(latents)),
                                             "writers": len(writers),
                                             "latent": f"{split}_latent.npy",
                                             "image": f"{split}_image.npy"}
            (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        finally:
            store.close()
            content.close()
            author.close()
        return output

    def test_split_halves_and_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = self._prepare(root)
            dataset = MergedDataset(merged, "train")
            try:
                item = dataset[0]
                self.assertEqual(tuple(item["content"].shape), (CHANNELS, LATENT, LATENT))
                self.assertEqual(tuple(item["author"].shape), (CHANNELS, LATENT, LATENT))
                self.assertEqual(tuple(item["image"].shape), (1, IMAGE, IMAGE))
                store = GlyphStore(root / "data")
                content = LatentStore(root / "content")
                writer = [w for w in store.writers if w["split"] == "train"][0]
                try:
                    expected = np.asarray(content.writer(writer)[0], np.float32)
                    torch.testing.assert_close(item["content"], torch.from_numpy(expected))
                finally:
                    store.close()
                    content.close()
            finally:
                dataset.close()

    def test_split_sizes_and_missing_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = self._prepare(root)
            train = MergedDataset(merged, "train")
            val = MergedDataset(merged, "val")
            try:
                self.assertGreater(len(train), 0)
                self.assertGreater(len(val), 0)
                with self.assertRaises(ValueError):
                    MergedDataset(merged, "test")
            finally:
                train.close()
                val.close()
