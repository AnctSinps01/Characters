"""Tests for the joint content+author decoder."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from JointCode.config import DecoderConfig
from JointCode.data import JointCodeDataset
from JointCode.model import JointDecoder
from JointCode.train import TrainConfig, learning_rate
from Ref2Latent.data import LatentStore
from VAE.data import GlyphStore

IMAGE = 64
LATENT = 8
CHANNELS = 2


def fixture(root: Path, writers=3, classes=6, per_writer=4, seed=0):
    rng = np.random.default_rng(seed)
    for name in ("data", "content", "author"):
        (root / name).mkdir(parents=True)
    entries = []
    for index in range(writers):
        labels = (np.arange(per_writer) + index) % classes
        np.save(root / "data" / f"w{index}.npy",
                rng.integers(0, 256, size=(per_writer, IMAGE, IMAGE), dtype=np.uint8))
        for key in ("content", "author"):
            np.save(root / key / f"w{index}.npy",
                    rng.normal(size=(per_writer, CHANNELS, LATENT, LATENT)).astype(np.float16))
        entries.append({"id": f"w{index}", "file": f"w{index}.npy",
                        "split": "train" if index < writers - 1 else "val",
                        "labels": [int(x) for x in labels]})
    np.save(root / "data" / "standards.npy",
            rng.integers(0, 256, size=(classes, IMAGE, IMAGE), dtype=np.uint8))
    (root / "data" / "manifest.json").write_text(json.dumps({
        "version": 1, "image_size": IMAGE,
        "characters": [chr(0x4E00 + i) for i in range(classes)],
        "writers": entries}), encoding="utf-8")
    for key in ("content", "author"):
        (root / key / "manifest.json").write_text(json.dumps({
            "latent_shape": [CHANNELS, LATENT, LATENT], "image_size": IMAGE,
            "classes": classes}), encoding="utf-8")


def small_config(**overrides):
    base = dict(latent_channels=CHANNELS, latent_size=LATENT, base_channels=8)
    base.update(overrides)
    return DecoderConfig(**base)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.model = JointDecoder(self.config)

    def test_output_shape_and_range(self):
        out = self.model(torch.randn(2, CHANNELS, LATENT, LATENT),
                         torch.randn(2, CHANNELS, LATENT, LATENT))
        self.assertEqual(tuple(out.shape), (2, 1, IMAGE, IMAGE))
        self.assertLessEqual(out.abs().max().item(), 1.0)

    def test_both_codes_are_required(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, CHANNELS, LATENT, LATENT),
                       torch.randn(2, CHANNELS, LATENT, LATENT - 1))

    def test_changing_either_code_changes_output(self):
        content = torch.randn(1, CHANNELS, LATENT, LATENT)
        author = torch.randn(1, CHANNELS, LATENT, LATENT)
        base = self.model(content, author)
        self.assertFalse(torch.allclose(base, self.model(content, author + 1.0)))
        self.assertFalse(torch.allclose(base, self.model(content + 1.0, author)))

    def test_gradients_reach_every_parameter(self):
        out = self.model(torch.randn(2, CHANNELS, LATENT, LATENT),
                         torch.randn(2, CHANNELS, LATENT, LATENT))
        out.square().mean().backward()
        missing = [n for n, p in self.model.named_parameters()
                   if p.grad is None or not torch.isfinite(p.grad).all()]
        self.assertEqual(missing, [])

    def test_small_batch_overfits(self):
        torch.manual_seed(0)
        model = JointDecoder(small_config(base_channels=16))
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        content = torch.randn(2, CHANNELS, LATENT, LATENT)
        author = torch.randn(2, CHANNELS, LATENT, LATENT)
        target = torch.randn(2, 1, IMAGE, IMAGE).clamp(-1, 1)
        first = None
        for _ in range(200):
            optimizer.zero_grad()
            loss = (model(content, author) - target).square().mean()
            loss.backward()
            optimizer.step()
            first = loss.item() if first is None else first
        self.assertLess(loss.item(), 0.3 * first)


class DataTests(unittest.TestCase):
    def test_pairs_and_code_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            store = GlyphStore(root / "data")
            content = LatentStore(root / "content")
            author = LatentStore(root / "author")
            try:
                dataset = JointCodeDataset(store, content, author, split="train")
                item = dataset[0]
                self.assertEqual(tuple(item["content"].shape), (CHANNELS, LATENT, LATENT))
                self.assertEqual(tuple(item["author"].shape), (CHANNELS, LATENT, LATENT))
                self.assertEqual(tuple(item["image"].shape), (1, IMAGE, IMAGE))
                self.assertIsInstance(item["writer_id"], str)
            finally:
                store.close()
                content.close()
                author.close()

    def test_mismatched_code_shapes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            (root / "author" / "manifest.json").write_text(json.dumps({
                "latent_shape": [3, LATENT, LATENT], "image_size": IMAGE,
                "classes": 6}), encoding="utf-8")
            store = GlyphStore(root / "data")
            content = LatentStore(root / "content")
            author = LatentStore(root / "author")
            try:
                with self.assertRaises(ValueError):
                    JointCodeDataset(store, content, author)
            finally:
                store.close()
                content.close()
                author.close()


class ConfigTests(unittest.TestCase):
    def test_schedule_and_validation(self):
        config = TrainConfig(steps=100, warmup_steps=10)
        self.assertLess(learning_rate(config, 5), learning_rate(config, 50))
        with self.assertRaises(ValueError):
            TrainConfig(steps=10, warmup_steps=10)
        with self.assertRaises(ValueError):
            DecoderConfig(base_channels=10)


if __name__ == "__main__":
    unittest.main()
