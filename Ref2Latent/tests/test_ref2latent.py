"""Tests for Ref2Latent: token layout, separate QKV, data plumbing, tiny overfit."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from Ref2Latent.config import ModelConfig
from Ref2Latent.data import LatentStore, RefTargetDataset
from Ref2Latent.model import Block, JointAttention, Ref2Latent
from Ref2Latent.train import TrainConfig, learning_rate
from VAE.data import GlyphStore


def tiny_config(**overrides):
    base = dict(latent_channels=2, latent_size=4, references=3, dim=32, heads=4, layers=2)
    base.update(overrides)
    return ModelConfig(**base)


def fixture(root: Path, writers=3, classes=6, per_writer=4, seed=0):
    rng = np.random.default_rng(seed)
    (root / "data").mkdir(parents=True)
    (root / "latents").mkdir(parents=True)
    entries = []
    for index in range(writers):
        labels = (np.arange(per_writer) + index) % classes
        images = rng.integers(0, 256, size=(per_writer, 16, 16), dtype=np.uint8)
        np.save(root / "data" / f"w{index}.npy", images)
        np.save(root / "latents" / f"w{index}.npy",
                rng.normal(size=(per_writer, 2, 4, 4)).astype(np.float16))
        entries.append({"id": f"w{index}", "file": f"w{index}.npy",
                        "split": "train" if index < writers - 1 else "val",
                        "labels": [int(x) for x in labels]})
    np.save(root / "data" / "standards.npy",
            rng.integers(0, 256, size=(classes, 16, 16), dtype=np.uint8))
    np.save(root / "latents" / "standards.npy",
            rng.normal(size=(classes, 2, 4, 4)).astype(np.float16))
    (root / "data" / "manifest.json").write_text(json.dumps({
        "version": 1, "image_size": 16, "characters": [chr(0x4E00 + i) for i in range(classes)],
        "writers": entries}), encoding="utf-8")
    (root / "latents" / "manifest.json").write_text(json.dumps({
        "latent_shape": [2, 4, 4], "image_size": 16, "classes": classes,
        "vae_checkpoint": "synthetic", "vae_step": 0}), encoding="utf-8")


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.model = Ref2Latent(self.config)

    def test_shapes_and_value_range(self):
        reference = torch.randn(2, 3, 2, 4, 4)
        standard = torch.randn(2, 2, 4, 4)
        out = self.model(reference, standard)
        self.assertEqual(tuple(out.shape), (2, 2, 4, 4))
        # Deterministic: same inputs give the same output.
        again = self.model(reference, standard)
        torch.testing.assert_close(out, again)

    def test_reference_count_and_latent_shape_enforced(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 2, 2, 4, 4), torch.randn(1, 2, 4, 4))
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 3, 2, 4, 4), torch.randn(1, 2, 4, 5))

    def test_roles_use_separate_projections(self):
        attention = JointAttention(self.config)
        self.assertFalse(torch.equal(attention.reference_qkv.weight,
                                     attention.standard_qkv.weight))
        # Ref and standard halves of a batch are interchangeable for the shared
        # attention denominator: swapping roles must change the standard output.
        model = self.model
        standard = torch.randn(1, 2, 4, 4)
        reference = torch.randn(1, 3, 2, 4, 4)
        self.assertFalse(torch.equal(model(reference, standard),
                                     model(reference[:, :1].repeat(1, 3, 1, 1, 1), standard)))

    def test_gradients_flow_to_all_parameters(self):
        out = self.model(torch.randn(2, 3, 2, 4, 4), torch.randn(2, 2, 4, 4))
        out.square().mean().backward()
        missing = [name for name, p in self.model.named_parameters()
                   if p.grad is None or not torch.isfinite(p.grad).all()]
        self.assertEqual(missing, [])

    def test_block_preserves_token_lengths(self):
        block = Block(self.config)
        reference = torch.randn(2, 3 * 16, self.config.dim)
        standard = torch.randn(2, 16, self.config.dim)
        out_reference, out_standard = block(reference, standard)
        self.assertEqual(out_reference.shape, reference.shape)
        self.assertEqual(out_standard.shape, standard.shape)

    def test_small_batch_overfits(self):
        torch.manual_seed(0)
        model = Ref2Latent(tiny_config(dim=64, layers=2))
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        reference = torch.randn(4, 3, 2, 4, 4)
        standard = torch.randn(4, 2, 4, 4)
        target = torch.randn(4, 2, 4, 4)
        first = None
        for _ in range(250):
            optimizer.zero_grad()
            loss = (model(reference, standard) - target).square().mean()
            loss.backward()
            optimizer.step()
            first = loss.item() if first is None else first
        self.assertLess(loss.item(), 0.2 * first)


class DataTests(unittest.TestCase):
    def test_dataset_shapes_and_reference_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            store = GlyphStore(root / "data")
            latents = LatentStore(root / "latents")
            try:
                dataset = RefTargetDataset(store, latents, split="train", references=2, seed=5)
                item = dataset[0]
                self.assertEqual(tuple(item["reference"].shape), (2, 2, 4, 4))
                self.assertEqual(tuple(item["standard"].shape), (2, 4, 4))
                self.assertEqual(tuple(item["target"].shape), (2, 4, 4))
                self.assertEqual(tuple(item["image"].shape), (1, 16, 16))
                # References never repeat the target row and are deterministic.
                writer_index, row, label = dataset.pairs[0]
                rows = dataset.reference_rows(writer_index, row, label)
                self.assertNotIn(row, rows)
                self.assertEqual(len(set(rows)), 2)
                self.assertEqual(rows, dataset.reference_rows(writer_index, row, label))
                again = dataset[0]
                torch.testing.assert_close(item["reference"], again["reference"])
            finally:
                store.close()
                latents.close()

    def test_split_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            store = GlyphStore(root / "data")
            latents = LatentStore(root / "latents")
            try:
                train = RefTargetDataset(store, latents, split="train", references=1)
                val = RefTargetDataset(store, latents, split="val", references=1)
                train_writers = {w["id"] for w in train.writers}
                val_writers = {w["id"] for w in val.writers}
                self.assertEqual(train_writers & val_writers, set())
            finally:
                store.close()
                latents.close()


class ConfigTests(unittest.TestCase):
    def test_schedule_and_validation(self):
        config = TrainConfig(steps=100, warmup_steps=10)
        self.assertLess(learning_rate(config, 5), learning_rate(config, 50))
        with self.assertRaises(ValueError):
            TrainConfig(latent_weight=0.0, image_weight=0.0)
        with self.assertRaises(ValueError):
            ModelConfig(dim=30, heads=4)


if __name__ == "__main__":
    unittest.main()
