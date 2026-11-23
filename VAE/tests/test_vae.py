"""Tests use only small synthetic local data; no remote training or downloads."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from VAE.data import GlyphStore, ReconstructionDataset
from VAE.model import GlyphVAE, VAEConfig, vae_loss
from VAE.train import TrainConfig, evaluate, run


def fixture(root, standards=True):
    root.mkdir()
    writers = []
    for i, split in enumerate(("train", "train", "val")):
        images = np.full((3, 16, 16), 220 - 50 * i, dtype=np.uint8)
        for c in range(3):
            images[c, 2+c:10+c, 4:7] = 20
        np.save(root / f"{i}.npy", images)
        writers.append({"id": str(i), "file": f"{i}.npy", "labels": [0, 1, 2], "split": split})
    if standards:
        np.save(root / "standards.npy", np.full((3, 16, 16), 255, dtype=np.uint8))
    (root / "manifest.json").write_text(json.dumps({"version": 1, "image_size": 16,
        "characters": ["甲", "乙", "丙"], "writers": writers}), encoding="utf-8")


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GlyphVAE(VAEConfig(image_size=16, base_channels=8, latent_channels=2))
        self.x = torch.rand(4, 1, 16, 16) * 2 - 1

    def test_shapes_and_mean_reconstruction_are_deterministic(self):
        a, mu, lv = self.model(self.x, sample=False)
        b, _, _ = self.model(self.x, sample=False)
        self.assertEqual(mu.shape, (4, 2, 2, 2))
        self.assertEqual(lv.shape, mu.shape)
        self.assertEqual(a.shape, self.x.shape)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertLessEqual(a.abs().max().item(), 1)

    def test_reparameterization_noise_and_gradient(self):
        mu = torch.randn(3, 2, requires_grad=True)
        lv = torch.zeros(3, 2, requires_grad=True)
        z = self.model.reparameterize(mu, lv, noise=torch.ones_like(mu))
        torch.testing.assert_close(z, mu + 1)
        z.sum().backward()
        torch.testing.assert_close(mu.grad, torch.ones_like(mu))
        torch.testing.assert_close(lv.grad, torch.full_like(lv, .5))

    def test_kl_matches_analytic_values_and_reductions(self):
        mu = torch.zeros(2, 4, 3, 3)
        lv = torch.zeros_like(mu)
        terms = vae_loss(self.x, self.x, mu, lv, .25)
        self.assertEqual(terms["loss"].item(), 0)
        mu.fill_(1)
        terms = vae_loss(self.x, self.x, mu, lv, .25)
        self.assertEqual(terms["kl"].item(), .5)
        self.assertEqual(terms["kl_per_image"].item(), 18)
        self.assertEqual(terms["loss"].item(), .125)

    def test_all_gradients_finite_including_variance(self):
        y, mu, lv = self.model(self.x)
        loss = vae_loss(y, self.x, mu, lv, 1e-4)["loss"]
        loss.backward()
        total = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e6, error_if_nonfinite=True)
        self.assertGreater(total, 0)
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(self.model.encoder[-1].weight.grad[2:].abs().sum(), 0)

    def test_stochastic_small_batch_reconstruction_improves(self):
        # Fixed simple strokes, independent posterior draws each update.
        x = torch.ones(4, 1, 16, 16)
        for i in range(4):
            x[i, :, 2+i:10+i, 3:6] = -1
        opt = torch.optim.Adam(self.model.parameters(), lr=.002)
        with torch.no_grad():
            initial = ((self.model(x, sample=False)[0] - x)**2).mean().item()
        largest = 0
        for _ in range(100):
            opt.zero_grad(set_to_none=True)
            y, mu, lv = self.model(x)
            terms = vae_loss(y, x, mu, lv, 1e-4)
            terms["loss"].backward()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, error_if_nonfinite=True)
            largest = max(largest, grad.item())
            opt.step()
        with torch.no_grad():
            final = ((self.model(x, sample=False)[0] - x)**2).mean().item()
        print(f"local stochastic VAE overfit: MSE {initial:.5f} -> {final:.5f}; max raw grad {largest:.3f}")
        self.assertLess(final, initial * .25)

    def test_wrong_shapes_rejected(self):
        with self.assertRaises(ValueError): self.model(torch.zeros(1, 1, 15, 16))
        with self.assertRaises(ValueError): self.model.decode(torch.zeros(1, 2, 3, 3))


class StoreTest(unittest.TestCase):
    def open_store(self, root):
        store = GlyphStore(root)
        self.addCleanup(store.close)
        return store


class DataTests(StoreTest):
    def test_zero_ratio_never_reads_standards_and_stays_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"; fixture(root, standards=False)
            store = self.open_store(root)
            ds = ReconstructionDataset(store, standard_ratio=0, length=200)
            for i in range(len(ds)):
                row = ds[i]
                self.assertFalse(row["is_standard"])
                self.assertIn(row["writer_id"], ("0", "1"))
                self.assertEqual(row["image"].dtype, torch.uint8)
            self.assertIsNone(store._standards)
            store.close()

    def test_mixture_deterministic_proportions_and_boundary_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"; fixture(root)
            store = self.open_store(root)
            pure = ReconstructionDataset(store, length=1000, standard_ratio=0)
            mixed = ReconstructionDataset(store, length=1000, standard_ratio=.3)
            standard = ReconstructionDataset(store, length=20, standard_ratio=1)
            fraction = sum(mixed[i]["is_standard"] for i in range(len(mixed))) / len(mixed)
            self.assertLess(abs(fraction - .3), .04)
            for i in range(30):
                torch.testing.assert_close(mixed[i]["image"], mixed[i]["image"])
                if not mixed[i]["is_standard"]:
                    torch.testing.assert_close(mixed[i]["image"], pure[i]["image"])
            self.assertTrue(all(standard[i]["is_standard"] for i in range(len(standard))))
            val = ReconstructionDataset(store, split="val", source="handwriting")
            self.assertEqual(len(val), 3)
            self.assertTrue(all(val[i]["writer_id"] == "2" for i in range(3)))
            store.close()

    def test_out_of_range_ratio_and_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"; fixture(root)
            store = self.open_store(root)
            for value in (-.1, 1.1, float('nan')):
                with self.assertRaises(ValueError):
                    ReconstructionDataset(store, standard_ratio=value, length=10)
            with self.assertRaises(ValueError): store._safe_path("../other.npy")
            store.close()


class TrainingTests(StoreTest):
    def test_validation_restores_rng_mode_and_counts_all_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"; fixture(root)
            model = GlyphVAE(VAEConfig(image_size=16, base_channels=8, latent_channels=2)).train()
            store = self.open_store(root)
            ds = ReconstructionDataset(store, split="val", source="handwriting")
            loader = DataLoader(ds, batch_size=2)
            state = torch.get_rng_state().clone()
            a = evaluate(model, loader, torch.device('cpu'), 'fp32', 1e-4, seed=3)
            torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
            b = evaluate(model, loader, torch.device('cpu'), 'fp32', 1e-4, seed=3)
            self.assertEqual(a, b)
            self.assertTrue(model.training)
            self.assertEqual(a['samples'], 3)
            store.close()

    def test_train_outputs_and_exact_cpu_resume(self):
        from VAE.train import atomic_save
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root / 'data')
            cfg = TrainConfig(steps=2, batch_size=2, standard_ratio=.4, warmup_steps=0,
                kl_warmup_steps=0, validate_every=1, save_every=1, log_every=1,
                val_samples=0, precision='fp32')
            args = SimpleNamespace(**asdict(cfg), data=root/'data', output=root/'run',
                                   base_channels=8, latent_channels=2, device='cpu', resume=None)
            backup = root / 'step1.pt'
            def save_and_copy(state, path):
                atomic_save(state, path)
                if state['step'] == 1: atomic_save(state, backup)
            with patch('VAE.train.atomic_save', side_effect=save_and_copy): run(args)
            full = torch.load(args.output/'last.pt', weights_only=False)
            rows = [json.loads(x) for x in (args.output/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['val_handwriting']['samples'], 3)
            self.assertEqual(rows[-1]['val_standard']['samples'], 3)
            self.assertTrue((args.output/'previews'/'standard_0000002.png').exists())
            self.assertTrue((args.output/'best.pt').exists())
            # A separate temporary directory holds the same run at step 1.
            args.output = root / 'resumed'; args.output.mkdir()
            (args.output/'run.json').write_text((root/'run'/'run.json').read_text())
            args.resume = backup
            run(args)
            resumed = torch.load(args.output/'last.pt', weights_only=False)
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
