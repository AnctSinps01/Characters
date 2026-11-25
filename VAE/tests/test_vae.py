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

    def test_downsample_variants_change_latent_shape(self):
        coarse = GlyphVAE(VAEConfig(image_size=64, base_channels=8, latent_channels=4,
                                    latent_downsample=16))
        y, mu, _ = coarse(torch.randn(2, 1, 64, 64))
        self.assertEqual(tuple(mu.shape[1:]), (4, 4, 4))
        self.assertEqual(tuple(y.shape), (2, 1, 64, 64))
        narrow = GlyphVAE(VAEConfig(image_size=64, base_channels=8, latent_channels=2))
        self.assertEqual(tuple(narrow.encode(torch.randn(1, 1, 64, 64))[0].shape[1:]), (2, 8, 8))
        with self.assertRaises(ValueError):
            VAEConfig(latent_downsample=12)

    def test_default_architecture_width_is_unchanged(self):
        model = GlyphVAE(VAEConfig())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8157577)
        self.assertEqual(model.config.latent_shape, (4, 8, 8))


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


class DualBlockTests(unittest.TestCase):
    def test_grid_sampler_is_a_full_grid(self):
        from VAE.train import GridBatchSampler
        table = np.arange(8 * 5).reshape(8, 5)          # 8 chars x 5 authors
        sampler = GridBatchSampler(table, list(range(8)), list(range(5)), 3, 0, 2, seed=11)
        batches = list(sampler)
        self.assertEqual([len(b) for b in batches], [9, 9])
        for batch in batches:
            pairs = [divmod(int(v), 5) for v in batch]   # (char, author) from the table value
            chars = [c for c, _ in pairs]
            authors = [a for _, a in pairs]
            self.assertEqual(len(set(chars)), 3)
            self.assertEqual(len(set(authors)), 3)
            self.assertEqual(sorted(chars.count(c) for c in set(chars)), [3, 3, 3])
            self.assertEqual(sorted(authors.count(a) for a in set(authors)), [3, 3, 3])
        self.assertEqual(batches, list(GridBatchSampler(table, list(range(8)), list(range(5)),
                                                        3, 0, 2, seed=11)))
        # Resume starts at the requested step.
        self.assertEqual(list(GridBatchSampler(table, list(range(8)), list(range(5)),
                                               3, 1, 2, seed=11))[0], batches[1])

    def test_dual_config_rules(self):
        from VAE.train import TrainConfig
        base = dict(char_cos_weight=1e-3, contrast_key="dual", groups_per_batch=4,
                    batch_size=16, content_channels=2, author_channels=1)
        TrainConfig(**base)
        with self.assertRaises(ValueError):
            TrainConfig(**{**base, "groups_per_batch": 3})     # 3*3 != 16
        with self.assertRaises(ValueError):
            TrainConfig(**{**base, "author_channels": 0})
        with self.assertRaises(ValueError):
            TrainConfig(**{**base, "standard_ratio": 0.1})
        with self.assertRaises(ValueError):
            TrainConfig(**{**base, "contrast_key": "char", "groups_per_batch": 4,
                           "members_per_group": 4})            # channel split needs dual

    def test_pair_table_matches_dataset_order(self):
        from VAE.train import pair_table
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "case"
            fixture(root)
            store = GlyphStore(root)
            try:
                table = pair_table(store, 'train')
                self.assertEqual(table.shape[0], len(store.characters))
                self.assertEqual(table.shape[1], 2)          # two train writers in the fixture
                self.assertGreaterEqual(table.min(), -1)
                # Every non-negative entry indexes a (writer, row) that exists.
                for wi, writer in enumerate([w for w in store.writers if w['split'] == 'train']):
                    for row, label in enumerate(writer['labels']):
                        self.assertEqual(table[label, wi] >= 0, True)
            finally:
                store.close()


class CharCosineTests(unittest.TestCase):
    def test_contrastive_loss_prefers_separation(self):
        from VAE.train import char_cosine_terms
        labels = torch.tensor([0, 0, 1, 1])
        separated = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        low, positive, negative = char_cosine_terms(separated, labels, temperature=0.1)
        self.assertGreater(positive.item(), 0.99)
        self.assertLess(negative.item(), 0.01)
        self.assertGreater(positive.item() - negative.item(), 0.9)
        collapsed = torch.ones(4, 2)
        high, collapse_pos, collapse_neg = char_cosine_terms(collapsed, labels, temperature=0.1)
        self.assertGreater(high.item(), low.item())
        self.assertGreater(collapse_neg.item(), 0.99)   # negatives indistinguishable
        self.assertAlmostEqual(collapse_pos.item() - collapse_neg.item(), 0.0, places=5)
        # A batch with no negatives carries no gradient: that is expected.
        single = torch.tensor([[1.0, 0.0], [0.3, 0.9]], requires_grad=True)
        loss, _, _ = char_cosine_terms(single, torch.tensor([0, 0]))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(single.grad.abs().sum().item(), 0.0)
        # With both classes present the gradient reaches the mean.
        mean = torch.tensor([[1.0, 0.0], [0.3, 0.9], [0.2, 0.1], [0.9, 0.4]],
                            requires_grad=True)
        char_cosine_terms(mean, labels)[0].backward()
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertGreater(mean.grad.abs().sum().item(), 0.0)
        # No same-class pair -> zeros, not an error.
        zero_loss, zero_pos, zero_neg = char_cosine_terms(torch.eye(3), torch.tensor([0, 1, 2]))
        self.assertEqual((zero_loss.item(), zero_pos.item(), zero_neg.item()), (0.0, 0.0, 0.0))

    def test_grouped_batch_sampler(self):
        from VAE.train import GroupedBatchSampler
        class_indices = {c: [c * 10 + i for i in range(5)] for c in range(8)}
        sampler = GroupedBatchSampler(class_indices, 4, 3, 0, 3, seed=7)
        batches = list(sampler)
        self.assertEqual([len(b) for b in batches], [12, 12, 12])
        for batch in batches:
            classes = [index // 10 for index in batch]
            counts = {c: classes.count(c) for c in set(classes)}
            self.assertEqual(len(counts), 4)
            self.assertTrue(all(v == 3 for v in counts.values()))
            self.assertEqual(len(set(batch)), len(batch))
        self.assertEqual(batches, list(GroupedBatchSampler(class_indices, 4, 3, 0, 3, seed=7)))
        self.assertEqual(len(list(GroupedBatchSampler(class_indices, 4, 3, 2, 3, seed=7))), 1)
        self.assertEqual(list(GroupedBatchSampler(class_indices, 4, 3, 2, 3, seed=7))[0], batches[2])

    def test_char_cosine_config_rules(self):
        from VAE.train import TrainConfig, schedule
        base = dict(char_cos_weight=3e-4, groups_per_batch=8, members_per_group=8, batch_size=64)
        config = TrainConfig(**base)
        self.assertAlmostEqual(schedule(config, config.char_cos_warmup)[2], 3e-4)
        self.assertAlmostEqual(schedule(config, 0)[2], 0.0)
        with self.assertRaises(ValueError):
            TrainConfig(char_cos_weight=3e-4)
        with self.assertRaises(ValueError):
            TrainConfig(**{**base, "members_per_group": 4})
        with self.assertRaises(ValueError):
            TrainConfig(**{**base, "standard_ratio": 0.1})
        with self.assertRaises(ValueError):
            TrainConfig(**{**base, "contrast_key": "painter"})
        self.assertEqual(TrainConfig(char_cos_weight=0.0).char_cos_weight, 0.0)
        author = TrainConfig(**{**base, "contrast_key": "author"})
        self.assertEqual(author.contrast_key, "author")


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
                                   base_channels=8, latent_channels=2,
                                   latent_downsample=8, device='cpu', resume=None)
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
