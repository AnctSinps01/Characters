"""Prepared-data integration checks for full-dataset same-character training."""
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from twist import LossConfig, distance_fields
from twist.checkpoint import restore_optimizer
from twist.images import NORMALIZATION
from pairs import PreparedPairs, ShuffledPairSampler
from train import parse_args, train
from settings import TrainingConfig


def fixture(root):
    root.mkdir()
    (root / "writers").mkdir()
    writers = []
    for writer, labels in enumerate(([0, 1, 2], [1, 0], [0, 1], [1, 0, 2], [0, 1])):
        images = []
        for label in labels:
            image = np.full((16, 16), 255, dtype=np.uint8)
            image[3:13, 6:9] = 0
            if label == 0:
                image[7:9, 3:13] = 0
            if label == 2:
                image[3:5, 3:13] = 0
            images.append(np.roll(image, writer % 3 - 1, axis=1))
        file = f"writers/{writer:05d}.npy"
        np.save(root / file, np.stack(images))
        writers.append({"id": str(writer), "file": file, "labels": labels,
                        "split": "train" if writer < 3 else "val"})
    (root / "manifest.json").write_text(json.dumps({"version": 1, "image_size": 16,
                  "characters": list("中十独"), "writers": writers}), encoding="utf-8")


def prepared(root, cache_size=4):
    return PreparedPairs(root, image_size=16, threshold=0, distance_cache_size=cache_size, verbose=False)


class PairTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_pairs_have_the_same_label_and_distinct_writers_without_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture(root)
            data = prepared(root)
            self.assertEqual(data.count, 12)
            self.assertEqual(data.pixels.dtype, torch.uint8)
            self.assertEqual(data.skipped, {"train": 1, "val": 1})
            self.assertEqual(len(data.records["train"]), 6)
            self.assertEqual(len(data.records["val"]), 4)
            generator = torch.Generator().manual_seed(42)
            for split in ("train", "val"):
                sources = data.records[split].repeat(20)
                pairs = data.match(sources, generator)
                torch.testing.assert_close(data.labels[pairs[:, 0]], data.labels[pairs[:, 1]])
                self.assertTrue(bool((data.writer_numbers[pairs[:, 0]] != data.writer_numbers[pairs[:, 1]]).all()))
                self.assertTrue(set(pairs.flatten().tolist()).issubset(set(data.records[split].tolist())))

    def test_shuffled_epochs_cover_every_source_and_restore_partner_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture(root)
            data = prepared(root)
            sampler = ShuffledPairSampler(data, 91)
            first = sampler.next(4)
            restored = ShuffledPairSampler(data, 91, sampler.state_dict())
            changed_seed = ShuffledPairSampler(data, 92, sampler.state_dict())
            tail = sampler.next(4)
            self.assertEqual(len(tail), 2)
            torch.testing.assert_close(tail, restored.next(4), rtol=0, atol=0)
            torch.testing.assert_close(tail, changed_seed.next(4), rtol=0, atol=0)
            self.assertEqual(set(torch.cat((first, tail))[:, 0].tolist()), set(data.records["train"].tolist()))
            torch.testing.assert_close(sampler.next(4), restored.next(4), rtol=0, atol=0)
            self.assertGreater(sampler.epoch_progress, 1)
            new_epoch = changed_seed.next(len(data.records["train"]))
            self.assertEqual(set(new_epoch[:, 0].tolist()), set(data.records["train"].tolist()))

    def test_faint_images_normalize_independently_and_have_foreground(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture(root)
            for writer, path in enumerate(sorted((root / "writers").glob("*.npy"))):
                pixels = np.load(path)
                # 全部原始像素 > 127.5；不同 writer 有不同亮度与对比度。
                np.save(path, np.where(pixels == 0, 190 + writer * 5, 250 + writer).astype(np.uint8))
            data = prepared(root)
            generator = torch.Generator().manual_seed(12)
            for split in ("train", "val"):
                a, b, distance_a, distance_b = data.batch(data.match(data.records[split], generator), "cpu")
                for image in (a, b):
                    torch.testing.assert_close(image.amin(dim=(1, 2, 3)), -torch.ones(len(image)))
                    torch.testing.assert_close(image.amax(dim=(1, 2, 3)), torch.ones(len(image)))
                torch.testing.assert_close(distance_a, distance_fields(a, threshold=0))
                torch.testing.assert_close(distance_b, distance_fields(b, threshold=0))
                self.assertTrue(bool((distance_a > 0).any()))
                self.assertTrue(bool((distance_b > 0).any()))

    def test_minmax_is_applied_after_resizing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture(root)
            path = root / "writers" / "00000.npy"
            pixels = np.load(path)
            pixels[0] = 255
            pixels[0, 7, 7] = 200
            np.save(path, pixels)
            data = PreparedPairs(root, image_size=8, threshold=0, verbose=False)
            image = data.normalized(torch.tensor([0]), "cpu")
            self.assertEqual(tuple(image.shape), (1, 1, 8, 8))
            self.assertEqual(float(image.min()), -1)
            self.assertEqual(float(image.max()), 1)
            self.assertTrue(bool(torch.isfinite(distance_fields(image)).all()))

    def test_constant_images_are_excluded_from_both_sides_before_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture(root)
            for writer, position, value in ((0, 2, 255), (1, 1, 200), (3, 0, 0)):
                path = root / "writers" / f"{writer:05d}.npy"
                pixels = np.load(path)
                pixels[position] = value
                np.save(path, pixels)
            data = prepared(root)
            self.assertEqual(data.constant_skipped, {"train": 2, "val": 1})
            self.assertEqual(data.skipped, {"train": 0, "val": 2})
            self.assertEqual([image["index"] for image in data.invalid_images], [2, 4, 7])
            self.assertEqual(data.invalid_images[1]["writer"], "1")
            self.assertEqual(data.invalid_images[1]["character"], "中")
            self.assertEqual(data.invalid_images[1]["pixel_min"], 200)
            self.assertFalse(set(data.partners.tolist()) & {2, 4, 7})
            generator = torch.Generator().manual_seed(12)
            for split in ("train", "val"):
                pairs = data.match(data.records[split].repeat(10), generator)
                self.assertFalse(set(pairs.flatten().tolist()) & {2, 4, 7})
                data.batch(pairs, "cpu")

    def test_resume_rejects_old_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old.pt"
            torch.save({"kind": "twist-all-pair", "version": 3}, checkpoint)
            with self.assertRaisesRegex(ValueError, "different normalization"):
                self.run_training(["--resume", checkpoint, "--device", "cpu"])

    def test_default_training_uses_three_losses_and_rejects_displacement_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            with patch("pairs.distance_fields", wraps=distance_fields) as transform:
                self.run_training([*self.common(root), "--output", root / "run", "--stop-after", 1])
                self.assertGreater(transform.call_count, 0)
            run = json.loads((root / "run" / "run.json").read_text())
            self.assertEqual(run["loss"]["similarity_weight"], 1)
            self.assertEqual(run["loss"]["distance_weight"], 1)
            self.assertNotIn("folding_weight", run["loss"])
            saved = torch.load(root / "run" / "last.pt", weights_only=True)
            values = saved["validation"]
            self.assertGreater(values["forward_distance"], 0)
            self.assertGreater(values["reverse_distance"], 0)
            self.assertAlmostEqual(values["distance"], values["forward_distance"] + values["reverse_distance"], places=6)
            self.assertAlmostEqual(values["total"], values["similarity"] + values["distance"]
                                   + 0.001*values["smoothness"], places=6)
            checkpoint = root / "old.pt"
            for version in (1, 2):
                torch.save({"kind": "twist-all-pair", "version": version}, checkpoint)
                with self.assertRaisesRegex(ValueError, "pixel-resolution stationary-velocity"):
                    self.run_training(["--resume", checkpoint, "--device", "cpu"])

    def test_preloaded_pixels_and_bounded_distance_cache_do_not_reopen_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture(root)
            data = prepared(root, cache_size=2)
            generator = torch.Generator().manual_seed(12)
            pair = data.match(data.records["train"][:1], generator)
            (root / "writers").rename(root / "moved")
            with patch("pairs.distance_fields", wraps=distance_fields) as transform:
                a, b, distance_a, distance_b = data.batch(pair, "cpu")
                data.batch(pair, "cpu")
                data.batch(pair.flip(1), "cpu")
                self.assertEqual(transform.call_count, 1)
                self.assertEqual(transform.call_args.args[0].shape[0], 2)
                torch.testing.assert_close(distance_a, distance_fields(a))
                torch.testing.assert_close(distance_b, distance_fields(b))
                more = data.match(data.records["train"][:3], generator)
                data.batch(more, "cpu")
            self.assertEqual(len(data.distance_cache), 2)
            for field in data.distance_cache.values():
                self.assertEqual(field.untyped_storage().nbytes(), 16 * 16 * 4)
            self.assertEqual(a.shape, b.shape)

    def test_small_cache_deduplicates_both_roles_without_losing_batch_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture(root)
            data = prepared(root, cache_size=1)
            generator = torch.Generator().manual_seed(12)
            pair = data.match(data.records["train"][:1], generator)
            pairs = torch.cat((pair, pair.flip(1), pair))
            with patch("pairs.distance_fields", wraps=distance_fields) as transform:
                a, b, distance_a, distance_b = data.batch(pairs, "cpu")
                self.assertEqual(transform.call_count, 1)
                self.assertEqual(transform.call_args.args[0].shape[0], 2)
            self.assertEqual(len(data.distance_cache), 1)
            torch.testing.assert_close(distance_a, distance_fields(a))
            torch.testing.assert_close(distance_b, distance_fields(b))
            _, _, no_a, no_b = data.batch(pairs, "cpu", with_distance=False)
            self.assertIsNone(no_a)
            self.assertIsNone(no_b)

    def run_training(self, arguments):
        with patch.object(sys, "argv", ["train.py", *map(str, arguments)]), redirect_stdout(io.StringIO()):
            train(parse_args())

    def common(self, root):
        return ["--data", root / "data", "--device", "cpu", "--precision", "fp32", "--steps", 5,
                "--image-size", 16, "--patch-size", 4, "--token-dim", 16, "--num-heads", 4,
                "--num-blocks", 2, "--squaring-steps", 5, "--batch-size", 4,
                "--log-every", 1, "--validate-every", 2, "--save-every", 2,
                "--val-batches", 2, "--preview-samples", 2, "--animation-frames", 3]

    def test_short_run_outputs_and_resume_match_continuous_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            common = self.common(root)
            self.run_training([*common, "--output", root / "whole"])
            self.run_training([*common, "--output", root / "split", "--stop-after", 3])
            summary = json.loads((root / "split" / "summary.json").read_text())
            self.assertEqual(summary["status"], "stopped_early")
            self.assertEqual(summary["step"], 3)
            self.run_training([*common, "--resume", root / "split" / "last.pt"])
            whole = torch.load(root / "whole" / "last.pt", weights_only=True)
            resumed = torch.load(root / "split" / "last.pt", weights_only=True)
            self.assertEqual(resumed["step"], 5)
            self.assertEqual(resumed["version"], 3)
            self.assertEqual(resumed["model"]["network.velocity_head.weight"].shape, (32, 16))
            for name, value in whole["model"].items():
                torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(whole["validation_pairs"], resumed["validation_pairs"], rtol=0, atol=0)
            self.assertNotIn("pixels", resumed)
            self.assertNotIn("distance_cache", resumed)
            self.assertNotIn("training_config", resumed)
            self.assertNotIn("loss_config", resumed)
            self.assertEqual(set(resumed["optimizer"]["param_groups"][0]), {"params"})
            self.assertEqual(resumed["normalization"], NORMALIZATION)
            self.assertEqual(json.loads((root / "split" / "excluded_images.json").read_text()), [])
            self.assertEqual(resumed["validation"]["count"], 4)
            self.assertEqual(resumed["validation"]["evaluated_step"], 5)
            self.assertGreater(resumed["validation"]["velocity_max"], 0)
            self.assertGreater(resumed["validation"]["jacobian_min"], 0)
            self.assertEqual(resumed["validation"]["folding_fraction"], 0)
            self.assertEqual(resumed["model_config"]["num_blocks"], 2)
            self.assertEqual(resumed["model_config"]["squaring_steps"], 5)
            descriptions = json.loads((root / "split" / "preview_pairs.json").read_text(encoding="utf-8"))
            self.assertEqual(len(descriptions), 2)
            with Image.open(next((root / "split" / "previews").glob("step_0000005_*_trajectory.gif"))) as gif:
                self.assertEqual(gif.n_frames, 3)
            self.assertFalse(list((root / "split").rglob("*.png")))
            manifest = json.loads((root / "data" / "manifest.json").read_text())
            manifest["changed"] = True
            (root / "data" / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                self.run_training(["--data", root / "data", "--device", "cpu", "--resume", root / "split" / "last.pt"])

    def test_resume_rejects_changed_eligible_source_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            self.run_training([*self.common(root), "--output", root / "run", "--stop-after", 1])
            path = root / "data" / "writers" / "00000.npy"
            pixels = np.load(path)
            pixels[0] = 255
            np.save(path, pixels)
            with self.assertRaisesRegex(ValueError, "pairs differ"):
                self.run_training(["--data", root / "data", "--device", "cpu", "--resume", root / "run" / "last.pt"])

    def test_resume_uses_live_config_and_preserves_state_without_saving_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            self.run_training([*self.common(root), "--output", root / "run", "--stop-after", 1])
            checkpoint_path = root / "run" / "last.pt"
            before = torch.load(checkpoint_path, weights_only=True)
            # 旧格式带有训练配置和优化器 lr；这些不能覆盖当前配置。
            legacy = {**before, "training_config": {"steps": 99999, "batch_size": 99},
                      "loss_config": {"mse_weight": 0}}
            legacy["optimizer"] = {"state": before["optimizer"]["state"],
                                   "param_groups": [{**group, "lr": 9.0}
                                                    for group in before["optimizer"]["param_groups"]]}
            torch.save(legacy, checkpoint_path)
            resume = ["--data", root / "data", "--device", "cpu", "--resume", checkpoint_path]
            @dataclass(frozen=True)
            class CurrentTraining(TrainingConfig):
                steps: int = 1
                batch_size: int = 2
                learning_rate: float = 2e-4
                val_batches: int = 1
                animation_frames: int = 3
            @dataclass(frozen=True)
            class CurrentLoss(LossConfig):
                similarity_weight: float = 10
                distance_weight: float = 20
                smoothness_weight: float = 0.001
                mse_uniform_ratio: float = 0.2
            # 在同一步重新评估，以验证仅改 loss 时模型、优化器与配对状态保持原值。
            with patch("train.TrainingConfig", CurrentTraining), patch("train.LossConfig", CurrentLoss), \
                    patch("train.restore_optimizer", wraps=restore_optimizer) as restored:
                self.run_training(resume)
                self.assertEqual(restored.call_args.args[0].param_groups[0]["lr"], 2e-4)
            after = torch.load(checkpoint_path, weights_only=True)
            best = torch.load(root / "run" / "best.pt", weights_only=True)
            for key in ("model", "optimizer", "sampler"):
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
            self.assertNotIn("loss_config", after)
            self.assertNotIn("training_config", after)
            self.assertNotIn("loss_config", best)
            self.assertEqual(set(after["optimizer"]["param_groups"][0]), {"params"})
            self.assertEqual(after["model_config"], before["model_config"])
            run = json.loads((root / "run" / "run.json").read_text())
            self.assertEqual(run["training"]["batch_size"], 2)
            self.assertEqual(run["loss"]["mse_uniform_ratio"], 0.2)
            self.assertEqual(after["validation"]["count"], 2)
            validation = after["validation"]
            expected = 10 * validation["similarity"] + 20 * validation["distance"] + 0.001 * validation["smoothness"]
            self.assertAlmostEqual(validation["total"], expected, places=5)
            self.assertAlmostEqual(best["best_total"], best["validation"]["total"], places=7)
            self.assertLessEqual(best["best_total"], validation["total"] + 1e-7)
            self.assertGreater(best["best_total"], before["best_total"])
            with patch("train.TrainingConfig", CurrentTraining), patch("train.LossConfig", CurrentLoss):
                self.run_training([*resume, "--steps", 2])
            continued = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(continued["step"], 2)
            self.assertTrue(all(float(state["step"]) == 2 for state in continued["optimizer"]["state"].values()))
            records = [json.loads(line) for line in (root / "run" / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["loss_config"], run["loss"])

    def test_interrupt_finishes_one_update_and_saves_sampler_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "data")
            original_step = torch.optim.Adam.step
            def interrupted(optimizer, *args, **kwargs):
                signal.raise_signal(signal.SIGINT)
                return original_step(optimizer, *args, **kwargs)
            with patch.object(torch.optim.Adam, "step", interrupted):
                self.run_training([*self.common(root), "--output", root / "run"])
            saved = torch.load(root / "run" / "last.pt", weights_only=True)
            summary = json.loads((root / "run" / "summary.json").read_text())
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(saved["step"], 1)
            self.assertEqual(saved["sampler"]["offset"], 4)
            self.assertEqual(saved["validation"]["evaluated_step"], 1)
            self.assertTrue(all(float(state["step"]) == 1 for state in saved["optimizer"]["state"].values()))


if __name__ == "__main__":
    unittest.main()
